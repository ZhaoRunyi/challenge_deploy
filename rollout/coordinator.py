from __future__ import annotations

from dataclasses import dataclass, replace
import queue
import threading
import time
from typing import Any, Callable, Literal, Protocol

import numpy as np

from clients.base import PolicyResponseFormatError

from .authority import (
    AuthorityHardwareController,
    AuthoritySnapshot,
    ExecutionAuthority,
    ExecutionState,
    ROLLOUT_STATES,
    RolloutActionEnvelope,
    SupervisorEvent,
    SupervisorEventKind,
)
from .buffer import StreamActionBuffer


ExecutionMode = Literal["streaming", "chunk_sync"]


class MalformedRolloutResponseError(ValueError):
    pass


class StableStepCallback(Protocol):
    def __call__(
        self,
        *,
        snapshot_before_command: Any,
        action_state: Any,
        is_intervention: bool,
        session_id: str | None,
        raw_action: np.ndarray | None,
    ) -> None: ...


RuntimeEventCallback = Callable[..., Any]
ChunkLogger = Callable[[int, int, int, np.ndarray], None]
SnapshotValidator = Callable[[Any], None]


@dataclass(frozen=True, slots=True)
class _InferenceRequest:
    request_id: int
    epoch: int
    session_id: str | None
    observation_seq: int
    requested_monotonic_s: float
    snapshot: Any
    requires_fresh_lane: bool


@dataclass(frozen=True, slots=True)
class _InferenceCompletion:
    request: _InferenceRequest
    actions: Any = None
    error: BaseException | None = None
    inference_seconds: float = 0.0
    expired: bool = False
    timeout_notice: bool = False


InferenceLaneFactory = Callable[[], Any]


class AsyncRolloutInference:
    """Asynchronous inference with replaceable transport lanes.

    Normal requests are serialized.  A hard deadline or epoch cancellation
    closes the active lane and lets the next cadence use an independent client
    supplied by ``lane_factory``.  Request id and epoch checks remain a second
    fence if a transport takes time to notice that its socket was closed.
    """

    def __init__(
        self,
        *,
        client: Any,
        prompt: str,
        request_timeout_s: float,
        lane_factory: InferenceLaneFactory,
    ) -> None:
        if request_timeout_s <= 0.0:
            raise ValueError("request_timeout_s must be positive")
        self.client = client
        self.prompt = prompt
        self.request_timeout_s = float(request_timeout_s)
        self.lane_factory = lane_factory
        self._lane_client = None
        self._results: queue.Queue[_InferenceCompletion] = queue.Queue()
        self._lock = threading.Lock()
        self._pending: _InferenceRequest | None = None
        self._pending_lane = None
        self._expired_request_ids: set[int] = set()
        self._completed_request_ids: set[int] = set()
        self._fresh_lane_required = True
        self._threads: set[threading.Thread] = set()
        self._next_request_id = 1
        self._closed = False

    def _configure_lane(self, lane_client: Any) -> None:
        lane_client.configure_inference_timeout(self.request_timeout_s)

    @staticmethod
    def _close_lane(lane_client: Any | None) -> None:
        if lane_client is None:
            return
        try:
            lane_client.close_inference_session()
        except Exception:
            pass

    def _retire_current_lane_locked(
        self,
        lane_client: Any,
    ) -> Any | None:
        if self._lane_client is not lane_client:
            return None
        self._lane_client = None
        self._fresh_lane_required = True
        return lane_client

    def _expire_pending_locked(self, request: _InferenceRequest) -> Any | None:
        self._expired_request_ids.add(request.request_id)
        if self._pending is not None and self._pending.request_id == request.request_id:
            self._pending = None
            request_lane = self._pending_lane
            self._pending_lane = None
        else:
            request_lane = None
        self._fresh_lane_required = True
        if request_lane is None:
            return None
        retired_lane = self._retire_current_lane_locked(request_lane)
        if retired_lane is not None:
            return retired_lane
        return request_lane

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._pending is not None

    def submit(
        self,
        snapshot: Any,
        *,
        authority: AuthoritySnapshot,
        observation_seq: int,
    ) -> int | None:
        with self._lock:
            if self._closed or self._pending is not None:
                return None
            request = _InferenceRequest(
                request_id=self._next_request_id,
                epoch=authority.epoch,
                session_id=authority.session_id,
                observation_seq=int(observation_seq),
                requested_monotonic_s=time.monotonic(),
                snapshot=snapshot,
                requires_fresh_lane=(
                    self._fresh_lane_required or self._lane_client is None
                ),
            )
            self._next_request_id += 1
            self._pending = request
            worker = threading.Thread(
                target=self._worker,
                args=(request,),
                name=f"rollout-inference-{request.request_id}",
                daemon=True,
            )
            self._threads.add(worker)
            worker.start()
            return request.request_id

    def invalidate_epoch(self, current_epoch: int) -> None:
        lane_to_close = None
        with self._lock:
            if self._pending is not None and self._pending.epoch != int(current_epoch):
                lane_to_close = self._expire_pending_locked(self._pending)
            if self._lane_client is not None:
                lane_to_close = self._retire_current_lane_locked(self._lane_client)
                self._fresh_lane_required = True
        self._close_lane(lane_to_close)

    def _worker(self, request: _InferenceRequest) -> None:
        lane_client: Any | None = None
        lane_owned = False
        lane_to_close: Any | None = None
        try:
            if request.requires_fresh_lane:
                lane_client = self.lane_factory()
                lane_owned = True
                if lane_client is None or lane_client is self.client:
                    raise RuntimeError(
                        "inference_lane_factory must return an independent client"
                    )
                self._configure_lane(lane_client)
                with self._lock:
                    request_is_active = (
                        not self._closed
                        and request.request_id not in self._expired_request_ids
                        and self._pending is not None
                        and self._pending.request_id == request.request_id
                    )
                    if request_is_active:
                        previous_lane = self._lane_client
                        self._lane_client = lane_client
                        self._fresh_lane_required = False
                        self._pending_lane = lane_client
                    else:
                        previous_lane = None
                if previous_lane is not None and previous_lane is not lane_client:
                    self._close_lane(previous_lane)
                if not request_is_active:
                    self._close_lane(lane_client)
                    completion = _InferenceCompletion(
                        request=request,
                        inference_seconds=(
                            time.monotonic() - request.requested_monotonic_s
                        ),
                        expired=True,
                    )
                    return
            else:
                with self._lock:
                    lane_client = self._lane_client
                    request_is_active = (
                        not self._closed
                        and request.request_id not in self._expired_request_ids
                        and self._pending is not None
                        and self._pending.request_id == request.request_id
                    )
                    if request_is_active and lane_client is not None:
                        self._pending_lane = lane_client
                if not request_is_active:
                    completion = _InferenceCompletion(
                        request=request,
                        inference_seconds=(
                            time.monotonic() - request.requested_monotonic_s
                        ),
                        expired=True,
                    )
                    return
                if lane_client is None:
                    raise RuntimeError("no rollout inference lane is available")

            supports_sessions = bool(lane_client.supports_policy_sessions)
            kwargs = {"session_id": request.session_id} if supports_sessions else {}
            actions = lane_client.infer_actions(
                request.snapshot,
                prompt=self.prompt,
                **kwargs,
            )
            completion = _InferenceCompletion(
                request=request,
                actions=actions,
                inference_seconds=(
                    time.monotonic() - request.requested_monotonic_s
                ),
            )
        except BaseException as exc:
            completion = _InferenceCompletion(
                request=request,
                error=exc,
                inference_seconds=(
                    time.monotonic() - request.requested_monotonic_s
                ),
            )
            with self._lock:
                if self._pending is not None and self._pending.request_id == request.request_id:
                    self._pending_lane = None
                if lane_client is not None:
                    lane_to_close = self._retire_current_lane_locked(lane_client)
                    if lane_to_close is None and lane_owned:
                        lane_to_close = lane_client
        finally:
            if lane_to_close is not None:
                self._close_lane(lane_to_close)
            with self._lock:
                self._completed_request_ids.add(request.request_id)
                self._threads.discard(threading.current_thread())
            self._results.put(completion)

    def poll(self) -> tuple[_InferenceCompletion, ...]:
        completions: list[_InferenceCompletion] = []
        lane_to_close = None
        now_s = time.monotonic()
        with self._lock:
            pending = self._pending
            if (
                pending is not None
                and pending.request_id not in self._completed_request_ids
                and now_s - pending.requested_monotonic_s >= self.request_timeout_s
            ):
                lane_to_close = self._expire_pending_locked(pending)
                completions.append(
                    _InferenceCompletion(
                        request=pending,
                        error=TimeoutError(
                            f"rollout inference request {pending.request_id} exceeded "
                            f"{self.request_timeout_s:.3f}s"
                        ),
                        inference_seconds=now_s - pending.requested_monotonic_s,
                        expired=True,
                        timeout_notice=True,
                    )
                )

        self._close_lane(lane_to_close)

        while True:
            try:
                completion = self._results.get_nowait()
            except queue.Empty:
                break
            retired_lane = None
            with self._lock:
                expired = completion.request.request_id in self._expired_request_ids
                deadline_exceeded = (
                    not expired
                    and completion.inference_seconds >= self.request_timeout_s
                )
                if deadline_exceeded:
                    retired_lane = self._expire_pending_locked(completion.request)
                    expired = True
                if (
                    self._pending is not None
                    and self._pending.request_id == completion.request.request_id
                ):
                    self._pending = None
                    self._pending_lane = None
                self._expired_request_ids.discard(completion.request.request_id)
                self._completed_request_ids.discard(completion.request.request_id)
            if retired_lane is not None:
                self._close_lane(retired_lane)
            if deadline_exceeded:
                completions.append(
                    replace(
                        completion,
                        error=TimeoutError(
                            f"rollout inference request "
                            f"{completion.request.request_id} exceeded "
                            f"{self.request_timeout_s:.3f}s"
                        ),
                        expired=True,
                        timeout_notice=True,
                    )
                )
            else:
                completions.append(replace(completion, expired=expired))
        return tuple(completions)

    def close(self, *, join_timeout_s: float = 0.2) -> None:
        lane_to_close = None
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._pending is not None:
                lane_to_close = self._expire_pending_locked(self._pending)
            if self._lane_client is not None:
                lane_to_close = self._lane_client
                self._lane_client = None
            threads = tuple(self._threads)
        self._close_lane(lane_to_close)
        deadline_s = time.monotonic() + max(0.0, float(join_timeout_s))
        for thread in threads:
            thread.join(timeout=max(0.0, deadline_s - time.monotonic()))


def _validated_actions(
    actions: Any,
    *,
    action_dim: int,
    chunk_size: int | None,
) -> np.ndarray:
    if actions is None:
        return np.empty((0, action_dim), dtype=np.float64)
    try:
        values = np.asarray(actions, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise MalformedRolloutResponseError(f"actions are not numeric: {exc}") from exc
    if values.ndim == 1:
        if values.size == 0:
            return np.empty((0, action_dim), dtype=np.float64)
        if values.shape[0] != action_dim:
            raise MalformedRolloutResponseError(
                f"action vector has dim {values.shape[0]}, expected {action_dim}"
            )
        values = values.reshape(1, action_dim)
    elif values.ndim == 2:
        if values.shape[1] != action_dim:
            raise MalformedRolloutResponseError(
                f"action chunk has shape {values.shape}, expected (*, {action_dim})"
            )
    else:
        raise MalformedRolloutResponseError(
            f"action response has shape {values.shape}, expected ({action_dim},) or (*, {action_dim})"
        )
    if not np.all(np.isfinite(values)):
        raise MalformedRolloutResponseError("action response contains NaN or infinity")
    if chunk_size is not None:
        values = values[: min(len(values), chunk_size)]
    return values.copy()


@dataclass(frozen=True, slots=True)
class CoordinatorTick:
    state: ExecutionState
    command_committed: bool = False
    command_seconds: float | None = None
    inference_seconds: tuple[float, ...] = ()
    inference_errors: tuple[str, ...] = ()
    fault: BaseException | None = None


class DynamicRolloutCoordinator:
    """Single-supervisor dynamic rollout/intervention execution core.

    The coordinator is deliberately topology-agnostic.  ``robot`` may expose
    only the two physical slaves or a hardware-selected rollout-mirror endpoint.
    All Piper role/gain/load/alignment details remain behind
    :class:`AuthorityHardwareController`.
    """

    def __init__(
        self,
        *,
        client: Any,
        source: Any,
        robot: Any,
        spec: Any,
        hardware_controller: AuthorityHardwareController,
        prompt: str,
        execution_mode: ExecutionMode,
        fps: float,
        chunk_size: int | None = None,
        inference_rate: float | None = None,
        latency_k: int = 0,
        min_smooth_steps: int = 8,
        buffer_max_chunks: int = 10,
        request_timeout_s: float = 15.0,
        inference_lane_factory: InferenceLaneFactory | None = None,
        intervention_sample_max_age_s: float | None = None,
        stable_step_callback: StableStepCallback | None = None,
        snapshot_validator: SnapshotValidator | None = None,
        runtime_event_callback: RuntimeEventCallback | None = None,
        log_chunk: ChunkLogger | None = None,
    ) -> None:
        if execution_mode not in ("streaming", "chunk_sync"):
            raise ValueError(f"unsupported execution_mode {execution_mode!r}")
        if fps < 0.0:
            raise ValueError("fps must be non-negative")
        if chunk_size is not None and chunk_size <= 0:
            raise ValueError("chunk_size must be positive when provided")
        if inference_rate is not None and inference_rate < 0.0:
            raise ValueError("inference_rate must be non-negative")
        if min_smooth_steps <= 0:
            raise ValueError("min_smooth_steps must be positive")
        self.client = client
        self.source = source
        self.robot = robot
        self.action_dim = int(spec.action_dim)
        self.hardware_controller = hardware_controller
        self.execution_mode = execution_mode
        self.chunk_size = chunk_size
        resolved_fps = float(fps)
        self.inference_rate = resolved_fps if inference_rate is None else float(inference_rate)
        self.latency_k = max(0, int(latency_k))
        self.min_smooth_steps = int(min_smooth_steps)
        default_sample_age = max(0.05, 2.0 / resolved_fps) if resolved_fps > 0.0 else 0.1
        self.intervention_sample_max_age_s = (
            default_sample_age
            if intervention_sample_max_age_s is None
            else float(intervention_sample_max_age_s)
        )
        if self.intervention_sample_max_age_s <= 0.0:
            raise ValueError("intervention_sample_max_age_s must be positive")
        self.authority = ExecutionAuthority()
        self.buffer = StreamActionBuffer(
            max_chunks=buffer_max_chunks,
            state_dim=int(spec.action_dim),
            smooth_method="temporal",
        )
        self.buffer.activate_epoch(self.authority.epoch)
        resolved_lane_factory = (
            inference_lane_factory
            if inference_lane_factory is not None
            else client.fork_rollout_inference_session
        )
        self.inference = AsyncRolloutInference(
            client=client,
            prompt=prompt,
            request_timeout_s=request_timeout_s,
            lane_factory=resolved_lane_factory,
        )
        self.stable_step_callback = stable_step_callback
        self.snapshot_validator = snapshot_validator
        self.runtime_event_callback = runtime_event_callback
        self.log_chunk = log_chunk
        self._observation_seq = 0
        self._chunk_index = 0
        self._rollout_command_count = 0
        self._next_inference_monotonic_s = 0.0
        self._last_intervention_frame_group_seq: int | None = None
        self._intervention_generation: int | None = None
        self._paused_from: ExecutionState | None = None
        self._closed = False

    def request_start(self, *, reason: str = "start requested") -> None:
        self.authority.post(SupervisorEventKind.START, reason=reason)

    def request_rollout(self, *, reason: str = "rollout requested") -> None:
        self.authority.post(SupervisorEventKind.ROLLOUT, reason=reason)

    def request_intervention(self, *, reason: str = "intervention requested") -> None:
        self.authority.post(SupervisorEventKind.INTERVENE, reason=reason)

    def request_pause(self, *, reason: str = "episode paused") -> None:
        self.authority.post(SupervisorEventKind.PAUSE, reason=reason)

    def request_stop(self, *, reason: str = "episode stopped") -> None:
        self.authority.post(SupervisorEventKind.STOP, reason=reason)

    def _emit_runtime(self, event: str, **fields: Any) -> None:
        if self.runtime_event_callback is not None:
            self.runtime_event_callback(event, **fields)

    def _emit_hardware_transaction(self, authority: str, result: Any) -> None:
        operations = []
        result_operations = getattr(result, "operations", None)
        if result_operations is None and hasattr(result, "writes"):
            result_operations = (result,)
        for operation in result_operations or ():
            writes = getattr(operation, "writes", None)
            if writes is not None:
                operations.append(
                    {
                        "kind": "linkage_role",
                        "target": getattr(getattr(operation, "target", None), "short_name", None),
                        "attempts": int(getattr(operation, "attempts", 0)),
                        "requested": len(writes),
                        "written": sum(bool(getattr(item, "written", False)) for item in writes),
                        "skipped": sum(not bool(getattr(item, "written", False)) for item in writes),
                        "verified": sum(bool(getattr(item, "verified", False)) for item in writes),
                        "motion_output_seeded": sum(
                            bool(getattr(item, "motion_output_seeded", False))
                            for item in (
                                tuple(writes)
                                + tuple(getattr(operation, "rollbacks", ()))
                            )
                        ),
                        "rollbacks": len(getattr(operation, "rollbacks", ())),
                    }
                )
                continue
            failures = getattr(operation, "failures", ())
            operations.append(
                {
                    "kind": "hardware_batch",
                    "operation": str(getattr(operation, "operation", type(operation).__name__)),
                    "ok": bool(getattr(operation, "ok", not failures)),
                    "failures": [repr(getattr(item, "error", item)) for item in failures],
                }
            )
        self._emit_runtime(
            "hardware_authority_transaction",
            authority=authority,
            generation=getattr(result, "generation", None),
            operations=operations,
        )

    def _transition(
        self,
        state: ExecutionState,
        *,
        reason: str,
        bump_epoch: bool = False,
        session_id: str | None = None,
        fault: BaseException | None = None,
    ) -> None:
        transition = self.authority.transition(
            state,
            reason=reason,
            bump_epoch=bump_epoch,
            session_id=session_id,
            fault=fault,
        )
        self._emit_runtime(
            "authority_transition",
            previous=transition.previous.value,
            current=transition.current.value,
            epoch=transition.epoch,
            session_id=transition.session_id,
            reason=transition.reason,
        )

    def _invalidate_rollout(self) -> None:
        epoch = self.authority.epoch
        self.buffer.activate_epoch(epoch, clear=True)
        self.inference.invalidate_epoch(epoch)
        self._next_inference_monotonic_s = 0.0

    def _resync_client(self, session_id: str | None) -> None:
        capability = self.client.resync_after_authority_change(
            session_id=session_id,
        )
        self._emit_runtime(
            "client_resync",
            epoch=self.authority.epoch,
            session_id=session_id,
            session_capability=getattr(capability, "value", str(capability)),
        )

    def _ensure_started(self, reason: str) -> None:
        if self.authority.state is not ExecutionState.STARTING:
            return
        startup_result = self.hardware_controller.start_authority()
        self._emit_hardware_transaction("STARTUP", startup_result)
        self.hardware_controller.require_healthy()
        self._transition(ExecutionState.IDLE, reason=reason)

    def _enter_rollout(self, reason: str) -> None:
        self._ensure_started("authority startup before rollout")
        if self.authority.state in ROLLOUT_STATES:
            return
        next_epoch = self.authority.epoch + 1
        session_id = self.authority.new_session_id(next_epoch)
        self._transition(
            ExecutionState.TO_ROLLOUT,
            reason=reason,
            bump_epoch=True,
            session_id=session_id,
        )
        self._invalidate_rollout()
        result = self.hardware_controller.enter_rollout()
        self._emit_hardware_transaction("ROLLOUT", result)
        self.hardware_controller.require_healthy()
        self._resync_client(session_id)
        self._intervention_generation = None
        self._last_intervention_frame_group_seq = None
        self._paused_from = None
        self._transition(
            ExecutionState.ROLLOUT_REACQUIRE,
            reason="rollout hardware transaction committed; reacquiring a fresh chunk",
            session_id=session_id,
        )

    def _enter_intervention(self, reason: str) -> None:
        self._ensure_started("authority startup before intervention")
        if self.authority.state is ExecutionState.INTERVENE_ACTIVE:
            return
        resume_paused_intervention = (
            self.authority.state is ExecutionState.EPISODE_PAUSED
            and self._paused_from is ExecutionState.INTERVENE_ACTIVE
        )
        self._transition(
            ExecutionState.TO_INTERVENE,
            reason=reason,
            bump_epoch=True,
            session_id=None,
        )
        self._invalidate_rollout()
        self._resync_client(None)
        if resume_paused_intervention:
            result = self.hardware_controller.resume_intervention()
            transaction_name = "INTERVENE_RESUME"
        else:
            result = self.hardware_controller.enter_intervention()
            transaction_name = "INTERVENE"
        self._emit_hardware_transaction(transaction_name, result)
        self._intervention_generation = int(result.generation)
        self._last_intervention_frame_group_seq = int(
            result.initial_dispatch_seq
        )
        self.hardware_controller.require_healthy()
        self._paused_from = None
        self._transition(
            ExecutionState.INTERVENE_ACTIVE,
            reason="intervention hardware transaction committed",
            session_id=None,
        )

    def _pause(self, reason: str) -> None:
        self._ensure_started("authority startup before pause")
        if self.authority.state is ExecutionState.EPISODE_PAUSED:
            return
        paused_from = self.authority.state
        self._paused_from = paused_from
        self._transition(
            ExecutionState.EPISODE_PAUSED,
            reason=reason,
            bump_epoch=True,
            session_id=None,
        )
        self._invalidate_rollout()
        self._resync_client(None)
        if paused_from is ExecutionState.INTERVENE_ACTIVE:
            result = self.hardware_controller.pause_intervention()
            self._emit_hardware_transaction("INTERVENE_PAUSE", result)
        else:
            self.hardware_controller.pause_episode()
        self.hardware_controller.require_healthy()

    def _return_paused_intervention_to_rollout(self, reason: str) -> None:
        """Perform the deferred FA->FC transaction before an episode stops."""

        if (
            self.authority.state is not ExecutionState.EPISODE_PAUSED
            or self._paused_from is not ExecutionState.INTERVENE_ACTIVE
        ):
            return
        self._transition(
            ExecutionState.TO_ROLLOUT,
            reason=f"return paused INTERVENE to ROLLOUT before stop: {reason}",
            session_id=None,
        )
        result = self.hardware_controller.enter_rollout()
        self._emit_hardware_transaction("ROLLOUT_BEFORE_STOP", result)
        self.hardware_controller.require_healthy()
        self._intervention_generation = None
        self._last_intervention_frame_group_seq = None
        self._transition(
            ExecutionState.ROLLOUT_REACQUIRE,
            reason="paused intervention returned to rollout hardware before stop",
            session_id=None,
        )

    def _stop(self, reason: str) -> None:
        self._ensure_started("authority startup before stop")
        if self.authority.state is ExecutionState.IDLE:
            self._paused_from = None
            return
        if self.authority.state is not ExecutionState.EPISODE_PAUSED:
            self._pause(f"hold before stopping episode: {reason}")
        self._return_paused_intervention_to_rollout(reason)
        self._transition(
            ExecutionState.IDLE,
            reason=reason,
            bump_epoch=True,
            session_id=None,
        )
        self._invalidate_rollout()
        self._resync_client(None)
        self._paused_from = None

    def _latch_fault(self, error: BaseException, reason: str) -> None:
        if self.authority.state is ExecutionState.FAULT:
            return
        try:
            self._transition(
                ExecutionState.FAULT,
                reason=reason,
                bump_epoch=True,
                session_id=None,
                fault=error,
            )
        finally:
            self._invalidate_rollout()
            try:
                hold_result = self.hardware_controller.hold_position()
                hold_items = []
                for item in getattr(hold_result, "results", ()):
                    hold_items.append(
                        {
                            "arm_id": getattr(
                                getattr(item, "arm_id", None),
                                "value",
                                str(getattr(item, "arm_id", "unknown")),
                            ),
                            "ok": getattr(item, "error", None) is None,
                            "result": str(getattr(item, "value", None)),
                            "error": repr(getattr(item, "error", None))
                            if getattr(item, "error", None) is not None
                            else None,
                        }
                    )
                self._emit_runtime(
                    "fault_hold_completed",
                    original_error=repr(error),
                    arms=hold_items,
                )
            except Exception as hold_error:
                self._emit_runtime(
                    "fault_hold_failed",
                    error=repr(hold_error),
                    original_error=repr(error),
                )
            self._paused_from = None
        self._emit_runtime("fault_latched", error=repr(error), reason=reason)

    def _handle_event(self, event: SupervisorEvent) -> None:
        if event.kind is SupervisorEventKind.FAULT:
            self._latch_fault(event.error or RuntimeError(event.reason), event.reason)
            return
        if event.kind is SupervisorEventKind.START:
            self._ensure_started(event.reason)
            return
        if event.kind is SupervisorEventKind.ROLLOUT:
            self._enter_rollout(event.reason)
            return
        if event.kind is SupervisorEventKind.INTERVENE:
            self._enter_intervention(event.reason)
            return
        if event.kind is SupervisorEventKind.PAUSE:
            self._pause(event.reason)
            return
        if event.kind is SupervisorEventKind.STOP:
            self._stop(event.reason)
            return
        raise AssertionError(f"unhandled supervisor event {event.kind}")

    def _process_events(self) -> None:
        # One queued intent per stable tick prevents an i->r key race from
        # executing two complete hardware role transactions back-to-back.
        for event in self.authority.drain_events(max_events=1):
            if self.authority.state is ExecutionState.FAULT:
                return
            try:
                self._handle_event(event)
            except Exception as exc:
                self._emit_hardware_transaction(
                    f"{event.kind.value.upper()}_FAILED",
                    exc,
                )
                self._latch_fault(exc, f"event {event.kind.value} failed: {event.reason}")
                return

    def _set_reacquire_if_starved(self, reason: str) -> None:
        if (
            self.authority.state is ExecutionState.ROLLOUT_ACTIVE
            and not self.buffer.has_any(expected_epoch=self.authority.epoch)
        ):
            self._transition(
                ExecutionState.ROLLOUT_REACQUIRE,
                reason=reason,
                session_id=self.authority.session_id,
            )

    def _fence_current_rollout_after_inference_failure(
        self,
        completion: _InferenceCompletion,
        *,
        reason: str,
    ) -> None:
        """Stop consuming an older chunk after a current inference failure.

        A timeout, transport error, empty response, or stale response means the
        next action is no longer backed by a successful current observation.
        Keep the authority epoch and server session intact so inference can be
        retried, but clear all buffered actions and hold the FC arms first.
        Completions from an obsolete epoch/session must not disturb a newer
        rollout.
        """

        request = completion.request
        request_is_current = (
            request.epoch == self.authority.epoch
            and request.session_id == self.authority.session_id
            and self.authority.state in ROLLOUT_STATES
        )
        if not request_is_current:
            return

        self.buffer.activate_epoch(self.authority.epoch, clear=True)
        self.hardware_controller.hold_position()
        self.hardware_controller.require_healthy()
        if self.authority.state is ExecutionState.ROLLOUT_ACTIVE:
            self._transition(
                ExecutionState.ROLLOUT_REACQUIRE,
                reason=reason,
                session_id=self.authority.session_id,
            )
        self._emit_runtime(
            "rollout_inference_failure_fenced",
            request_id=request.request_id,
            epoch=request.epoch,
            reason=reason,
        )

    def _poll_inference(self) -> tuple[tuple[float, ...], tuple[str, ...]]:
        durations: list[float] = []
        errors: list[str] = []
        for completion in self.inference.poll():
            durations.append(completion.inference_seconds)
            if completion.timeout_notice:
                message = repr(completion.error)
                errors.append(message)
                self._fence_current_rollout_after_inference_failure(
                    completion,
                    reason="rollout inference timed out; retrying",
                )
                self._emit_runtime(
                    "rollout_inference_timeout",
                    request_id=completion.request.request_id,
                    epoch=completion.request.epoch,
                )
                continue

            if completion.expired:
                self._emit_runtime(
                    "rollout_response_discarded",
                    request_id=completion.request.request_id,
                    response_epoch=completion.request.epoch,
                    current_epoch=self.authority.epoch,
                    reason="expired_or_old_epoch",
                )
                continue
            if completion.error is not None:
                if isinstance(completion.error, PolicyResponseFormatError):
                    self._latch_fault(
                        completion.error,
                        "malformed rollout action response",
                    )
                    errors.append(repr(completion.error))
                    break
                message = repr(completion.error)
                errors.append(message)
                self._fence_current_rollout_after_inference_failure(
                    completion,
                    reason="rollout inference failed; retrying",
                )
                self._emit_runtime(
                    "rollout_inference_error",
                    request_id=completion.request.request_id,
                    epoch=completion.request.epoch,
                    error=message,
                )
                continue

            try:
                actions = _validated_actions(
                    completion.actions,
                    action_dim=self.action_dim,
                    chunk_size=self.chunk_size,
                )
            except MalformedRolloutResponseError as exc:
                self._latch_fault(exc, "malformed rollout action response")
                errors.append(repr(exc))
                break
            if len(actions) == 0:
                self._fence_current_rollout_after_inference_failure(
                    completion,
                    reason="empty rollout chunk; retrying",
                )
                self._emit_runtime(
                    "rollout_empty_chunk",
                    request_id=completion.request.request_id,
                    epoch=completion.request.epoch,
                )
                continue
            envelope = RolloutActionEnvelope(
                epoch=completion.request.epoch,
                request_id=completion.request.request_id,
                observation_seq=completion.request.observation_seq,
                session_id=completion.request.session_id,
                actions=actions,
            )
            if not self.authority.accepts_rollout_response(envelope):
                self._emit_runtime(
                    "rollout_response_discarded",
                    request_id=envelope.request_id,
                    response_epoch=envelope.epoch,
                    current_epoch=self.authority.epoch,
                    reason="authority_rejected",
                )
                continue
            integrated = self.buffer.integrate_envelope(
                envelope,
                max_k=self.latency_k if self.execution_mode == "streaming" else 0,
                min_m=self.min_smooth_steps,
                blend=self.execution_mode == "streaming",
            )
            if not integrated:
                self._emit_runtime(
                    "rollout_response_discarded",
                    request_id=envelope.request_id,
                    response_epoch=envelope.epoch,
                    current_epoch=self.authority.epoch,
                    reason="buffer_epoch_rejected",
                )
                continue
            if self.log_chunk is not None:
                self.log_chunk(
                    self._chunk_index,
                    len(actions),
                    self._rollout_command_count,
                    actions[0],
                )
            self._chunk_index += 1
        return tuple(durations), tuple(errors)

    def _schedule_inference_if_needed(self) -> None:
        if self.authority.state not in ROLLOUT_STATES or self.inference.busy:
            return
        epoch = self.authority.epoch
        if self.execution_mode == "chunk_sync" and self.buffer.has_any(expected_epoch=epoch):
            return
        now_s = time.monotonic()
        if self.execution_mode == "streaming" and now_s < self._next_inference_monotonic_s:
            return
        snapshot = self.source.capture_snapshot()
        if self.snapshot_validator is not None:
            self.snapshot_validator(snapshot)
        self._observation_seq += 1
        observation_time_s = float(snapshot.timestamp_s)
        request_id = self.inference.submit(
            snapshot,
            authority=self.authority.snapshot(),
            observation_seq=self._observation_seq,
        )
        if request_id is None:
            return
        period_s = 1.0 / self.inference_rate if self.inference_rate > 0.0 else 0.0
        self._next_inference_monotonic_s = now_s + period_s
        self._emit_runtime(
            "rollout_inference_requested",
            request_id=request_id,
            epoch=self.authority.epoch,
            session_id=self.authority.session_id,
            observation_seq=self._observation_seq,
            observation_time_s=observation_time_s,
        )

    def _emit_stable_step(
        self,
        *,
        snapshot_before_command: Any,
        action_state: Any,
        is_intervention: bool,
        raw_action: np.ndarray | None,
    ) -> None:
        if self.stable_step_callback is None:
            return
        self.stable_step_callback(
            snapshot_before_command=snapshot_before_command,
            action_state=action_state,
            is_intervention=is_intervention,
            session_id=self.authority.session_id,
            raw_action=None
            if raw_action is None
            else np.asarray(raw_action, dtype=np.float64).copy(),
        )

    def _execute_rollout_action(
        self,
        *,
        inference_seconds: tuple[float, ...],
        inference_errors: tuple[str, ...],
    ) -> CoordinatorTick:
        epoch = self.authority.epoch
        popped = self.buffer.pop_next_enveloped_action(expected_epoch=epoch)
        if popped is None:
            self._set_reacquire_if_starved("rollout action buffer empty; reacquiring")
            return CoordinatorTick(
                state=self.authority.state,
                inference_seconds=inference_seconds,
                inference_errors=inference_errors,
            )
        action, envelope = popped
        snapshot_before_command = self.source.capture_snapshot()
        if self.snapshot_validator is not None:
            self.snapshot_validator(snapshot_before_command)
        command_started_s = time.monotonic()
        with self.authority.rollout_command_gate(envelope) as allowed:
            if not allowed:
                return CoordinatorTick(
                    state=self.authority.state,
                    inference_seconds=inference_seconds,
                    inference_errors=inference_errors,
                )
            self.client.command_action(self.robot, action)
            action_state = self.client.action_state_after_command(
                self.robot,
                snapshot_before_command,
            )
        command_seconds = time.monotonic() - command_started_s
        if self.authority.state is ExecutionState.ROLLOUT_REACQUIRE:
            self._transition(
                ExecutionState.ROLLOUT_ACTIVE,
                reason=f"fresh rollout request {envelope.request_id} committed",
                session_id=self.authority.session_id,
            )
        timestamp_s = float(snapshot_before_command.timestamp_s)
        self._emit_stable_step(
            snapshot_before_command=snapshot_before_command,
            action_state=action_state,
            is_intervention=False,
            raw_action=action,
        )
        self._rollout_command_count += 1
        self._emit_runtime(
            "rollout_action_committed",
            epoch=envelope.epoch,
            session_id=envelope.session_id,
            request_id=envelope.request_id,
            observation_seq=envelope.observation_seq,
            timestamp_s=timestamp_s,
        )
        return CoordinatorTick(
            state=self.authority.state,
            command_committed=True,
            command_seconds=command_seconds,
            inference_seconds=inference_seconds,
            inference_errors=inference_errors,
        )

    def _sample_intervention_step(
        self,
        *,
        inference_seconds: tuple[float, ...],
        inference_errors: tuple[str, ...],
    ) -> CoordinatorTick:
        snapshot = self.source.capture_snapshot()
        if self.snapshot_validator is not None:
            self.snapshot_validator(snapshot)
        sampled_at_s = time.monotonic()
        sample = self.hardware_controller.sample_intervention_action_state()
        if sample is None:
            return CoordinatorTick(
                state=self.authority.state,
                inference_seconds=inference_seconds,
                inference_errors=inference_errors,
            )
        if (
            self._intervention_generation is not None
            and sample.generation != self._intervention_generation
        ):
            raise RuntimeError(
                "intervention sample gateway generation does not match the committed transition"
            )
        previous_frame_group_seq = self._last_intervention_frame_group_seq
        if sample.is_new_dispatch:
            if (
                previous_frame_group_seq is not None
                and sample.frame_group_seq <= previous_frame_group_seq
            ):
                raise RuntimeError(
                    "hardware reused or reordered an INTERVENE paired dispatch receipt"
                )
            if (
                abs(sampled_at_s - sample.committed_monotonic_s)
                > self.intervention_sample_max_age_s
            ):
                raise RuntimeError(
                    "INTERVENE paired dispatch exceeded the observation-to-command age limit"
                )
        elif (
            previous_frame_group_seq is None
            or sample.frame_group_seq != previous_frame_group_seq
        ):
            raise RuntimeError(
                "stationary INTERVENE sample does not match the last paired dispatch"
            )
        self._last_intervention_frame_group_seq = sample.frame_group_seq
        timestamp_s = float(snapshot.timestamp_s)
        self._emit_stable_step(
            snapshot_before_command=snapshot,
            action_state=sample.action_state,
            is_intervention=True,
            raw_action=None,
        )
        self._emit_runtime(
            "intervention_step_committed",
            epoch=self.authority.epoch,
            gateway_generation=sample.generation,
            frame_group_seq=sample.frame_group_seq,
            is_new_dispatch=sample.is_new_dispatch,
            timestamp_s=timestamp_s,
        )
        return CoordinatorTick(
            state=self.authority.state,
            command_committed=True,
            inference_seconds=inference_seconds,
            inference_errors=inference_errors,
        )

    def step(self) -> CoordinatorTick:
        """Run one fixed-rate supervisor tick; never call this concurrently."""

        if self._closed:
            raise RuntimeError("dynamic rollout coordinator is closed")
        self.authority.bind_supervisor()
        try:
            self._process_events()
            if self.authority.state is ExecutionState.FAULT:
                snapshot = self.authority.snapshot()
                return CoordinatorTick(
                    state=snapshot.state,
                    fault=snapshot.fault,
                )
            if (
                self.authority.state is not ExecutionState.STARTING
                and self.authority.state not in ROLLOUT_STATES
                and self.authority.state is not ExecutionState.INTERVENE_ACTIVE
            ):
                self.hardware_controller.require_healthy()
            inference_seconds, inference_errors = self._poll_inference()
            if self.authority.state is ExecutionState.FAULT:
                snapshot = self.authority.snapshot()
                return CoordinatorTick(
                    state=snapshot.state,
                    inference_seconds=inference_seconds,
                    inference_errors=inference_errors,
                    fault=snapshot.fault,
                )
            if self.authority.state in ROLLOUT_STATES:
                self.hardware_controller.require_healthy()
                self._schedule_inference_if_needed()
                return self._execute_rollout_action(
                    inference_seconds=inference_seconds,
                    inference_errors=inference_errors,
                )
            if self.authority.state is ExecutionState.INTERVENE_ACTIVE:
                return self._sample_intervention_step(
                    inference_seconds=inference_seconds,
                    inference_errors=inference_errors,
                )
            return CoordinatorTick(
                state=self.authority.state,
                inference_seconds=inference_seconds,
                inference_errors=inference_errors,
            )
        except Exception as exc:
            self._latch_fault(exc, "dynamic rollout supervisor tick failed")
            snapshot = self.authority.snapshot()
            return CoordinatorTick(
                state=snapshot.state,
                fault=snapshot.fault,
            )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.inference.close()
