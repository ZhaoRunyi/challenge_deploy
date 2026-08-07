from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import replace
import threading
import time
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

import numpy as np

from .constants import (
    DEFAULT_ARM_STEP_LENGTH,
    PIPER_GRIPPER_FULL_OPEN_METERS,
)
from .conversions import (
    joints_feedback_to_rad,
    opening_to_sdk_gripper,
    sdk_gripper_to_opening,
)
from .linkage_gateway import DEFAULT_FRAME_FRESHNESS_S, GatewayReadyTimeoutError
from .piper import (
    DEFAULT_GRIPPER_EFFORT,
    SinglePiperArm,
    build_dual_joint_target_action_state,
    gripper_effort_value,
)
from .schemas import DualPiperState, PiperArmState
from .topology import (
    ALL_ARM_IDS,
    MASTER_ARM_IDS,
    SLAVE_ARM_IDS,
    ArmId,
    ArmOperationError,
    ArmOperationResult,
    BatchOperationError,
    BatchOperationResult,
    FaultRecord,
    LatchedFaultError,
    LifecycleResult,
    LinkageRole,
    PhysicalArmStates,
    RoleObservation,
    RoleTransactionError,
    RoleTransactionResult,
    RoleWriteResult,
    StaticMode,
    TopologyConfigError,
    TransitionInProgressError,
    isolated_can_names,
    isolated_slave_can_names,
)


DEFAULT_ROLE_VERIFY_TIMEOUT_S = 0.5
DRIVER_FEEDBACK_POLL_INTERVAL_S = 0.01
FC_ROLE_SEED_REPETITIONS = 5
FC_ROLE_SEED_INTERVAL_S = 0.01


class ArmLike(Protocol):
    name: str
    can_name: str

    def connect(self, *, read_only: bool = True) -> None: ...

    def disconnect(self) -> None: ...

    def abort_construction(self) -> None: ...

    def is_enabled(self) -> bool: ...

    def has_complete_driver_feedback(self) -> bool: ...

    def send_enable_without_reset(self) -> None: ...

    def enable_without_reset(self, *, retries: int, sleep_s: float) -> bool: ...

    def configure_linkage_role(self, role: LinkageRole) -> None: ...

    def read_state(self, *, prefer_joint_ctrl: bool = False) -> Any: ...

    def command_joint_positions(
        self,
        qpos: Iterable[float],
        *,
        speed_percent: int = 100,
        gripper_effort: int | None = None,
        command_gripper: bool = True,
    ) -> None: ...

    def command_end_pose(
        self,
        pose: Iterable[float],
        *,
        speed_percent: int = 50,
        gripper_effort: int | None = None,
        command_gripper: bool = True,
    ) -> None: ...

    def move_to_joint_positions(
        self,
        target_qpos: Iterable[float],
        *,
        hz: float = 30.0,
        step_sizes: Iterable[float],
        speed_percent: int = 100,
        gripper_effort: int | None = None,
    ) -> None: ...


class RoleObserver(Protocol):
    def observe(self, arm_id: ArmId, arm: ArmLike) -> RoleObservation: ...


class GatewayLifecycle(Protocol):
    @property
    def generation(self) -> int: ...

    def begin_generation(
        self,
        *,
        seed_gripper_efforts: Mapping[ArmId, int | None],
        seed_gripper_angles_um: Mapping[ArmId, int | None] | None = None,
    ) -> int: ...

    def start(self, *, automatic_dispatch: bool = True) -> int: ...

    def stop(self) -> None: ...

    def stop_at_cycle_boundary(self, *, timeout_s: float) -> bool: ...

    def wait_until_ready(self, *, timeout_s: float) -> Any: ...

    def wait_for_target_pair(self, *, timeout_s: float) -> Mapping[ArmId, Any]: ...


class SdkTrafficRoleObserver:
    """Observe FA/FC from fresh, mutually exclusive Piper traffic.

    On an isolated CAN, an FA arm emits 0x155-0x157/0x159 while an FC arm emits
    0x2A5-0x2A8. Ambiguous or stale traffic is deliberately reported as unknown;
    callers may then issue one explicit 0x470 write instead of guessing.
    """

    def __init__(
        self,
        *,
        freshness_s: float = DEFAULT_FRAME_FRESHNESS_S,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if freshness_s <= 0.0:
            raise ValueError("freshness_s must be positive")
        self.freshness_s = freshness_s
        self.clock = clock

    def observe(self, arm_id: ArmId, arm: ArmLike) -> RoleObservation:
        interface = getattr(arm, "interface", None)
        if interface is None:
            return RoleObservation.unknown(arm_id, "arm has no Piper SDK interface")
        now_s = self.clock()
        try:
            joint_ctrl = interface.GetArmJointCtrl()
            gripper_ctrl = interface.GetArmGripperCtrl()
            joint_feedback = interface.GetArmJointMsgs()
            gripper_feedback = interface.GetArmGripperMsgs()
        except Exception as exc:
            return RoleObservation.unknown(arm_id, f"SDK role observation failed: {exc}")

        def complete_fresh_family(*messages: Any) -> bool:
            timestamps = tuple(
                float(getattr(message, "time_stamp", 0.0))
                for message in messages
            )
            return (
                min(timestamps) > 0.0
                and 0.0 <= now_s - min(timestamps) <= self.freshness_s
            )

        ctrl_fresh = complete_fresh_family(joint_ctrl, gripper_ctrl)
        feedback_fresh = complete_fresh_family(joint_feedback, gripper_feedback)
        if ctrl_fresh and not feedback_fresh:
            return RoleObservation(
                arm_id,
                LinkageRole.TEACHING_INPUT,
                now_s,
                "fresh FA joint/gripper control family observed",
                True,
            )
        if feedback_fresh and not ctrl_fresh:
            return RoleObservation(
                arm_id,
                LinkageRole.MOTION_OUTPUT,
                now_s,
                "fresh FC joint/gripper feedback family observed",
                True,
            )
        if ctrl_fresh and feedback_fresh:
            evidence = "ambiguous fresh FA control and FC feedback traffic"
        elif arm_id.is_master:
            return RoleObservation(
                arm_id,
                LinkageRole.TEACHING_INPUT,
                now_s,
                "dedicated master has no continuous FC feedback",
                True,
            )
        else:
            evidence = "no fresh linkage-role traffic"
        return RoleObservation.unknown(arm_id, evidence)


class LinkageRoleController:
    """Observe first, write only mismatches, and transact paired role changes."""

    def __init__(
        self,
        arms: Mapping[ArmId, ArmLike],
        *,
        observer: RoleObserver,
        verify_timeout_s: float | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if verify_timeout_s is None:
            verify_timeout_s = max(
                DEFAULT_ROLE_VERIFY_TIMEOUT_S,
                float(getattr(observer, "freshness_s", 0.0)),
            )
        if verify_timeout_s < 0.0:
            raise ValueError("verify_timeout_s must be non-negative")
        self.arms = arms
        self.observer = observer
        self.verify_timeout_s = float(verify_timeout_s)
        self.sleeper = sleeper
        self._known: dict[ArmId, LinkageRole] = {}
        self._pending: dict[ArmId, LinkageRole] = {}
        self._lock = threading.RLock()

    def observe(self, arm_id: ArmId) -> RoleObservation:
        observation = self.observer.observe(arm_id, self.arms[arm_id])
        with self._lock:
            if observation.role is not None and observation.fresh:
                pending = self._pending.get(arm_id)
                if pending is None or observation.role is pending:
                    self._known[arm_id] = observation.role
                    self._pending.pop(arm_id, None)
        return observation

    def _observe_confirmed(
        self,
        arm_id: ArmId,
        *,
        expected: LinkageRole | None = None,
    ) -> RoleObservation:
        deadline = time.monotonic() + self.verify_timeout_s
        last = self.observe(arm_id)
        while not (
            last.fresh
            and last.role is not None
            and (expected is None or last.role is expected)
        ):
            if time.monotonic() >= deadline:
                return last
            self.sleeper(min(0.01, max(0.0, deadline - time.monotonic())))
            last = self.observe(arm_id)
        return last

    def observe_many(self, arm_ids: Sequence[ArmId]) -> dict[ArmId, RoleObservation]:
        return {arm_id: self.observe(arm_id) for arm_id in arm_ids}

    def observe_confirmed(self, arm_id: ArmId) -> RoleObservation:
        """Poll within the configured role window for fresh role traffic."""

        return self._observe_confirmed(arm_id)

    def observe_confirmed_many(
        self,
        arm_ids: Sequence[ArmId],
    ) -> dict[ArmId, RoleObservation]:
        return {
            arm_id: self.observe_confirmed(arm_id)
            for arm_id in arm_ids
        }

    def last_confirmed_role(self, arm_id: ArmId) -> LinkageRole | None:
        """Return only a role previously confirmed by fresh role traffic."""

        with self._lock:
            return self._known.get(arm_id)

    def confirm_pending(self, arm_id: ArmId, target: LinkageRole) -> None:
        """Promote an accepted role after independent target-traffic proof."""

        with self._lock:
            if (
                self._pending.get(arm_id) is not target
                and self._known.get(arm_id) is not target
            ):
                raise RuntimeError(
                    f"{arm_id.value} has no accepted {target.short_name} role"
                )
            self._known[arm_id] = target
            self._pending.pop(arm_id, None)

    def ensure_one(
        self,
        arm_id: ArmId,
        target: LinkageRole,
        *,
        force: bool = False,
        after_write: Callable[[ArmId, LinkageRole], bool] | None = None,
    ) -> RoleWriteResult:
        before = self._observe_confirmed(arm_id)
        if not force and before.role is target and before.fresh:
            return RoleWriteResult(
                arm_id=arm_id,
                previous=before.role,
                target=target,
                written=False,
                verified=before.fresh,
                observation=before,
            )
        with self._lock:
            pending = self._pending.get(arm_id)
            known = self._known.get(arm_id)
        if not force and (pending is target or known is target):
            return RoleWriteResult(
                arm_id=arm_id,
                previous=known,
                target=target,
                written=False,
                verified=False,
                observation=before,
                pending=True,
            )
        motion_output_seeded = False
        try:
            # Once a role write is attempted, the previous role is no longer a
            # safe fallback until fresh traffic confirms the resulting role.
            with self._lock:
                self._known.pop(arm_id, None)
                self._pending.pop(arm_id, None)
            self.arms[arm_id].configure_linkage_role(target)
            with self._lock:
                self._pending[arm_id] = target
            observed = self._observe_confirmed(arm_id, expected=target)
            verified = observed.role is target and observed.fresh
            verification_error: BaseException | None = None
            pending = False
            if verified:
                motion_output_seeded = bool(
                    after_write(arm_id, target)
                    if after_write is not None
                    else False
                )
                with self._lock:
                    self._known[arm_id] = target
                    self._pending.pop(arm_id, None)
            elif target is LinkageRole.TEACHING_INPUT:
                pending = True
            else:
                verification_error = TimeoutError(
                    f"0x470 {target.short_name} write was not verified by fresh expected traffic, "
                    f"observed: {observed.evidence}"
                )
                with self._lock:
                    self._pending.pop(arm_id, None)
            return RoleWriteResult(
                arm_id=arm_id,
                previous=before.role,
                target=target,
                written=True,
                verified=verified,
                observation=observed,
                error=verification_error,
                motion_output_seeded=motion_output_seeded,
                pending=pending,
            )
        except BaseException as exc:
            with self._lock:
                self._pending.pop(arm_id, None)
            return RoleWriteResult(
                arm_id=arm_id,
                previous=before.role,
                target=target,
                written=True,
                verified=False,
                observation=self.observe(arm_id),
                error=exc,
                motion_output_seeded=motion_output_seeded,
            )

    def _ensure_parallel(
        self,
        arm_ids: Sequence[ArmId],
        target: LinkageRole,
        *,
        force: bool = False,
        after_write: Callable[[ArmId, LinkageRole], bool] | None = None,
    ) -> tuple[RoleWriteResult, ...]:
        with ThreadPoolExecutor(max_workers=len(arm_ids), thread_name_prefix="piper_role") as executor:
            futures = {
                executor.submit(
                    self.ensure_one,
                    arm_id,
                    target,
                    force=force,
                    after_write=after_write,
                ): arm_id
                for arm_id in arm_ids
            }
            by_id: dict[ArmId, RoleWriteResult] = {}
            for future in as_completed(futures):
                arm_id = futures[future]
                try:
                    by_id[arm_id] = future.result()
                except BaseException as exc:
                    by_id[arm_id] = RoleWriteResult(
                        arm_id=arm_id,
                        previous=None,
                        target=target,
                        written=False,
                        verified=False,
                        observation=RoleObservation.unknown(arm_id, "role worker failed"),
                        error=exc,
                    )
        return tuple(by_id[arm_id] for arm_id in arm_ids)

    def ensure_pair(
        self,
        arm_ids: Sequence[ArmId],
        target: LinkageRole,
        *,
        retry_once: bool = True,
        after_write: Callable[[ArmId, LinkageRole], bool] | None = None,
        rollback_after_write_factory: Callable[
            [], Callable[[ArmId, LinkageRole], bool] | None
        ]
        | None = None,
    ) -> RoleTransactionResult:
        if len(arm_ids) != 2:
            raise ValueError("a linkage role transaction requires exactly two arms")
        all_writes: list[RoleWriteResult] = []
        all_rollbacks: list[RoleWriteResult] = []
        max_attempts = 2 if retry_once else 1

        for attempt in range(1, max_attempts + 1):
            # Attempt one observes and skips already-correct roles.  A retry is
            # deliberately a whole paired transaction after both arms have
            # been returned to verified FC.
            writes = self._ensure_parallel(
                arm_ids,
                target,
                force=attempt > 1,
                after_write=after_write,
            )
            all_writes.extend(writes)
            if all(result.accepted for result in writes):
                return RoleTransactionResult(
                    target=target,
                    attempts=attempt,
                    writes=tuple(all_writes),
                    rollbacks=tuple(all_rollbacks),
                )

            # A mixed FA/FC master pair is not an acceptable rollback state.
            # Capture a new seed before every rollback because a master that
            # successfully entered FA may have moved while its peer timed out.
            # Best-effort write and verify FC on both sides after every failed
            # attempt, including the final one.
            rollback_after_write = (
                rollback_after_write_factory()
                if rollback_after_write_factory is not None
                else after_write
            )
            rollback_results = self._ensure_parallel(
                arm_ids,
                LinkageRole.MOTION_OUTPUT,
                force=True,
                after_write=rollback_after_write,
            )
            all_rollbacks.extend(rollback_results)
            rollback_ok = len(rollback_results) == 2 and all(
                result.ok for result in rollback_results
            )
            if not rollback_ok or attempt == max_attempts:
                raise RoleTransactionError(
                    f"paired 0x470 transaction to {target.short_name} failed after {attempt} attempt(s)",
                    target=target,
                    attempts=attempt,
                    writes=tuple(all_writes),
                    rollbacks=tuple(all_rollbacks),
                )
class _IsolatedBase:
    def __init__(
        self,
        arms: Mapping[ArmId, ArmLike],
        *,
        role_observer: RoleObserver | None = None,
        role_verify_timeout_s: float | None = None,
        motion_watchdog_max_state_age_s: float = DEFAULT_FRAME_FRESHNESS_S,
        slave_gripper_efforts: Mapping[ArmId, int] | None = None,
        commands_enabled: bool | None = None,
        prefer_joint_ctrl: bool = False,
        name: str = "dual_piper",
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if role_observer is None:
            raise TopologyConfigError(
                "isolated hardware requires an explicit role observer"
            )
        self.physical_arms = dict(arms)
        self.commands_enabled = (
            bool(commands_enabled)
            if commands_enabled is not None
            else all(bool(getattr(arm, "commands_enabled", True)) for arm in arms.values())
        )
        self.prefer_joint_ctrl = bool(prefer_joint_ctrl)
        self.name = str(name)
        self.sleeper = sleeper
        if (
            motion_watchdog_max_state_age_s <= 0.0
            or not np.isfinite(motion_watchdog_max_state_age_s)
        ):
            raise TopologyConfigError(
                "motion_watchdog_max_state_age_s must be finite and positive"
            )
        self.motion_watchdog_max_state_age_s = motion_watchdog_max_state_age_s
        self.role_controller = LinkageRoleController(
            self.physical_arms,
            observer=role_observer,
            verify_timeout_s=role_verify_timeout_s,
            sleeper=sleeper,
        )
        self._fault: FaultRecord | None = None
        self._effective_slave_gripper_efforts: dict[ArmId, int] = {}
        for arm_id in SLAVE_ARM_IDS:
            if arm_id not in self.physical_arms:
                continue
            configured = gripper_effort_value(
                None
                if slave_gripper_efforts is None
                else slave_gripper_efforts.get(arm_id)
            )
            self._effective_slave_gripper_efforts[arm_id] = configured
        self._state_lock = threading.RLock()
        self._motion_gate = threading.RLock()
        self._last_verified_joint_targets: dict[ArmId, np.ndarray] = {}
        self._transitioning = False
        self._generation = 0
        self.last_enable_operations: tuple[
            BatchOperationResult | RoleTransactionResult,
            ...,
        ] = ()

    @property
    def faulted(self) -> bool:
        return self._fault is not None

    @property
    def fault(self) -> FaultRecord | None:
        return self._fault

    @property
    def generation(self) -> int:
        with self._state_lock:
            return self._generation

    @property
    def effective_slave_gripper_efforts(self) -> Mapping[ArmId, int]:
        with self._state_lock:
            return dict(self._effective_slave_gripper_efforts)

    def gripper_full_opening_m(self, side: str) -> float:
        if side not in ("left", "right"):
            raise ValueError("Piper side must be 'left' or 'right'")
        if not any(arm_id.side == side for arm_id in self.physical_arms):
            raise ValueError(f"isolated runtime does not contain a {side} arm")
        return PIPER_GRIPPER_FULL_OPEN_METERS

    def action_state_for_joint_targets(
        self,
        left_qpos: Iterable[float],
        right_qpos: Iterable[float],
        observation_state: DualPiperState,
    ) -> DualPiperState:
        """Build `/action` from the exact targets submitted on isolated CANs."""

        return build_dual_joint_target_action_state(
            left_qpos,
            right_qpos,
            observation_state,
        )

    def read_state(self, *, prefer_joint_ctrl: bool | None = None) -> DualPiperState:
        use_ctrl = self.prefer_joint_ctrl if prefer_joint_ctrl is None else bool(prefer_joint_ctrl)
        return DualPiperState(
            left=self.left.read_state(prefer_joint_ctrl=use_ctrl),
            right=self.right.read_state(prefer_joint_ctrl=use_ctrl),
        )

    def action_state_for_end_pose_targets(
        self,
        left_pose: Iterable[float],
        right_pose: Iterable[float],
        observation_state: DualPiperState,
    ) -> DualPiperState:
        """Build EE `/action` with master joints only when masters are present."""

        if not isinstance(observation_state, DualPiperState):
            raise TypeError("isolated EE action state requires DualPiperState feedback")
        base_state = observation_state
        if all(arm_id in self.physical_arms for arm_id in MASTER_ARM_IDS):
            physical = self.read_physical_states()
            base_state = DualPiperState(
                left=physical[ArmId.MASTER_LEFT],
                right=physical[ArmId.MASTER_RIGHT],
            )
        targets = {
            "left": np.asarray(list(left_pose), dtype=np.float64),
            "right": np.asarray(list(right_pose), dtype=np.float64),
        }
        command_timestamp_s = time.time()

        def commanded_arm(side: str, state: PiperArmState) -> PiperArmState:
            target = targets[side]
            if target.shape != (7,) or not np.all(np.isfinite(target)):
                raise ValueError(f"isolated {side} EE action target must be finite 7-D")
            qpos = np.asarray(state.qpos, dtype=np.float64).copy()
            qpos[6] = target[6]
            qpos_command = np.asarray(state.qpos_command, dtype=np.float64).copy()
            qpos_command[6] = target[6]
            return replace(
                state,
                qpos=qpos,
                qpos_command=qpos_command,
                end_pose=target.copy(),
                timestamp_s=max(float(state.timestamp_s), command_timestamp_s),
                end_pose_timestamp_s=command_timestamp_s,
                command_timestamp_s=command_timestamp_s,
            )

        return DualPiperState(
            left=commanded_arm("left", base_state.left),
            right=commanded_arm("right", base_state.right),
        )

    def _latch_fault(
        self,
        operation: str,
        cause: BaseException | str,
        *,
        arm_id: ArmId | None = None,
        partial_result: BatchOperationResult | None = None,
    ) -> None:
        with self._state_lock:
            if self._fault is None:
                self._fault = FaultRecord(
                    operation=operation,
                    message=str(cause),
                    arm_id=arm_id,
                    partial_result=partial_result,
                )

    def _remember_verified_joint_targets(
        self,
        targets: Mapping[ArmId, Iterable[float]],
    ) -> None:
        """Keep only targets that were freshly observed or successfully sent.

        Fault handling must never turn an arbitrary SDK cache entry into a new
        command.  This cache is therefore updated only after a complete health
        validation or a successful FC joint-position submission.
        """

        copied: dict[ArmId, np.ndarray] = {}
        for arm_id, values in targets.items():
            target = np.asarray(values, dtype=np.float64)
            if target.shape != (7,) or not np.all(np.isfinite(target)):
                continue
            copied[arm_id] = target.copy()
        if not copied:
            return
        with self._state_lock:
            self._last_verified_joint_targets.update(copied)

    def _last_verified_joint_target(self, arm_id: ArmId) -> np.ndarray | None:
        with self._state_lock:
            target = self._last_verified_joint_targets.get(arm_id)
            return None if target is None else target.copy()

    def _require_motion_available(self) -> int:
        with self._state_lock:
            if self._fault is not None:
                raise LatchedFaultError(self._fault)
            if self._transitioning:
                raise TransitionInProgressError("hardware role transition is in progress")
            return self._generation

    def _check_generation(self, generation: int) -> None:
        with self._state_lock:
            if self._fault is not None:
                raise LatchedFaultError(self._fault)
            if self._transitioning or generation != self._generation:
                raise TransitionInProgressError(
                    f"stale hardware generation {generation}; current generation is {self._generation}"
                )

    @contextmanager
    def _hardware_transaction(self):
        # The gate spans the whole transaction.  Policy fan-out and role swaps
        # therefore cannot pass one another between a generation check and TX.
        with self._motion_gate:
            with self._state_lock:
                if self._fault is not None:
                    raise LatchedFaultError(self._fault)
                if self._transitioning:
                    raise TransitionInProgressError("another hardware transition is in progress")
                self._transitioning = True
                self._generation += 1
                generation = self._generation
            try:
                yield generation
            finally:
                with self._state_lock:
                    self._transitioning = False

    def _parallel(
        self,
        operation: str,
        arm_ids: Sequence[ArmId],
        call: Callable[[ArmId, ArmLike], Any],
    ) -> BatchOperationResult:
        def invoke(arm_id: ArmId) -> ArmOperationResult:
            started = time.time()
            try:
                value = call(arm_id, self.physical_arms[arm_id])
                return ArmOperationResult(arm_id, operation, started, time.time(), value=value)
            except BaseException as exc:
                return ArmOperationResult(arm_id, operation, started, time.time(), error=exc)

        with ThreadPoolExecutor(max_workers=len(arm_ids), thread_name_prefix=f"piper_{operation}") as executor:
            futures = {executor.submit(invoke, arm_id): arm_id for arm_id in arm_ids}
            by_id = {futures[future]: future.result() for future in as_completed(futures)}
        return BatchOperationResult(operation, tuple(by_id[arm_id] for arm_id in arm_ids))

    def _require_batch(self, result: BatchOperationResult, *, latch: bool = True) -> BatchOperationResult:
        if result.ok:
            return result
        first = result.failures[0]
        if latch:
            self._latch_fault(
                result.operation,
                first.error or "unknown failure",
                arm_id=first.arm_id,
                partial_result=result,
            )
        raise BatchOperationError(result)

    def connect(self, *, read_only: bool = True) -> BatchOperationResult:
        # Commands are gated by ``commands_enabled``.  Isolated runtime startup
        # deliberately skips piper_sdk PiperInit(), whose implicit firmware and
        # limit queries are not part of a motion session.  Shared uses the
        # original SinglePiperArm.connect behavior unchanged.
        result = self._parallel(
            "connect",
            tuple(self.physical_arms),
            lambda arm_id, arm: arm.connect(read_only=True),
        )
        return self._require_batch(result)

    def _wait_for_driver_feedback(
        self,
        arm_ids: Sequence[ArmId],
    ) -> BatchOperationResult:
        timeout_s = max(
            self.role_controller.verify_timeout_s,
            self.motion_watchdog_max_state_age_s,
        )

        def require_feedback_cache(arm_id: ArmId, arm: ArmLike) -> None:
            complete = getattr(arm, "has_complete_driver_feedback", None)
            driver_feedback_ready = not callable(complete) or complete()
            position_error: ArmOperationError | None = None
            try:
                self._require_fresh_position_family(
                    arm_id,
                    prefer_joint_ctrl=False,
                    max_state_age_s=timeout_s,
                    operation="connect",
                    latch_fault=False,
                )
            except ArmOperationError as exc:
                position_error = exc
            if not driver_feedback_ready:
                raise ArmOperationError(
                    arm_id,
                    "connect",
                    "FC feedback cache is incomplete",
                )
            if position_error is not None:
                raise position_error

        deadline_s = time.monotonic() + timeout_s
        while True:
            ready = self._parallel(
                "wait for driver feedback",
                arm_ids,
                require_feedback_cache,
            )
            if ready.ok:
                return ready
            remaining_s = deadline_s - time.monotonic()
            if remaining_s <= 0.0:
                return self._require_batch(ready)
            self.sleeper(min(DRIVER_FEEDBACK_POLL_INTERVAL_S, remaining_s))

    def disconnect(self) -> BatchOperationResult:
        # Disconnect and state reads remain available after a fault latch.
        def disconnect_arm(arm_id: ArmId, arm: ArmLike) -> None:
            if getattr(arm, "connected", None) is False:
                arm.abort_construction()
                return
            arm.disconnect()

        return self._parallel(
            "disconnect",
            tuple(self.physical_arms),
            disconnect_arm,
        )

    def abort_construction(self) -> BatchOperationResult:
        """Best-effort cleanup for an isolated runtime that never connected."""

        def abort_arm(arm_id: ArmId, arm: ArmLike) -> None:
            arm.abort_construction()

        return self._parallel(
            "abort isolated construction",
            tuple(self.physical_arms),
            abort_arm,
        )

    def _enable_arms(
        self,
        arm_ids: Sequence[ArmId],
        *,
        retries: int = 5,
        sleep_s: float = 0.5,
    ) -> BatchOperationResult:
        def enable_arm(arm_id: ArmId, arm: ArmLike) -> bool:
            if arm.is_enabled() or arm.enable_without_reset(
                retries=retries,
                sleep_s=sleep_s,
            ):
                return True
            raise ArmOperationError(arm_id, "enable", "arm did not report enabled")

        result = self._parallel(
            "enable",
            tuple(arm_ids),
            enable_arm,
        )
        return self._require_batch(result)

    def enable_all(self, *, retries: int = 5, sleep_s: float = 0.5) -> BatchOperationResult:
        return self._enable_arms(
            tuple(self.physical_arms),
            retries=retries,
            sleep_s=sleep_s,
        )

    def enable(self, *, retries: int = 5, sleep_s: float = 0.5) -> bool:
        """Runner-compatible, reset-free enable with observe-first FC roles."""

        try:
            operations: list[BatchOperationResult | RoleTransactionResult] = []
            with self._hardware_transaction():
                max_state_age_s = self._resolved_motion_watchdog_age()
                if all(arm_id in self.physical_arms for arm_id in SLAVE_ARM_IDS):
                    operations.append(
                        self._configure_pair_internal(
                            SLAVE_ARM_IDS,
                            LinkageRole.MOTION_OUTPUT,
                            speed_percent=50,
                            gripper_effort=None,
                        )
                    )
                if all(arm_id in self.physical_arms for arm_id in MASTER_ARM_IDS):
                    operations.append(
                        self._configure_pair_internal(
                            MASTER_ARM_IDS,
                            LinkageRole.MOTION_OUTPUT,
                            speed_percent=50,
                            gripper_effort=None,
                        )
                    )
                seed_states, _, _ = self._capture_role_aware_states(
                    arm_ids=tuple(self.physical_arms),
                    max_state_age_s=max_state_age_s,
                    require_enabled=False,
                    operation="capture FC arms before enable",
                )
                operations.append(
                    self._command_captured_holds(
                        seed_states,
                        tuple(self.physical_arms),
                        speed_percent=50,
                        gripper_effort=None,
                    )
                )
                operations.append(
                    self.enable_all(retries=retries, sleep_s=sleep_s)
                )
            self.last_enable_operations = tuple(operations)
            return True
        except BaseException as exc:
            if not self.faulted:
                self._latch_fault("enable isolated hardware", exc)
            raise

    def read_physical_states(
        self,
        *,
        prefer_joint_ctrl_arm_ids: Sequence[ArmId] = (),
    ) -> PhysicalArmStates:
        control_ids = frozenset(prefer_joint_ctrl_arm_ids)
        result = self._parallel(
            "read physical states",
            tuple(self.physical_arms),
            lambda arm_id, arm: arm.read_state(prefer_joint_ctrl=arm_id in control_ids),
        )
        self._require_batch(result, latch=False)
        return PhysicalArmStates(
            states={item.arm_id: item.value for item in result.results},
            captured_at_s=time.time(),
        )

    def _capture_role_aware_states(
        self,
        *,
        arm_ids: Sequence[ArmId],
        max_state_age_s: float,
        require_enabled: bool,
        operation: str,
        role_observations: Mapping[ArmId, RoleObservation] | None = None,
    ) -> tuple[
        PhysicalArmStates,
        tuple[ArmId, ...],
        dict[ArmId, RoleObservation],
    ]:
        selected_arm_ids = tuple(arm_ids)
        observations = (
            self.role_controller.observe_many(selected_arm_ids)
            if role_observations is None
            else {
                arm_id: role_observations[arm_id]
                for arm_id in selected_arm_ids
            }
        )
        resolved_roles = {
            arm_id: (
                observation.role
                if observation.fresh
                else self.role_controller.last_confirmed_role(arm_id)
            )
            for arm_id, observation in observations.items()
        }
        motion_output_ids = tuple(
            arm_id
            for arm_id, role in resolved_roles.items()
            if role is LinkageRole.MOTION_OUTPUT
        )
        if motion_output_ids:
            self._wait_for_driver_feedback(motion_output_ids)
        feedback_states = self.read_physical_states()
        control_candidate_ids = tuple(
            arm_id
            for arm_id, role in resolved_roles.items()
            if role is not LinkageRole.MOTION_OUTPUT
        )
        control_states = (
            self.read_physical_states(
                prefer_joint_ctrl_arm_ids=control_candidate_ids,
            )
            if control_candidate_ids
            else feedback_states
        )
        selected_states = {
            arm_id: feedback_states[arm_id]
            for arm_id in selected_arm_ids
        }
        control_ids: list[ArmId] = []
        for arm_id in selected_arm_ids:
            resolved_role = resolved_roles[arm_id]
            use_control = resolved_role is LinkageRole.TEACHING_INPUT
            if resolved_role is None:
                feedback_timestamp_s = float(
                    feedback_states[arm_id].qpos_timestamp_s
                )
                control_timestamp_s = float(
                    control_states[arm_id].qpos_timestamp_s
                )
                use_control = control_timestamp_s > feedback_timestamp_s
            if use_control:
                selected_states[arm_id] = control_states[arm_id]
                control_ids.append(arm_id)

        states = PhysicalArmStates(
            states=selected_states,
            captured_at_s=time.time(),
        )
        self._require_fresh_positions(
            states,
            selected_arm_ids,
            max_state_age_s=max_state_age_s,
            operation=operation,
            prefer_joint_ctrl_arm_ids=tuple(control_ids),
        )
        if require_enabled:
            try:
                self._validate_health_states(
                    states,
                    max_state_age_s=max_state_age_s,
                    prefer_joint_ctrl_arm_ids=tuple(control_ids),
                )
            except BaseException as exc:
                if not self.faulted:
                    failed_arm_id = (
                        exc.arm_id
                        if isinstance(exc, ArmOperationError)
                        else None
                    )
                    self._latch_fault(operation, exc, arm_id=failed_arm_id)
                raise
        return states, tuple(control_ids), observations

    def require_healthy(
        self,
        *,
        max_state_age_s: float | None = None,
        prefer_joint_ctrl_arm_ids: Sequence[ArmId] = (),
    ) -> PhysicalArmStates:
        """Validate every constructed arm from fresh cached SDK frame families."""

        if max_state_age_s is None:
            max_state_age_s = self._resolved_motion_watchdog_age()
        if max_state_age_s <= 0.0:
            raise ValueError("max_state_age_s must be positive")
        if self.faulted:
            raise LatchedFaultError(self.fault)
        try:
            states = self.read_physical_states(
                prefer_joint_ctrl_arm_ids=prefer_joint_ctrl_arm_ids,
            )
            self._validate_health_states(
                states,
                max_state_age_s=max_state_age_s,
                prefer_joint_ctrl_arm_ids=prefer_joint_ctrl_arm_ids,
            )
            self._remember_verified_joint_targets(
                {
                    arm_id: state.qpos
                    for arm_id, state in states.states.items()
                    if arm_id not in prefer_joint_ctrl_arm_ids
                }
            )
            return states
        except BaseException as exc:
            if not self.faulted:
                arm_id = exc.arm_id if isinstance(exc, ArmOperationError) else None
                self._latch_fault("health check", exc, arm_id=arm_id)
            raise

    def _validate_health_states(
        self,
        states: PhysicalArmStates,
        *,
        max_state_age_s: float,
        prefer_joint_ctrl_arm_ids: Sequence[ArmId] = (),
    ) -> None:
        control_ids = frozenset(prefer_joint_ctrl_arm_ids)
        now_s = time.time()
        for arm_id, state in states.states.items():
            # FA control traffic is event-driven: a stationary teaching arm has
            # no periodic position family to use as a health heartbeat.  The
            # gateway validates every target family when one is produced; the
            # continuous hardware health gate therefore remains on FC arms.
            if arm_id in control_ids:
                continue
            self._require_fresh_position_family(
                arm_id,
                prefer_joint_ctrl=arm_id in control_ids,
                max_state_age_s=max_state_age_s,
                operation="health check",
            )
            position_timestamps = (
                float(getattr(state, "qpos_timestamp_s", 0.0)),
                float(getattr(state, "gripper_position_timestamp_s", 0.0)),
            )
            if (
                min(position_timestamps) <= 0.0
                or not 0.0
                <= now_s - min(position_timestamps)
                <= max_state_age_s
            ):
                raise ArmOperationError(
                    arm_id,
                    "health check",
                    "joint/gripper position frame family is stale",
                )
            if not bool(state.enabled):
                raise ArmOperationError(
                    arm_id,
                    "health check",
                    "arm is not enabled",
                )
            errors = getattr(state, "status", {}).get("errors", {})
            active_errors = [
                name for name, active in errors.items() if active
            ]
            if active_errors:
                raise ArmOperationError(
                    arm_id,
                    "health check",
                    f"driver errors {active_errors}",
                )
            self._require_fresh_healthy_driver(arm_id)

    def _resolved_motion_watchdog_age(self) -> float:
        return float(self.motion_watchdog_max_state_age_s)

    def _check_interpolation_generation(
        self,
        generation: int,
        *,
        transaction_active: bool,
    ) -> None:
        with self._state_lock:
            if self._fault is not None:
                raise LatchedFaultError(self._fault)
            transition_matches = self._transitioning == transaction_active
            if generation != self._generation or not transition_matches:
                raise TransitionInProgressError(
                    f"stale isolated interpolation generation {generation}; "
                    f"current generation is {self._generation}"
                )

    def _interpolate_joint_targets(
        self,
        targets: Mapping[ArmId, np.ndarray],
        *,
        operation: str,
        generation: int,
        transaction_active: bool,
        hz: float,
        step_sizes: Iterable[float],
        speed_percent: int,
        gripper_effort: int | None,
        gripper_efforts: Mapping[ArmId | str, int] | None,
        initial_states: PhysicalArmStates | None = None,
        prefer_joint_ctrl_arm_ids: Sequence[ArmId] = (),
        command_slave_grippers: bool = True,
    ) -> BatchOperationResult:
        max_state_age_s = self._resolved_motion_watchdog_age()
        if hz <= 0.0 or not np.isfinite(hz):
            raise ValueError("isolated interpolation hz must be finite and positive")
        steps = self._seven_values(
            step_sizes,
            label="step_sizes",
            allow_zero=False,
        )
        arm_ids = tuple(targets)
        self._validate_targets(targets, joint_positions=True)
        states = initial_states or self.require_healthy(
            max_state_age_s=max_state_age_s,
            prefer_joint_ctrl_arm_ids=prefer_joint_ctrl_arm_ids,
        )
        current = {
            arm_id: np.asarray(states[arm_id].qpos, dtype=np.float64).copy()
            for arm_id in arm_ids
        }

        while True:
            self._check_interpolation_generation(
                generation,
                transaction_active=transaction_active,
            )
            next_targets: dict[ArmId, np.ndarray] = {}
            final_cycle = True
            for arm_id in arm_ids:
                difference = targets[arm_id] - current[arm_id]
                arm_final = bool(np.all(np.abs(difference) <= steps))
                final_cycle = final_cycle and arm_final
                next_targets[arm_id] = np.where(
                    np.abs(difference) <= steps,
                    targets[arm_id],
                    current[arm_id] + np.sign(difference) * steps,
                )

            result = self._parallel(
                operation,
                arm_ids,
                lambda arm_id, arm: arm.command_joint_positions(
                    next_targets[arm_id],
                    speed_percent=speed_percent,
                    gripper_effort=self._gripper_effort_for(
                        arm_id,
                        gripper_effort=gripper_effort,
                        gripper_efforts=gripper_efforts,
                    ),
                    command_gripper=command_slave_grippers and arm_id.is_slave,
                ),
            )
            checked = self._require_batch(result)
            self._remember_verified_joint_targets(next_targets)
            current = next_targets
            if final_cycle:
                self.require_healthy(
                    max_state_age_s=max_state_age_s,
                    prefer_joint_ctrl_arm_ids=prefer_joint_ctrl_arm_ids,
                )
                self._record_slave_efforts(
                    arm_ids,
                    gripper_effort=gripper_effort,
                    gripper_efforts=gripper_efforts,
                )
                return checked

            self.sleeper(1.0 / hz)
            self.require_healthy(
                max_state_age_s=max_state_age_s,
                prefer_joint_ctrl_arm_ids=prefer_joint_ctrl_arm_ids,
            )

    def _align_targets_to_sources(
        self,
        *,
        target_arm_ids: Sequence[ArmId],
        source_arm_ids: Sequence[ArmId],
        operation: str,
        hz: float = 30.0,
        step_sizes: Iterable[float] = DEFAULT_ARM_STEP_LENGTH,
        speed_percent: int = 50,
        gripper_effort: int | None = None,
        gripper_efforts: Mapping[ArmId | str, int] | None = None,
        prefer_joint_ctrl_arm_ids: Sequence[ArmId] = (),
        transaction_generation: int,
        preserve_target_gripper: bool = False,
        command_slave_grippers: bool = True,
    ) -> BatchOperationResult:
        max_state_age_s = self._resolved_motion_watchdog_age()
        target_ids = tuple(target_arm_ids)
        source_ids = tuple(source_arm_ids)
        if not target_ids or len(target_ids) != len(source_ids):
            raise ValueError("alignment requires matching target and source arms")
        states = self.require_healthy(
            max_state_age_s=max_state_age_s,
            prefer_joint_ctrl_arm_ids=prefer_joint_ctrl_arm_ids,
        )
        targets = {
            target_id: np.asarray(states[source_id].qpos, dtype=np.float64).copy()
            for target_id, source_id in zip(target_ids, source_ids)
        }
        if preserve_target_gripper:
            for target_id in target_ids:
                targets[target_id][6] = float(states[target_id].qpos[6])
        return self._interpolate_joint_targets(
            targets,
            operation=operation,
            generation=transaction_generation,
            transaction_active=True,
            hz=hz,
            step_sizes=step_sizes,
            speed_percent=speed_percent,
            gripper_effort=gripper_effort,
            gripper_efforts=gripper_efforts,
            initial_states=states,
            prefer_joint_ctrl_arm_ids=prefer_joint_ctrl_arm_ids,
            command_slave_grippers=command_slave_grippers,
        )

    def _configure_pair_internal(
        self,
        arm_ids: Sequence[ArmId],
        role: LinkageRole,
        *,
        speed_percent: int,
        gripper_effort: int | None,
        gripper_efforts: Mapping[ArmId | str, int] | None = None,
    ) -> RoleTransactionResult:
        def prepare_rollback_after_write() -> Callable[[ArmId, LinkageRole], bool]:
            return self._motion_output_current_seed_hook(
                speed_percent=speed_percent,
                gripper_effort=gripper_effort,
                gripper_efforts=gripper_efforts,
            )

        try:
            return self.role_controller.ensure_pair(
                arm_ids,
                role,
                retry_once=True,
                after_write=self._motion_output_current_seed_hook(
                    speed_percent=speed_percent,
                    gripper_effort=gripper_effort,
                    gripper_efforts=gripper_efforts,
                ),
                rollback_after_write_factory=prepare_rollback_after_write,
            )
        except RoleTransactionError as exc:
            failed_arm = next((item.arm_id for item in exc.writes if not item.ok), None)
            self._latch_fault(
                f"paired linkage role {role.short_name}",
                exc,
                arm_id=failed_arm,
            )
            raise

    def configure_master_pair(self, role: LinkageRole) -> RoleTransactionResult:
        with self._motion_gate:
            self._require_motion_available()
            return self._configure_pair_internal(
                MASTER_ARM_IDS,
                role,
                speed_percent=50,
                gripper_effort=None,
            )

    @staticmethod
    def _seven_values(
        values: Iterable[float],
        *,
        label: str,
        allow_zero: bool,
    ) -> np.ndarray:
        array = np.asarray(list(values), dtype=np.float64)
        if array.shape != (7,) or not np.all(np.isfinite(array)):
            raise ValueError(f"{label} must contain seven finite values")
        if np.any(array < 0.0) or (not allow_zero and np.any(array == 0.0)):
            qualifier = "non-negative" if allow_zero else "positive"
            raise ValueError(f"{label} must contain seven {qualifier} values")
        return array

    @staticmethod
    def _paired_slave_id(arm_id: ArmId) -> ArmId:
        return ArmId.SLAVE_LEFT if arm_id.side == "left" else ArmId.SLAVE_RIGHT

    def _gripper_effort_for(
        self,
        arm_id: ArmId,
        *,
        gripper_effort: int | None,
        gripper_efforts: Mapping[ArmId | str, int] | None,
    ) -> int:
        value: int | None = None
        if gripper_efforts is not None:
            value = gripper_efforts.get(arm_id)
            if value is None:
                value = gripper_efforts.get(self._paired_slave_id(arm_id))
        if value is None:
            if gripper_effort is not None:
                value = int(gripper_effort)
            else:
                with self._state_lock:
                    value = self._effective_slave_gripper_efforts[
                        self._paired_slave_id(arm_id)
                    ]
        return gripper_effort_value(value)

    def _record_slave_efforts(
        self,
        arm_ids: Sequence[ArmId],
        *,
        gripper_effort: int | None,
        gripper_efforts: Mapping[ArmId | str, int] | None,
    ) -> None:
        with self._state_lock:
            for arm_id in arm_ids:
                if arm_id.is_slave:
                    self._effective_slave_gripper_efforts[arm_id] = (
                        self._gripper_effort_for(
                            arm_id,
                            gripper_effort=gripper_effort,
                            gripper_efforts=gripper_efforts,
                        )
                    )

    def _validate_target(
        self,
        arm_id: ArmId,
        values: np.ndarray,
        *,
        joint_positions: bool,
    ) -> None:
        if values.shape != (7,):
            target_kind = "joint" if joint_positions else "end-pose"
            raise ValueError(f"{arm_id.value} {target_kind} target must contain seven values")
        if not np.all(np.isfinite(values)):
            error = ArmOperationError(
                arm_id,
                "validate target",
                "target contains NaN or infinity",
            )
            self._latch_fault("validate target", error, arm_id=arm_id)
            raise error

    def _validate_targets(
        self,
        targets: Mapping[ArmId, np.ndarray],
        *,
        joint_positions: bool,
    ) -> None:
        for arm_id, values in targets.items():
            self._validate_target(arm_id, values, joint_positions=joint_positions)

    def validate_bimanual_targets(
        self,
        left_values: Iterable[float],
        right_values: Iterable[float],
        *,
        joint_positions: bool,
    ) -> None:
        """Validate a selected model target without issuing any CAN command."""

        left = np.asarray(list(left_values), dtype=np.float64)
        right = np.asarray(list(right_values), dtype=np.float64)
        targets = {
            arm_id: (left if arm_id.side == "left" else right).copy()
            for arm_id in self.physical_arms
        }
        with self._motion_gate:
            self._require_motion_available()
            self._validate_targets(
                targets,
                joint_positions=joint_positions,
            )

    def _require_fresh_positions(
        self,
        states: PhysicalArmStates,
        arm_ids: Sequence[ArmId],
        *,
        max_state_age_s: float,
        operation: str,
        prefer_joint_ctrl_arm_ids: Sequence[ArmId] = (),
    ) -> None:
        if max_state_age_s <= 0.0:
            raise ValueError("max_state_age_s must be positive")
        now_s = time.time()
        control_ids = frozenset(prefer_joint_ctrl_arm_ids)
        for arm_id in arm_ids:
            self._require_fresh_position_family(
                arm_id,
                prefer_joint_ctrl=arm_id in control_ids,
                max_state_age_s=max_state_age_s,
                operation=operation,
            )
            state = states[arm_id]
            timestamp = float(getattr(state, "qpos_timestamp_s", 0.0))
            if (
                timestamp <= 0.0
                or not 0.0 <= now_s - timestamp <= max_state_age_s
            ):
                exc = ArmOperationError(arm_id, operation, "joint feedback is stale")
                self._latch_fault(operation, exc, arm_id=arm_id)
                raise exc

    def _require_fresh_position_family(
        self,
        arm_id: ArmId,
        *,
        prefer_joint_ctrl: bool,
        max_state_age_s: float,
        operation: str,
        latch_fault: bool = True,
    ) -> None:
        interface = getattr(self.physical_arms[arm_id], "interface", None)
        if interface is None:
            return
        try:
            joint = (
                interface.GetArmJointCtrl()
                if prefer_joint_ctrl
                else interface.GetArmJointMsgs()
            )
            gripper = (
                interface.GetArmGripperCtrl()
                if prefer_joint_ctrl
                else interface.GetArmGripperMsgs()
            )
            timestamps = (
                float(getattr(joint, "time_stamp", 0.0)),
                float(getattr(gripper, "time_stamp", 0.0)),
            )
        except Exception as exc:
            error = ArmOperationError(arm_id, operation, exc)
            if latch_fault:
                self._latch_fault(operation, error, arm_id=arm_id)
            raise error
        now_s = time.time()
        if (
            min(timestamps) <= 0.0
            or not 0.0 <= now_s - min(timestamps) <= max_state_age_s
        ):
            error = ArmOperationError(
                arm_id,
                operation,
                "joint/gripper position frame family is stale or incomplete",
            )
            if latch_fault:
                self._latch_fault(operation, error, arm_id=arm_id)
            raise error

    def _require_fresh_healthy_driver(
        self,
        arm_id: ArmId,
    ) -> None:
        interface = getattr(self.physical_arms[arm_id], "interface", None)
        if interface is None:
            return
        try:
            low_speed = interface.GetArmLowSpdInfoMsgs()
            complete_family = self.physical_arms[
                arm_id
            ].has_complete_driver_feedback()
            statuses = [
                getattr(low_speed, f"motor_{index}").foc_status
                for index in range(1, 7)
            ]
        except Exception as exc:
            error = ArmOperationError(arm_id, "driver watchdog", exc)
            self._latch_fault("driver watchdog", error, arm_id=arm_id)
            raise error
        fault_fields = (
            "voltage_too_low",
            "motor_overheating",
            "driver_overcurrent",
            "driver_overheating",
            "collision_status",
            "driver_error_status",
            "stall_status",
        )
        has_fault = any(
            bool(getattr(status, field, False))
            for status in statuses
            for field in fault_fields
        )
        if not complete_family or has_fault:
            error = ArmOperationError(
                arm_id,
                "driver watchdog",
                "driver family is stale, incomplete, or reports a fault",
            )
            self._latch_fault("driver watchdog", error, arm_id=arm_id)
            raise error

    def _command_captured_holds(
        self,
        states: PhysicalArmStates,
        arm_ids: Sequence[ArmId],
        *,
        speed_percent: int,
        gripper_effort: int | None,
        gripper_efforts: Mapping[ArmId | str, int] | None = None,
        command_slave_grippers: bool = True,
    ) -> BatchOperationResult:
        targets = {
            arm_id: np.asarray(states[arm_id].qpos, dtype=np.float64).copy()
            for arm_id in arm_ids
        }
        self._validate_targets(targets, joint_positions=True)
        result = self._parallel(
            "hold joint position",
            arm_ids,
            lambda arm_id, arm: arm.command_joint_positions(
                targets[arm_id],
                speed_percent=speed_percent,
                gripper_effort=self._gripper_effort_for(
                    arm_id,
                    gripper_effort=gripper_effort,
                    gripper_efforts=gripper_efforts,
                ),
                command_gripper=command_slave_grippers and arm_id.is_slave,
            ),
        )
        checked = self._require_batch(result)
        self._remember_verified_joint_targets(targets)
        self._record_slave_efforts(
            arm_ids,
            gripper_effort=gripper_effort,
            gripper_efforts=gripper_efforts,
        )
        return checked

    def _motion_output_seed_hook(
        self,
        states: PhysicalArmStates,
        *,
        speed_percent: int,
        gripper_effort: int | None,
        gripper_efforts: Mapping[ArmId | str, int] | None = None,
    ) -> Callable[[ArmId, LinkageRole], bool]:
        """Seed and reset-free enable after fresh FC traffic confirms the role."""

        def seed_after_write(arm_id: ArmId, role: LinkageRole) -> bool:
            if role is not LinkageRole.MOTION_OUTPUT:
                return False
            self._command_captured_holds(
                states,
                (arm_id,),
                speed_percent=speed_percent,
                gripper_effort=gripper_effort,
                gripper_efforts=gripper_efforts,
            )
            self.physical_arms[arm_id].send_enable_without_reset()
            for repetition in range(FC_ROLE_SEED_REPETITIONS):
                self._command_captured_holds(
                    states,
                    (arm_id,),
                    speed_percent=speed_percent,
                    gripper_effort=gripper_effort,
                    gripper_efforts=gripper_efforts,
                )
                if repetition + 1 < FC_ROLE_SEED_REPETITIONS:
                    self.sleeper(FC_ROLE_SEED_INTERVAL_S)
            if not self.physical_arms[arm_id].enable_without_reset(
                retries=5,
                sleep_s=0.2,
            ):
                raise ArmOperationError(
                    arm_id,
                    "enter FC",
                    "arm did not report enabled after the role write",
                )
            return True

        return seed_after_write

    def _motion_output_current_seed_hook(
        self,
        *,
        speed_percent: int,
        gripper_effort: int | None,
        gripper_efforts: Mapping[ArmId | str, int] | None = None,
    ) -> Callable[[ArmId, LinkageRole], bool]:
        """Capture fresh FC feedback, then seed and reset-free enable it."""

        def seed_current_position(arm_id: ArmId, role: LinkageRole) -> bool:
            if role is not LinkageRole.MOTION_OUTPUT:
                return False
            observation = self.role_controller.observe_confirmed(arm_id)
            states, _, _ = self._capture_role_aware_states(
                arm_ids=(arm_id,),
                max_state_age_s=self._resolved_motion_watchdog_age(),
                require_enabled=False,
                operation="capture current position after entering FC",
                role_observations={arm_id: observation},
            )
            return self._motion_output_seed_hook(
                states,
                speed_percent=speed_percent,
                gripper_effort=gripper_effort,
                gripper_efforts=gripper_efforts,
            )(arm_id, role)

        return seed_current_position

    def best_effort_hold_motion_output_arms(
        self,
        *,
        speed_percent: int = 50,
        gripper_efforts: Mapping[ArmId | str, int] | None = None,
    ) -> BatchOperationResult:
        """Hold confirmed FC arms without reissuing an unverified SDK cache.

        Fresh joint and gripper feedback is preferred.  If feedback was lost,
        the last freshly verified state or fully successful joint target is
        used.  An arm with neither source is deliberately skipped instead of
        guessing its current position.
        """

        selected = tuple(self.physical_arms)
        max_state_age_s = self.motion_watchdog_max_state_age_s

        def hold_arm(arm_id: ArmId, arm: ArmLike) -> str:
            try:
                observation = self.role_controller.observe(arm_id)
            except Exception as exc:
                observation = None
                observation_error = exc
            else:
                observation_error = None
            if observation is not None and observation.fresh:
                confirmed_role = observation.role
            else:
                confirmed_role = self.role_controller.last_confirmed_role(arm_id)
            if confirmed_role is not LinkageRole.MOTION_OUTPUT:
                if observation_error is not None:
                    return f"skipped: linkage role unavailable ({observation_error!r})"
                return "skipped: arm has no last-confirmed FC role"

            target: np.ndarray | None = None
            if max_state_age_s is not None:
                try:
                    state = arm.read_state(prefer_joint_ctrl=False)
                    self._require_fresh_position_family(
                        arm_id,
                        prefer_joint_ctrl=False,
                        max_state_age_s=float(max_state_age_s),
                        operation="best-effort fault hold",
                        latch_fault=False,
                    )
                    timestamp_s = float(getattr(state, "qpos_timestamp_s", 0.0))
                    age_s = time.time() - timestamp_s
                    candidate = np.asarray(state.qpos, dtype=np.float64)
                    if (
                        candidate.shape == (7,)
                        and np.all(np.isfinite(candidate))
                        and timestamp_s > 0.0
                        and 0.0 <= age_s <= float(max_state_age_s)
                    ):
                        target = candidate.copy()
                except Exception:
                    target = None
            if target is None:
                target = self._last_verified_joint_target(arm_id)
            if target is None:
                return "skipped: no fresh feedback or verified safe target"

            arm.command_joint_positions(
                target,
                speed_percent=speed_percent,
                gripper_effort=self._gripper_effort_for(
                    arm_id,
                    gripper_effort=None,
                    gripper_efforts=gripper_efforts,
                ),
                command_gripper=arm_id.is_slave,
            )
            return "held"

        result = self._parallel("best-effort fault hold", selected, hold_arm)
        if not result.ok:
            raise BatchOperationError(result)
        return result

    def _dispatch_targets(
        self,
        targets: Mapping[ArmId, np.ndarray],
        *,
        method: str,
        operation: str,
        speed_percent: int,
        gripper_effort: int | None,
        gripper_efforts: Mapping[ArmId | str, int] | None,
    ) -> BatchOperationResult:
        arm_ids = tuple(targets)
        with self._motion_gate:
            generation = self._require_motion_available()
            self._validate_targets(
                targets,
                joint_positions=method == "command_joint_positions",
            )
            self._check_generation(generation)
            self.require_healthy(
                max_state_age_s=self._resolved_motion_watchdog_age(),
            )
            self._check_generation(generation)
            result = self._parallel(
                operation,
                arm_ids,
                lambda arm_id, arm: getattr(arm, method)(
                    targets[arm_id],
                    speed_percent=speed_percent,
                    gripper_effort=self._gripper_effort_for(
                        arm_id,
                        gripper_effort=gripper_effort,
                        gripper_efforts=gripper_efforts,
                    ),
                    command_gripper=arm_id.is_slave,
                ),
            )
            checked = self._require_batch(result)
            if method == "command_joint_positions":
                self._remember_verified_joint_targets(targets)
            self._record_slave_efforts(
                arm_ids,
                gripper_effort=gripper_effort,
                gripper_efforts=gripper_efforts,
            )
            return checked

    def _command_bimanual(
        self,
        left_values: Iterable[float],
        right_values: Iterable[float],
        *,
        method: str,
        speed_percent: int,
        gripper_effort: int | None,
        gripper_efforts: Mapping[ArmId | str, int] | None,
    ) -> BatchOperationResult:
        left = np.asarray(list(left_values), dtype=np.float64)
        right = np.asarray(list(right_values), dtype=np.float64)
        targets = {
            arm_id: (left if arm_id.side == "left" else right).copy()
            for arm_id in self.physical_arms
        }
        return self._dispatch_targets(
            targets,
            method=method,
            operation=f"bimanual {method}",
            speed_percent=speed_percent,
            gripper_effort=gripper_effort,
            gripper_efforts=gripper_efforts,
        )

    def _side_arm_ids(self, side: str, *, mirrored: bool) -> tuple[ArmId, ...]:
        return tuple(
            arm_id
            for arm_id in self.physical_arms
            if arm_id.side == side and (mirrored or arm_id.is_slave)
        )

    def _dispatch_side(
        self,
        side: str,
        method: str,
        values: np.ndarray,
        *,
        mirrored: bool,
        speed_percent: int,
        gripper_effort: int | None,
    ) -> BatchOperationResult:
        arm_ids = self._side_arm_ids(side, mirrored=mirrored)
        return self._dispatch_targets(
            {arm_id: values.copy() for arm_id in arm_ids},
            method=method,
            operation=f"{side} {'mirrored' if mirrored else 'task'} {method}",
            speed_percent=speed_percent,
            gripper_effort=gripper_effort,
            gripper_efforts=None,
        )

    def move_to_joint_positions(
        self,
        qpos: Iterable[float],
        *,
        hz: float = 30.0,
        step_sizes: Iterable[float] = DEFAULT_ARM_STEP_LENGTH,
        speed_percent: int = 100,
        gripper_effort: int | None = None,
    ) -> BatchOperationResult:
        values = np.asarray(list(qpos), dtype=np.float64)
        if values.shape != (14,):
            raise ValueError(f"{self.name} expects a 14-D target, got {values.shape}")
        targets = {
            arm_id: values[:7].copy() if arm_id.side == "left" else values[7:].copy()
            for arm_id in self.physical_arms
        }
        with self._motion_gate:
            generation = self._require_motion_available()
            return self._interpolate_joint_targets(
                targets,
                operation="move isolated arms",
                generation=generation,
                transaction_active=False,
                hz=hz,
                step_sizes=step_sizes,
                speed_percent=speed_percent,
                gripper_effort=gripper_effort,
                gripper_efforts=None,
            )

    def command_bimanual_joint_positions(
        self,
        left_qpos: Iterable[float],
        right_qpos: Iterable[float],
        *,
        speed_percent: int = 100,
        gripper_effort: int | None = None,
        gripper_efforts: Mapping[ArmId | str, int] | None = None,
    ) -> BatchOperationResult:
        return self._command_bimanual(
            left_qpos,
            right_qpos,
            method="command_joint_positions",
            speed_percent=speed_percent,
            gripper_effort=gripper_effort,
            gripper_efforts=gripper_efforts,
        )

    def command_bimanual_end_poses(
        self,
        left_pose: Iterable[float],
        right_pose: Iterable[float],
        *,
        speed_percent: int = 50,
        gripper_effort: int | None = None,
        gripper_efforts: Mapping[ArmId | str, int] | None = None,
    ) -> BatchOperationResult:
        return self._command_bimanual(
            left_pose,
            right_pose,
            method="command_end_pose",
            speed_percent=speed_percent,
            gripper_effort=gripper_effort,
            gripper_efforts=gripper_efforts,
        )


class IsolatedArmEndpoint:
    """Client-compatible per-side endpoint whose state always belongs to the slave."""

    def __init__(self, owner: "IsolatedFourArmSystem | IsolatedSlaveSystem", side: str, *, mirror: bool) -> None:
        if side not in ("left", "right"):
            raise ValueError(f"unsupported arm side {side!r}")
        self.owner = owner
        self.side = side
        self.mirror = mirror
        self.arm_id = ArmId.SLAVE_LEFT if side == "left" else ArmId.SLAVE_RIGHT

    @property
    def physical_arm(self) -> ArmLike:
        return self.owner.physical_arms[self.arm_id]

    @property
    def name(self) -> str:
        return self.physical_arm.name

    @property
    def can_name(self) -> str:
        return self.physical_arm.can_name

    @property
    def interface(self) -> Any:
        return getattr(self.physical_arm, "interface", None)

    @property
    def commands_enabled(self) -> bool:
        return bool(getattr(self.physical_arm, "commands_enabled", self.owner.commands_enabled))

    def read_state(self, *, prefer_joint_ctrl: bool = False) -> Any:
        return self.physical_arm.read_state(prefer_joint_ctrl=prefer_joint_ctrl)

    def command_joint_positions(
        self,
        qpos: Iterable[float],
        *,
        speed_percent: int = 100,
        gripper_effort: int | None = None,
    ) -> BatchOperationResult:
        values = np.asarray(list(qpos), dtype=np.float64)
        return self.owner._dispatch_side(
            self.side,
            "command_joint_positions",
            values,
            mirrored=self.mirror,
            speed_percent=speed_percent,
            gripper_effort=gripper_effort,
        )

    def command_end_pose(
        self,
        pose: Iterable[float],
        *,
        speed_percent: int = 50,
        gripper_effort: int | None = None,
    ) -> BatchOperationResult:
        values = np.asarray(list(pose), dtype=np.float64)
        return self.owner._dispatch_side(
            self.side,
            "command_end_pose",
            values,
            mirrored=self.mirror,
            speed_percent=speed_percent,
            gripper_effort=gripper_effort,
        )


def _isolated_config_kwargs(
    robot_config: Mapping[str, Any],
    kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    configured = dict(kwargs)
    if "role_observer" not in configured:
        configured["role_observer"] = SdkTrafficRoleObserver()
    configured.setdefault(
        "motion_watchdog_max_state_age_s",
        DEFAULT_ROLE_VERIFY_TIMEOUT_S,
    )
    configured.setdefault(
        "slave_gripper_efforts",
        {
            arm_id: int(robot_config[arm_id.value].get("gripper_effort", DEFAULT_GRIPPER_EFFORT))
            for arm_id in SLAVE_ARM_IDS
        },
    )
    return configured


def _abort_arm_construction_quietly(arm: ArmLike) -> None:
    try:
        arm.abort_construction()
    except Exception:
        pass


def _construct_isolated_arms(
    arm_ids: Sequence[ArmId],
    names: Mapping[ArmId, str],
    commands_enabled: bool,
    factory: Callable[[ArmId, str, bool], ArmLike],
) -> dict[ArmId, ArmLike]:
    arms: dict[ArmId, ArmLike] = {}
    try:
        for arm_id in arm_ids:
            arms[arm_id] = factory(
                arm_id,
                names[arm_id],
                commands_enabled,
            )
    except BaseException:
        for arm in reversed(tuple(arms.values())):
            _abort_arm_construction_quietly(arm)
        raise
    return arms


class IsolatedSlaveSystem(_IsolatedBase):
    """Two-arm isolated subsystem for task/slave-only deployments."""

    def __init__(self, arms: Mapping[ArmId, ArmLike], **kwargs: Any) -> None:
        if set(arms) != set(SLAVE_ARM_IDS):
            raise TopologyConfigError("IsolatedSlaveSystem requires slave_left and slave_right")
        super().__init__(arms, **kwargs)
        self.left = IsolatedArmEndpoint(self, "left", mirror=False)
        self.right = IsolatedArmEndpoint(self, "right", mirror=False)

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        commands_enabled: bool = True,
        prefer_joint_ctrl: bool = False,
        name: str = "dual_piper",
        arm_factory: Callable[[ArmId, str, bool], ArmLike] | None = None,
        **kwargs: Any,
    ) -> "IsolatedSlaveSystem":
        robot_config = config.get("robot")
        if not isinstance(robot_config, Mapping):
            raise TopologyConfigError("config.robot must be a mapping")
        names = isolated_slave_can_names(robot_config)
        factory = arm_factory or _default_arm_factory
        arms = _construct_isolated_arms(
            SLAVE_ARM_IDS,
            names,
            commands_enabled,
            factory,
        )
        try:
            configured_kwargs = _isolated_config_kwargs(
                robot_config,
                kwargs,
            )
            return cls(
                arms,
                commands_enabled=commands_enabled,
                prefer_joint_ctrl=prefer_joint_ctrl,
                name=name,
                **configured_kwargs,
            )
        except BaseException:
            for arm in reversed(tuple(arms.values())):
                _abort_arm_construction_quietly(arm)
            raise


class IsolatedFourArmSystem(_IsolatedBase):
    """Four-CAN facade with slave observations and optional master mirroring."""

    def __init__(
        self,
        arms: Mapping[ArmId, ArmLike],
        *,
        mirror_commands: bool = True,
        **kwargs: Any,
    ) -> None:
        if set(arms) != set(ALL_ARM_IDS):
            missing = sorted(arm.value for arm in set(ALL_ARM_IDS) - set(arms))
            extra = sorted(str(arm) for arm in set(arms) - set(ALL_ARM_IDS))
            raise TopologyConfigError(f"four-arm system mismatch; missing={missing}, extra={extra}")
        can_names = [arm.can_name for arm in arms.values()]
        if len(can_names) != len(set(can_names)):
            raise TopologyConfigError("isolated four-arm system requires unique CAN interfaces")
        super().__init__(arms, **kwargs)
        self.mirror_commands = bool(mirror_commands)
        self.left = IsolatedArmEndpoint(self, "left", mirror=self.mirror_commands)
        self.right = IsolatedArmEndpoint(self, "right", mirror=self.mirror_commands)

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        commands_enabled: bool = True,
        mirror_commands: bool = True,
        prefer_joint_ctrl: bool = False,
        name: str = "dual_piper",
        arm_factory: Callable[[ArmId, str, bool], ArmLike] | None = None,
        **kwargs: Any,
    ) -> "IsolatedFourArmSystem":
        robot_config = config.get("robot")
        if not isinstance(robot_config, Mapping):
            raise TopologyConfigError("config.robot must be a mapping")
        names = isolated_can_names(robot_config)
        factory = arm_factory or _default_arm_factory
        arms = _construct_isolated_arms(
            ALL_ARM_IDS,
            names,
            commands_enabled,
            factory,
        )
        try:
            configured_kwargs = _isolated_config_kwargs(
                robot_config,
                kwargs,
            )
            return cls(
                arms,
                mirror_commands=mirror_commands,
                commands_enabled=commands_enabled,
                prefer_joint_ctrl=prefer_joint_ctrl,
                name=name,
                **configured_kwargs,
            )
        except BaseException:
            for arm in reversed(tuple(arms.values())):
                _abort_arm_construction_quietly(arm)
            raise

    @property
    def slave_left(self) -> ArmLike:
        return self.physical_arms[ArmId.SLAVE_LEFT]

    @property
    def slave_right(self) -> ArmLike:
        return self.physical_arms[ArmId.SLAVE_RIGHT]

    @property
    def master_left(self) -> ArmLike:
        return self.physical_arms[ArmId.MASTER_LEFT]

    @property
    def master_right(self) -> ArmLike:
        return self.physical_arms[ArmId.MASTER_RIGHT]

    def _prepare_rollout_internal(
        self,
        *,
        max_state_age_s: float,
        retries: int,
        sleep_s: float,
        speed_percent: int,
        gripper_effort: int | None,
        gripper_efforts: Mapping[ArmId | str, int] | None = None,
        command_slave_grippers: bool = True,
        preserve_master_roles: bool = False,
    ) -> tuple[
        tuple[BatchOperationResult | RoleTransactionResult, ...],
        PhysicalArmStates,
    ]:
        operations: list[BatchOperationResult | RoleTransactionResult] = []
        operations.append(
            self._configure_pair_internal(
                SLAVE_ARM_IDS,
                LinkageRole.MOTION_OUTPUT,
                speed_percent=speed_percent,
                gripper_effort=gripper_effort,
                gripper_efforts=gripper_efforts,
            )
        )
        prepared_arm_ids = SLAVE_ARM_IDS if preserve_master_roles else ALL_ARM_IDS
        if not preserve_master_roles:
            operations.append(
                self._configure_pair_internal(
                    MASTER_ARM_IDS,
                    LinkageRole.MOTION_OUTPUT,
                    speed_percent=speed_percent,
                    gripper_effort=gripper_effort,
                    gripper_efforts=gripper_efforts,
                )
            )
        states, _, _ = self._capture_role_aware_states(
            arm_ids=prepared_arm_ids,
            max_state_age_s=max_state_age_s,
            require_enabled=False,
            operation="capture FC arms before isolated rollout enable",
        )
        operations.append(
            self._command_captured_holds(
                states,
                prepared_arm_ids,
                speed_percent=speed_percent,
                gripper_effort=gripper_effort,
                gripper_efforts=gripper_efforts,
                command_slave_grippers=command_slave_grippers,
            )
        )
        operations.append(
            self._enable_arms(prepared_arm_ids, retries=retries, sleep_s=sleep_s)
        )
        final_states = (
            states
            if preserve_master_roles
            else self.require_healthy(max_state_age_s=max_state_age_s)
        )
        return tuple(operations), final_states

    def enable(self, *, retries: int = 5, sleep_s: float = 0.5) -> bool:
        """Safely establish four-arm ROLLOUT without an FA→FC jump."""

        max_state_age_s = self._resolved_motion_watchdog_age()
        try:
            with self._hardware_transaction():
                operations, _ = self._prepare_rollout_internal(
                    max_state_age_s=max_state_age_s,
                    retries=retries,
                    sleep_s=sleep_s,
                    speed_percent=50,
                    gripper_effort=None,
                )
            self.last_enable_operations = operations
            return True
        except BaseException as exc:
            if not self.faulted:
                self._latch_fault("enable isolated four-arm hardware", exc)
            raise


    def _move_all_to_slave_targets(
        self,
        states: PhysicalArmStates,
        *,
        speed_percent: int,
        gripper_effort: int | None,
        gripper_efforts: Mapping[ArmId | str, int] | None,
        transaction_generation: int,
    ) -> BatchOperationResult:
        targets = {
            arm_id: np.asarray(
                states[self._paired_slave_id(arm_id)].qpos,
                dtype=np.float64,
            ).copy()
            for arm_id in ALL_ARM_IDS
        }
        return self._interpolate_joint_targets(
            targets,
            operation="align four arms to slave holds",
            generation=transaction_generation,
            transaction_active=True,
            hz=30.0,
            step_sizes=DEFAULT_ARM_STEP_LENGTH,
            speed_percent=speed_percent,
            gripper_effort=gripper_effort,
            gripper_efforts=gripper_efforts,
            initial_states=states,
        )

    def _align_slaves_to_canonical_targets(
        self,
        target_commits: Mapping[ArmId, Any],
        *,
        speed_percent: int,
        gripper_effort: int | None,
        gripper_efforts: Mapping[ArmId | str, int] | None,
        transaction_generation: int,
    ) -> BatchOperationResult:
        states = self.require_healthy(
            max_state_age_s=self._resolved_motion_watchdog_age(),
            prefer_joint_ctrl_arm_ids=MASTER_ARM_IDS,
        )
        targets: dict[ArmId, np.ndarray] = {}
        for slave_id in SLAVE_ARM_IDS:
            commit = target_commits[slave_id]
            gripper_opening = float(states[slave_id].qpos[6])
            if commit.gripper_angle_um is not None:
                gripper_opening = sdk_gripper_to_opening(commit.gripper_angle_um)
            targets[slave_id] = np.concatenate(
                (
                    joints_feedback_to_rad(commit.joint_values_mdeg),
                    np.array([gripper_opening], dtype=np.float64),
                )
            )
        return self._interpolate_joint_targets(
            targets,
            operation="align slaves to canonical master targets",
            generation=transaction_generation,
            transaction_active=True,
            hz=30.0,
            step_sizes=DEFAULT_ARM_STEP_LENGTH,
            speed_percent=speed_percent,
            gripper_effort=gripper_effort,
            gripper_efforts=gripper_efforts,
            initial_states=states,
            prefer_joint_ctrl_arm_ids=MASTER_ARM_IDS,
        )

    def _start_gateway(
        self,
        gateway: GatewayLifecycle,
        *,
        automatic_dispatch: bool,
        speed_percent: int,
        gripper_effort: int | None,
        gripper_efforts: Mapping[ArmId | str, int] | None,
        seed_gripper_angles_um: Mapping[ArmId, int],
        transaction_generation: int,
        wait_for_initial_targets: bool = True,
    ) -> BatchOperationResult:
        gateway.begin_generation(
            seed_gripper_efforts=self.effective_slave_gripper_efforts,
            seed_gripper_angles_um=seed_gripper_angles_um,
        )
        if not wait_for_initial_targets:
            gateway.start(automatic_dispatch=automatic_dispatch)
            self._confirm_intervention_roles(
                "start gateway without initial targets"
            )
            return BatchOperationResult("start gateway", ())
        gateway.start(automatic_dispatch=False)
        while True:
            try:
                target_commits = gateway.wait_for_target_pair(
                    timeout_s=self._resolved_motion_watchdog_age(),
                )
                break
            except GatewayReadyTimeoutError:
                self.require_healthy(
                    max_state_age_s=self._resolved_motion_watchdog_age(),
                    prefer_joint_ctrl_arm_ids=MASTER_ARM_IDS,
                )
        self._confirm_intervention_roles(
            "confirm roles from canonical master targets"
        )
        aligned = self._align_slaves_to_canonical_targets(
            target_commits,
            speed_percent=speed_percent,
            gripper_effort=gripper_effort,
            gripper_efforts=gripper_efforts,
            transaction_generation=transaction_generation,
        )
        if automatic_dispatch:
            gateway.start(automatic_dispatch=True)
        gateway.wait_until_ready(
            timeout_s=self._resolved_motion_watchdog_age(),
        )
        return aligned

    def _activate_intervention_gateway(
        self,
        gateway: GatewayLifecycle,
        *,
        automatic_dispatch: bool,
        speed_percent: int,
        gripper_effort: int | None,
        gripper_efforts: Mapping[ArmId | str, int] | None,
        transaction_generation: int,
        wait_for_initial_targets: bool = True,
        preserve_gripper_positions: bool = True,
    ) -> tuple[RoleTransactionResult, BatchOperationResult, PhysicalArmStates]:
        seed_states = self.require_healthy(
            max_state_age_s=self._resolved_motion_watchdog_age(),
            prefer_joint_ctrl_arm_ids=MASTER_ARM_IDS,
        )
        seed_gripper_angles_um = (
            {
                slave_id: opening_to_sdk_gripper(seed_states[slave_id].qpos[6])
                for slave_id in SLAVE_ARM_IDS
            }
            if preserve_gripper_positions
            else {}
        )
        master_roles = self._configure_pair_internal(
            MASTER_ARM_IDS,
            LinkageRole.TEACHING_INPUT,
            speed_percent=speed_percent,
            gripper_effort=gripper_effort,
            gripper_efforts=gripper_efforts,
        )
        try:
            aligned = self._start_gateway(
                gateway,
                automatic_dispatch=automatic_dispatch,
                wait_for_initial_targets=wait_for_initial_targets,
                speed_percent=speed_percent,
                gripper_effort=gripper_effort,
                gripper_efforts=gripper_efforts,
                seed_gripper_angles_um=seed_gripper_angles_um,
                transaction_generation=transaction_generation,
            )
            states = self.require_healthy(
                max_state_age_s=self._resolved_motion_watchdog_age(),
                prefer_joint_ctrl_arm_ids=MASTER_ARM_IDS,
            )
            return master_roles, aligned, states
        except BaseException:
            gateway.stop()
            self._configure_pair_internal(
                MASTER_ARM_IDS,
                LinkageRole.MOTION_OUTPUT,
                speed_percent=speed_percent,
                gripper_effort=gripper_effort,
                gripper_efforts=gripper_efforts,
            )
            raise

    def initialize_static_teleop(
        self,
        gateway: GatewayLifecycle,
        *,
        speed_percent: int = 50,
        gripper_effort: int | None = None,
        gripper_efforts: Mapping[ArmId | str, int] | None = None,
    ) -> LifecycleResult:
        operations: list[BatchOperationResult | RoleTransactionResult] = []
        max_state_age_s = self._resolved_motion_watchdog_age()
        try:
            with self._hardware_transaction() as generation:
                operations.append(self.connect(read_only=True))
                master_observations = self.role_controller.observe_confirmed_many(
                    MASTER_ARM_IDS
                )
                preserve_master_roles = all(
                    observation.role is LinkageRole.TEACHING_INPUT
                    for observation in master_observations.values()
                )
                rollout_operations, _ = self._prepare_rollout_internal(
                    max_state_age_s=max_state_age_s,
                    retries=5,
                    sleep_s=0.5,
                    speed_percent=speed_percent,
                    gripper_effort=gripper_effort,
                    gripper_efforts=gripper_efforts,
                    command_slave_grippers=False,
                    preserve_master_roles=preserve_master_roles,
                )
                operations.extend(rollout_operations)
                if not preserve_master_roles:
                    operations.append(
                        self._align_targets_to_sources(
                            target_arm_ids=SLAVE_ARM_IDS,
                            source_arm_ids=MASTER_ARM_IDS,
                            operation="align slaves to FC master positions",
                            speed_percent=speed_percent,
                            gripper_effort=gripper_effort,
                            gripper_efforts=gripper_efforts,
                            transaction_generation=generation,
                            preserve_target_gripper=True,
                            command_slave_grippers=False,
                        )
                    )
                master_roles, aligned, final_states = self._activate_intervention_gateway(
                    gateway,
                    automatic_dispatch=True,
                    wait_for_initial_targets=False,
                    preserve_gripper_positions=not preserve_master_roles,
                    speed_percent=speed_percent,
                    gripper_effort=gripper_effort,
                    gripper_efforts=gripper_efforts,
                    transaction_generation=generation,
                )
                operations.extend((master_roles, aligned))
            return LifecycleResult(StaticMode.TELEOP, generation, tuple(operations), final_states)
        except BaseException as exc:
            if not self.faulted:
                self._latch_fault("initialize static teleop", exc)
            raise

    def enter_intervention(
        self,
        gateway: GatewayLifecycle,
        *,
        speed_percent: int = 50,
        gripper_effort: int | None = None,
        gripper_efforts: Mapping[ArmId | str, int] | None = None,
    ) -> LifecycleResult:
        operations: list[BatchOperationResult | RoleTransactionResult] = []
        max_state_age_s = self._resolved_motion_watchdog_age()
        try:
            with self._hardware_transaction() as generation:
                operations.append(
                    self._configure_pair_internal(
                        SLAVE_ARM_IDS,
                        LinkageRole.MOTION_OUTPUT,
                        speed_percent=speed_percent,
                        gripper_effort=gripper_effort,
                        gripper_efforts=gripper_efforts,
                    )
                )
                operations.append(
                    self._configure_pair_internal(
                        MASTER_ARM_IDS,
                        LinkageRole.MOTION_OUTPUT,
                        speed_percent=speed_percent,
                        gripper_effort=gripper_effort,
                        gripper_efforts=gripper_efforts,
                    )
                )
                before = self.require_healthy(max_state_age_s=max_state_age_s)
                operations.append(
                    self._command_captured_holds(
                        before,
                        ALL_ARM_IDS,
                        speed_percent=speed_percent,
                        gripper_effort=gripper_effort,
                        gripper_efforts=gripper_efforts,
                    )
                )
                operations.append(
                    self._align_targets_to_sources(
                        target_arm_ids=MASTER_ARM_IDS,
                        source_arm_ids=SLAVE_ARM_IDS,
                        operation="align masters to slaves",
                        speed_percent=speed_percent,
                        gripper_effort=gripper_effort,
                        gripper_efforts=gripper_efforts,
                        transaction_generation=generation,
                    )
                )
                master_roles, aligned, after = self._activate_intervention_gateway(
                    gateway,
                    automatic_dispatch=True,
                    speed_percent=speed_percent,
                    gripper_effort=gripper_effort,
                    gripper_efforts=gripper_efforts,
                    transaction_generation=generation,
                )
                operations.extend((master_roles, aligned))
            return LifecycleResult("intervention", generation, tuple(operations), after)
        except BaseException as exc:
            if not self.faulted:
                self._latch_fault("enter intervention", exc)
            raise

    def _require_established_intervention_roles(
        self,
        *,
        operation: str,
        observations: Mapping[ArmId, RoleObservation] | None = None,
    ) -> dict[ArmId, RoleObservation]:
        if observations is None:
            observations = self.role_controller.observe_many(ALL_ARM_IDS)
        expected_master_roles = {
            ArmId.MASTER_LEFT: LinkageRole.TEACHING_INPUT,
            ArmId.MASTER_RIGHT: LinkageRole.TEACHING_INPUT,
        }
        for arm_id, expected_role in expected_master_roles.items():
            observation = observations[arm_id]
            observed_role = bool(
                observation.fresh and observation.role is expected_role
            )
            confirmed_role = (
                self.role_controller.last_confirmed_role(arm_id)
                is expected_role
            )
            if observed_role or confirmed_role:
                continue
            raise ArmOperationError(
                arm_id,
                operation,
                f"expected established {expected_role.short_name} role, observed "
                f"{getattr(observation.role, 'short_name', None)!r}: "
                f"{observation.evidence}",
            )
        for arm_id in SLAVE_ARM_IDS:
            observation = observations[arm_id]
            observed_fc = bool(
                observation.fresh
                and observation.role is LinkageRole.MOTION_OUTPUT
            )
            confirmed_fc = (
                self.role_controller.last_confirmed_role(arm_id)
                is LinkageRole.MOTION_OUTPUT
            )
            if observed_fc or confirmed_fc:
                continue
            raise ArmOperationError(
                arm_id,
                operation,
                "slave has no last-confirmed FC role: "
                f"{observation.evidence}",
            )
        return observations

    def _confirm_intervention_roles(self, operation: str) -> None:
        role_observations = self.role_controller.observe_many(ALL_ARM_IDS)
        for master_id in MASTER_ARM_IDS:
            self.role_controller.confirm_pending(
                master_id,
                LinkageRole.TEACHING_INPUT,
            )
        self._require_established_intervention_roles(
            operation=operation,
            observations=role_observations,
        )

    def pause_intervention(
        self,
        gateway: GatewayLifecycle,
        *,
        gripper_efforts: Mapping[ArmId | str, int] | None = None,
    ) -> LifecycleResult:
        """Pause host relay while preserving FA masters and holding FC slaves."""

        operations: list[BatchOperationResult | RoleTransactionResult] = []
        max_state_age_s = self._resolved_motion_watchdog_age()
        try:
            with self._hardware_transaction() as generation:
                gateway.stop_at_cycle_boundary(timeout_s=max_state_age_s)
                self._require_established_intervention_roles(
                    operation="pause intervention",
                )
                states = self.require_healthy(
                    max_state_age_s=max_state_age_s,
                    prefer_joint_ctrl_arm_ids=MASTER_ARM_IDS,
                )
                operations.append(
                    self._command_captured_holds(
                        states,
                        SLAVE_ARM_IDS,
                        speed_percent=50,
                        gripper_effort=None,
                        gripper_efforts=gripper_efforts,
                    )
                )
            return LifecycleResult(
                "intervention_paused",
                generation,
                tuple(operations),
                states,
            )
        except BaseException as exc:
            gateway.stop()
            if not self.faulted:
                arm_id = exc.arm_id if isinstance(exc, ArmOperationError) else None
                self._latch_fault("pause intervention", exc, arm_id=arm_id)
            raise

    def resume_intervention(
        self,
        gateway: GatewayLifecycle,
        *,
        speed_percent: int = 50,
        gripper_efforts: Mapping[ArmId | str, int] | None = None,
    ) -> LifecycleResult:
        """Resume a paused relay without changing any FA/FC role or gain."""

        operations: list[BatchOperationResult | RoleTransactionResult] = []
        max_state_age_s = self._resolved_motion_watchdog_age()
        try:
            with self._hardware_transaction() as generation:
                self._require_established_intervention_roles(
                    operation="resume intervention",
                )
                before = self.require_healthy(
                    max_state_age_s=max_state_age_s,
                    prefer_joint_ctrl_arm_ids=MASTER_ARM_IDS,
                )
                operations.append(
                    self._command_captured_holds(
                        before,
                        SLAVE_ARM_IDS,
                        speed_percent=speed_percent,
                        gripper_effort=None,
                        gripper_efforts=gripper_efforts,
                    )
                )
                operations.append(
                    self._start_gateway(
                        gateway,
                        automatic_dispatch=True,
                        speed_percent=speed_percent,
                        gripper_effort=None,
                        gripper_efforts=gripper_efforts,
                        seed_gripper_angles_um={
                            slave_id: opening_to_sdk_gripper(before[slave_id].qpos[6])
                            for slave_id in SLAVE_ARM_IDS
                        },
                        transaction_generation=generation,
                    )
                )
                after = self.require_healthy(
                    max_state_age_s=max_state_age_s,
                    prefer_joint_ctrl_arm_ids=MASTER_ARM_IDS,
                )
            return LifecycleResult(
                "intervention",
                generation,
                tuple(operations),
                after,
            )
        except BaseException as exc:
            gateway.stop()
            if not self.faulted:
                arm_id = exc.arm_id if isinstance(exc, ArmOperationError) else None
                self._latch_fault("resume intervention", exc, arm_id=arm_id)
            raise

    def exit_intervention(
        self,
        gateway: GatewayLifecycle,
        *,
        speed_percent: int = 50,
        gripper_effort: int | None = None,
        gripper_efforts: Mapping[ArmId | str, int] | None = None,
    ) -> LifecycleResult:
        operations: list[BatchOperationResult | RoleTransactionResult] = []
        max_state_age_s = self._resolved_motion_watchdog_age()
        try:
            with self._hardware_transaction() as generation:
                gateway.stop_at_cycle_boundary(timeout_s=max_state_age_s)
                operations.append(
                    self._configure_pair_internal(
                        MASTER_ARM_IDS,
                        LinkageRole.MOTION_OUTPUT,
                        speed_percent=speed_percent,
                        gripper_effort=gripper_effort,
                        gripper_efforts=gripper_efforts,
                    )
                )
                motion_output_states = self.require_healthy(
                    max_state_age_s=max_state_age_s,
                )
                operations.append(
                    self._move_all_to_slave_targets(
                        motion_output_states,
                        speed_percent=speed_percent,
                        gripper_effort=gripper_effort,
                        gripper_efforts=gripper_efforts,
                        transaction_generation=generation,
                    )
                )
                after = self.require_healthy(max_state_age_s=max_state_age_s)
            return LifecycleResult("rollout", generation, tuple(operations), after)
        except BaseException as exc:
            gateway.stop()
            if not self.faulted:
                self._latch_fault("exit intervention", exc)
            raise


def _default_arm_factory(arm_id: ArmId, can_name: str, commands_enabled: bool) -> SinglePiperArm:
    return SinglePiperArm(
        name=arm_id.value,
        can_name=can_name,
        commands_enabled=commands_enabled,
        calculate_fk_per_frame=False,
    )


__all__ = [
    "ArmLike",
    "GatewayLifecycle",
    "IsolatedArmEndpoint",
    "IsolatedFourArmSystem",
    "IsolatedSlaveSystem",
    "LinkageRoleController",
    "RoleObserver",
    "SdkTrafficRoleObserver",
]
