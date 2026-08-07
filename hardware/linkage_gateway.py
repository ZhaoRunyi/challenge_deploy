from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import struct
import threading
import time
from typing import Any, Callable, Mapping, Sequence

import can

from .topology import ArmId, PiperTopologyError


MOTION_CTRL_ID = 0x151
JOINT_CTRL_IDS = (0x155, 0x156, 0x157)
GRIPPER_CTRL_ID = 0x159
LINKAGE_INPUT_IDS = frozenset((MOTION_CTRL_ID, *JOINT_CTRL_IDS, GRIPPER_CTRL_ID))
DEFAULT_FRAME_FRESHNESS_S = 0.050
DEFAULT_JOINT_FRAME_SKEW_S = 0.010


class LinkageGatewayError(PiperTopologyError):
    pass


class GatewayTransmitError(LinkageGatewayError):
    def __init__(self, arm_id: ArmId, can_id: int, cause: BaseException) -> None:
        self.arm_id = arm_id
        self.can_id = can_id
        self.cause = cause
        super().__init__(f"gateway transmit 0x{can_id:03X} for {arm_id.value} failed: {cause}")


class GatewayFaultLatchedError(LinkageGatewayError):
    def __init__(self, fault: BaseException) -> None:
        self.fault = fault
        super().__init__(f"linkage gateway fault is latched: {fault}")


class GatewayBoundaryTimeoutError(LinkageGatewayError):
    pass


class GatewayReadyTimeoutError(LinkageGatewayError):
    pass


class GatewayDispatchTimeoutError(LinkageGatewayError):
    pass


class GatewayDispatchInterruptedError(LinkageGatewayError):
    pass


@dataclass(frozen=True, slots=True)
class GatewayConfig:
    freshness_s: float = DEFAULT_FRAME_FRESHNESS_S
    max_skew_s: float = DEFAULT_JOINT_FRAME_SKEW_S
    gripper_effort: int = 1000
    speed_percent: int = 100
    recv_timeout_s: float = 0.020
    dispatch_timeout_s: float = 0.100

    def __post_init__(self) -> None:
        if self.freshness_s <= 0.0:
            raise ValueError("freshness_s must be positive")
        if self.max_skew_s < 0.0 or self.max_skew_s > self.freshness_s:
            raise ValueError("max_skew_s must be in [0, freshness_s]")
        if not 0 <= self.gripper_effort <= 0xFFFF:
            raise ValueError("gripper_effort must fit the uint16 CAN field")
        if not 0 <= self.speed_percent <= 100:
            raise ValueError("speed_percent must be in [0, 100]")
        if self.recv_timeout_s <= 0.0:
            raise ValueError("recv_timeout_s must be positive")
        if self.dispatch_timeout_s <= 0.0:
            raise ValueError("dispatch_timeout_s must be positive")


@dataclass(frozen=True, slots=True)
class CanonicalTargetCommit:
    generation: int
    frame_group_seq: int
    committed_at_s: float
    joint_values_mdeg: tuple[int, ...]
    gripper_angle_um: int | None
    motion_ctrl_payload: bytes
    joint_payloads: tuple[bytes, bytes, bytes]
    gripper_payload: bytes | None

    def to_can_messages(self) -> tuple[can.Message, ...]:
        messages: tuple[can.Message, ...] = (
            can.Message(
                arbitration_id=MOTION_CTRL_ID,
                data=self.motion_ctrl_payload,
                dlc=8,
                is_extended_id=False,
            ),
            *(
                can.Message(
                    arbitration_id=can_id,
                    data=payload,
                    dlc=8,
                    is_extended_id=False,
                )
                for can_id, payload in zip(JOINT_CTRL_IDS, self.joint_payloads)
            ),
        )
        if self.gripper_payload is not None:
            messages += (
                can.Message(
                    arbitration_id=GRIPPER_CTRL_ID,
                    data=self.gripper_payload,
                    dlc=8,
                    is_extended_id=False,
                ),
            )
        return messages


@dataclass(frozen=True, slots=True)
class PairedDispatchReceipt:
    """Proof that one immutable left/right target pair was fully submitted."""

    generation: int
    dispatch_seq: int
    completed_at_s: float
    left_commit: CanonicalTargetCommit
    right_commit: CanonicalTargetCommit


@dataclass(slots=True)
class GatewayCounters:
    received: int = 0
    ignored: int = 0
    rejected: int = 0
    stale_generation: int = 0
    joint_cycles: int = 0
    gripper_commands: int = 0
    transmitted_frames: int = 0
    drained_frames: int = 0
    source_backlog_drops: int = 0


@dataclass(frozen=True, slots=True)
class _BufferedFrame:
    message: can.Message
    received_at_s: float


class SemanticLinkageGateway:
    """Decode an isolated FA stream into immutable canonical candidates.

    Joint payloads are forwarded only as a complete 0x155/156/157 cycle.
    0x151 and 0x159 are always synthesized, so installation, enable/clear,
    gripper-zero, and effort side effects are never copied from the master.

    This class never transmits to the slave bus.  The bimanual supervisor is
    the only owner allowed to submit a validated left/right candidate pair.
    """

    def __init__(
        self,
        *,
        arm_id: ArmId,
        master_bus: can.BusABC,
        slave_bus: can.BusABC,
        config: GatewayConfig | None = None,
        clock: Callable[[], float] = time.monotonic,
        owns_buses: bool = False,
    ) -> None:
        if arm_id not in (ArmId.SLAVE_LEFT, ArmId.SLAVE_RIGHT):
            raise ValueError("gateway arm_id must identify a physical slave arm")
        self.arm_id = arm_id
        self.master_bus = master_bus
        self.slave_bus = slave_bus
        self.config = config or GatewayConfig()
        self.clock = clock
        self.owns_buses = owns_buses
        self.counters = GatewayCounters()
        self._joint_frames: dict[int, _BufferedFrame] = {}
        self._next_joint_id = JOINT_CTRL_IDS[0]
        self._last_joint_values: tuple[int, ...] | None = None
        self._gripper_target_seed_um: int | None = None
        self._source_gripper_baseline_um: int | None = None
        self._canonical_gripper_angle_um: int | None = None
        self._generation_gripper_effort = self.config.gripper_effort
        self._last_gripper_payload: bytes | None = None
        self._latest_joint_payloads: tuple[bytes, bytes, bytes] | None = None
        self._last_commit: CanonicalTargetCommit | None = None
        self._frame_group_seq = 0
        self._pending_family_ids: set[int] = set()
        self._validated_family_seen: dict[int, float] = {}
        self._generation = 0
        self._active = False
        self._draining = False
        self._fault: BaseException | None = None
        self._candidate_callback: Callable[["SemanticLinkageGateway"], None] | None = None
        self._fault_callback: (
            Callable[["SemanticLinkageGateway", BaseException], None] | None
        ) = None
        self._closed = False
        self._lock = threading.RLock()
        self._boundary_condition = threading.Condition(self._lock)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    @classmethod
    def from_socketcan(
        cls,
        *,
        arm_id: ArmId,
        master_can_name: str,
        slave_can_name: str,
        config: GatewayConfig | None = None,
    ) -> "SemanticLinkageGateway":
        master_bus = can.Bus(
            interface="socketcan",
            channel=master_can_name,
            receive_own_messages=False,
            can_filters=[
                {"can_id": can_id, "can_mask": 0x7FF, "extended": False}
                for can_id in LINKAGE_INPUT_IDS
            ],
        )
        try:
            slave_bus = can.Bus(
                interface="socketcan",
                channel=slave_can_name,
                receive_own_messages=False,
            )
        except BaseException:
            master_bus.shutdown()
            raise
        return cls(
            arm_id=arm_id,
            master_bus=master_bus,
            slave_bus=slave_bus,
            config=config,
            owns_buses=True,
        )

    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation

    @property
    def fault(self) -> BaseException | None:
        with self._lock:
            return self._fault

    def _require_healthy_locked(self) -> None:
        if self._fault is not None:
            raise GatewayFaultLatchedError(self._fault)

    def _reset_decode_state_locked(self) -> None:
        self._joint_frames.clear()
        self._next_joint_id = JOINT_CTRL_IDS[0]
        self._last_joint_values = None
        self._validated_family_seen.clear()

    def _set_owner_callbacks(
        self,
        *,
        candidate_callback: Callable[["SemanticLinkageGateway"], None],
        fault_callback: Callable[["SemanticLinkageGateway", BaseException], None],
    ) -> None:
        with self._lock:
            if self._candidate_callback is not None or self._fault_callback is not None:
                raise LinkageGatewayError(f"{self.arm_id.value} gateway already has an owner")
            self._candidate_callback = candidate_callback
            self._fault_callback = fault_callback

    def _fence_from_owner(self, *, generation: int, fault: BaseException | None) -> None:
        with self._lock:
            self._active = False
            self._draining = False
            self._generation = generation
            if fault is not None:
                self._fault = fault
            self._reset_decode_state_locked()
            self._stop_event.set()
            self._boundary_condition.notify_all()

    def _drain_master_rx_locked(self) -> None:
        while True:
            message = self.master_bus.recv(timeout=0.0)
            if message is None:
                return
            self.counters.drained_frames += 1

    def begin_generation(
        self,
        *,
        seed_gripper_efforts: Mapping[ArmId, int | None] | int | None,
        seed_gripper_angles_um: Mapping[ArmId, int | None] | int | None = None,
    ) -> int:
        if isinstance(seed_gripper_efforts, Mapping):
            requested_effort = seed_gripper_efforts.get(self.arm_id)
        else:
            requested_effort = seed_gripper_efforts
        generation_effort = (
            self.config.gripper_effort
            if requested_effort is None
            else int(requested_effort)
        )
        if isinstance(seed_gripper_angles_um, Mapping):
            requested_angle = seed_gripper_angles_um.get(self.arm_id)
        else:
            requested_angle = seed_gripper_angles_um
        if not 0 <= generation_effort <= 0xFFFF:
            raise ValueError("seed gripper effort must fit the uint16 CAN field")
        with self._lock:
            self._require_healthy_locked()
            if self._thread is not None and self._thread.is_alive():
                raise LinkageGatewayError("stop the gateway before beginning a new generation")
            self._drain_master_rx_locked()
            self._generation += 1
            self._reset_decode_state_locked()
            self._gripper_target_seed_um = (
                None if requested_angle is None else int(requested_angle)
            )
            self._source_gripper_baseline_um = None
            self._canonical_gripper_angle_um = self._gripper_target_seed_um
            self._generation_gripper_effort = generation_effort
            self._last_gripper_payload = None
            self._latest_joint_payloads = None
            self._last_commit = None
            self._frame_group_seq = 0
            self._pending_family_ids.clear()
            self._draining = False
            self._active = True
            self._boundary_condition.notify_all()
            return self._generation

    def start(self) -> int:
        with self._lock:
            self._require_healthy_locked()
            if not self._active:
                raise LinkageGatewayError("begin_generation before start")
            generation = self._generation
            if self._thread is not None and self._thread.is_alive():
                return generation
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run,
                name=f"piper_linkage_{self.arm_id.side}",
                daemon=True,
            )
            self._thread.start()
            return generation

    def stop(self) -> None:
        with self._lock:
            was_running = self._active or (
                self._thread is not None and self._thread.is_alive()
            )
            self._active = False
            self._draining = False
            if was_running:
                self._generation += 1
            self._reset_decode_state_locked()
            self._stop_event.set()
            self._boundary_condition.notify_all()
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(0.1, self.config.recv_timeout_s * 3.0))
        with self._lock:
            if self._thread is thread and (thread is None or not thread.is_alive()):
                self._thread = None

    def stop_at_cycle_boundary(self, *, timeout_s: float = 0.100) -> bool:
        """Drain at most the already-started joint triplet, then fence decode.

        Once draining begins, no new 0x155 cycle may start.  If an incomplete
        triplet cannot finish before the deadline, the gateway is fenced and a
        typed timeout is raised; callers must not perform a role swap.
        """

        if timeout_s <= 0.0:
            raise ValueError("timeout_s must be positive")
        deadline = time.monotonic() + timeout_s
        timed_out = False
        with self._boundary_condition:
            self._draining = True
            while self._active and self._joint_frames:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    timed_out = True
                    break
                self._boundary_condition.wait(timeout=remaining)
            self._active = False
            self._draining = False
            self._generation += 1
            self._reset_decode_state_locked()
            self._stop_event.set()
            self._boundary_condition.notify_all()
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
        with self._lock:
            if self._thread is thread and (thread is None or not thread.is_alive()):
                self._thread = None
            elif thread is not None and thread.is_alive():
                timed_out = True
        if timed_out:
            raise GatewayBoundaryTimeoutError(
                f"{self.arm_id.value} semantic decoder did not stop at a complete "
                f"frame boundary within {timeout_s:.3f}s"
            )
        return True

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self.stop()
        if self.owns_buses:
            self.master_bus.shutdown()
            self.slave_bus.shutdown()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            with self._lock:
                if not self._active:
                    break
                generation = self._generation
            try:
                message = self.master_bus.recv(timeout=self.config.recv_timeout_s)
            except BaseException as exc:
                with self._lock:
                    self._fault = exc
                    self._active = False
                    self._generation += 1
                    self._joint_frames.clear()
                    self._next_joint_id = JOINT_CTRL_IDS[0]
                    self._boundary_condition.notify_all()
                    fault_callback = self._fault_callback
                if fault_callback is not None:
                    fault_callback(self, exc)
                break
            if message is not None:
                source_timestamp_s, source_age_s = self._source_timestamp_and_age(
                    message
                )
                if source_age_s > self.config.freshness_s:
                    with self._lock:
                        self.counters.rejected += 1
                        self.counters.source_backlog_drops += 1
                    continue
                try:
                    self.process_message(
                        message,
                        generation=generation,
                        received_at_s=source_timestamp_s,
                    )
                except GatewayFaultLatchedError:
                    break

    def _source_timestamp_and_age(self, message: can.Message) -> tuple[float, float]:
        source_timestamp_s = float(getattr(message, "timestamp", 0.0))
        if source_timestamp_s <= 0.0:
            return self.clock(), 0.0
        return source_timestamp_s, max(0.0, time.time() - source_timestamp_s)

    @staticmethod
    def _valid_classic_frame(message: can.Message) -> bool:
        return (
            bool(getattr(message, "is_rx", True))
            and not message.is_extended_id
            and not message.is_remote_frame
            and not message.is_error_frame
            and not bool(getattr(message, "is_fd", False))
            and int(message.dlc) == 8
            and len(message.data) == 8
        )

    @staticmethod
    def _decode_joint_frame(message: can.Message) -> tuple[int, int]:
        return struct.unpack(">ii", bytes(message.data))

    def _decode_joint_cycle(self) -> tuple[int, ...]:
        values: list[int] = []
        for can_id in JOINT_CTRL_IDS:
            values.extend(self._decode_joint_frame(self._joint_frames[can_id].message))
        return tuple(values)

    def _clear_joint_cycle_locked(self) -> None:
        self._joint_frames.clear()
        self._next_joint_id = JOINT_CTRL_IDS[0]
        self._boundary_condition.notify_all()

    def _family_ready_locked(self, now_s: float) -> bool:
        required = JOINT_CTRL_IDS
        if any(can_id not in self._validated_family_seen for can_id in required):
            return False
        return all(
            0.0 <= now_s - self._validated_family_seen[can_id] <= self.config.freshness_s
            for can_id in required
        )

    def _publish_complete_candidate_locked(self, source_now_s: float) -> bool:
        has_joint_cycle = set(JOINT_CTRL_IDS).issubset(self._pending_family_ids)
        has_gripper_update = (
            GRIPPER_CTRL_ID in self._pending_family_ids
            and self._last_gripper_payload is not None
        )
        if (
            self._latest_joint_payloads is None
            or self._last_joint_values is None
            or not self._family_ready_locked(source_now_s)
            or not (has_joint_cycle or has_gripper_update)
        ):
            self._boundary_condition.notify_all()
            return False
        committed_at_s = self.clock()
        self._frame_group_seq += 1
        self._last_commit = CanonicalTargetCommit(
            generation=self._generation,
            frame_group_seq=self._frame_group_seq,
            committed_at_s=committed_at_s,
            joint_values_mdeg=self._last_joint_values,
            gripper_angle_um=self._canonical_gripper_angle_um,
            motion_ctrl_payload=bytes(self._motion_ctrl_message().data),
            joint_payloads=self._latest_joint_payloads,
            gripper_payload=self._last_gripper_payload,
        )
        self._pending_family_ids.clear()
        self._boundary_condition.notify_all()
        return True

    def _motion_ctrl_message(self) -> can.Message:
        data = bytes(
            (
                0x01,  # CAN control
                0x01,  # MOVE J
                self.config.speed_percent,
                0xAD,  # high-follow mode
                0x00,  # no offline residence
                0x00,  # installation direction unchanged
                0x00,
                0x00,
            )
        )
        return can.Message(arbitration_id=MOTION_CTRL_ID, data=data, dlc=8, is_extended_id=False)

    def _process_joint_locked(
        self,
        message: can.Message,
        received_at_s: float,
    ) -> None:
        if self._draining and not self._joint_frames:
            self.counters.rejected += 1
            return
        can_id = int(message.arbitration_id)
        if can_id != self._next_joint_id:
            self._clear_joint_cycle_locked()
            self.counters.rejected += 1
            return
        self._joint_frames[can_id] = _BufferedFrame(
            message=message,
            received_at_s=received_at_s,
        )
        if can_id != JOINT_CTRL_IDS[-1]:
            self._next_joint_id = JOINT_CTRL_IDS[JOINT_CTRL_IDS.index(can_id) + 1]
        if any(can_id not in self._joint_frames for can_id in JOINT_CTRL_IDS):
            return

        frames = tuple(self._joint_frames[can_id] for can_id in JOINT_CTRL_IDS)
        timestamps = tuple(frame.received_at_s for frame in frames)
        if received_at_s - min(timestamps) > self.config.freshness_s:
            self._clear_joint_cycle_locked()
            self.counters.rejected += 1
            return
        if max(timestamps) - min(timestamps) > self.config.max_skew_s:
            self._clear_joint_cycle_locked()
            self.counters.rejected += 1
            return

        values = self._decode_joint_cycle()
        joint_payloads = tuple(
            bytes(self._joint_frames[can_id].message.data)
            for can_id in JOINT_CTRL_IDS
        )
        self._clear_joint_cycle_locked()
        self._last_joint_values = values
        self._latest_joint_payloads = (
            joint_payloads[0],
            joint_payloads[1],
            joint_payloads[2],
        )
        for can_id, frame in zip(JOINT_CTRL_IDS, frames):
            self._validated_family_seen[can_id] = frame.received_at_s
            self._pending_family_ids.add(can_id)
        self._publish_complete_candidate_locked(received_at_s)
        self.counters.joint_cycles += 1

    def _process_gripper_locked(
        self,
        message: can.Message,
        received_at_s: float,
    ) -> None:
        angle_um = struct.unpack(">i", bytes(message.data[:4]))[0]
        if self._gripper_target_seed_um is not None:
            if self._source_gripper_baseline_um is None:
                self._source_gripper_baseline_um = angle_um
                self.counters.gripper_commands += 1
                return
            angle_um = (
                self._gripper_target_seed_um
                + angle_um
                - self._source_gripper_baseline_um
            )
        # Effort is frozen to config; status is normal enable; zeroing is always stripped.
        payload = struct.pack(">iHBB", angle_um, self._generation_gripper_effort, 0x01, 0x00)
        self._canonical_gripper_angle_um = angle_um
        self._last_gripper_payload = payload
        self._validated_family_seen[GRIPPER_CTRL_ID] = received_at_s
        self._pending_family_ids.add(GRIPPER_CTRL_ID)
        self._publish_complete_candidate_locked(received_at_s)
        self.counters.gripper_commands += 1

    def process_message(
        self,
        message: can.Message,
        *,
        generation: int | None = None,
        received_at_s: float | None = None,
    ) -> None:
        now_s = self.clock() if received_at_s is None else float(received_at_s)
        candidate_callback: Callable[[SemanticLinkageGateway], None] | None = None
        with self._lock:
            self._require_healthy_locked()
            previous_frame_group_seq = self._frame_group_seq
            expected_generation = self._generation if generation is None else int(generation)
            self.counters.received += 1
            if not self._active or expected_generation != self._generation:
                self.counters.stale_generation += 1
            else:
                can_id = int(message.arbitration_id)
                self._process_valid_generation_message_locked(
                    message,
                    can_id=can_id,
                    received_at_s=now_s,
                )
            if self._frame_group_seq != previous_frame_group_seq:
                candidate_callback = self._candidate_callback
        if candidate_callback is not None:
            candidate_callback(self)

    def _process_valid_generation_message_locked(
        self,
        message: can.Message,
        *,
        can_id: int,
        received_at_s: float,
    ) -> None:
        if can_id not in LINKAGE_INPUT_IDS:
            self.counters.ignored += 1
            return
        if not self._valid_classic_frame(message):
            self.counters.rejected += 1
            return
        if self._draining and can_id not in JOINT_CTRL_IDS:
            self.counters.rejected += 1
            return
        if can_id == MOTION_CTRL_ID:
            # Input 0x151 is intentionally never copied; a canonical one is
            # synthesized when a complete candidate is published.
            self._validated_family_seen[MOTION_CTRL_ID] = received_at_s
            self._pending_family_ids.add(MOTION_CTRL_ID)
            self._publish_complete_candidate_locked(received_at_s)
            return
        if can_id in JOINT_CTRL_IDS:
            self._process_joint_locked(message, received_at_s)
            return
        self._process_gripper_locked(message, received_at_s)


class BimanualLinkageGateway:
    def __init__(self, left: SemanticLinkageGateway, right: SemanticLinkageGateway) -> None:
        if left.arm_id is not ArmId.SLAVE_LEFT or right.arm_id is not ArmId.SLAVE_RIGHT:
            raise ValueError("bimanual gateway sides are reversed")
        self.left = left
        self.right = right
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._dispatch_lock = threading.Lock()
        self._generation = max(left.generation, right.generation)
        self._active = False
        self._fault: BaseException | None = None
        self._automatic_dispatch = False
        self._dispatch_seq = 0
        self._last_dispatched_sequences = {
            ArmId.SLAVE_LEFT: 0,
            ArmId.SLAVE_RIGHT: 0,
        }
        self._last_dispatch_receipt: PairedDispatchReceipt | None = None
        self._transition_dispatch_receipt: PairedDispatchReceipt | None = None
        self._supervisor_stop_event = threading.Event()
        self._supervisor_thread: threading.Thread | None = None
        self._closed = False
        self.left._set_owner_callbacks(
            candidate_callback=self._on_candidate_available,
            fault_callback=self._on_semantic_fault,
        )
        self.right._set_owner_callbacks(
            candidate_callback=self._on_candidate_available,
            fault_callback=self._on_semantic_fault,
        )

    def _synchronize_generation_counters(self) -> None:
        with self.left._lock:
            with self.right._lock:
                common = max(
                    self._generation,
                    self.left._generation,
                    self.right._generation,
                )
                self.left._generation = common
                self.right._generation = common
                self._generation = common

    def _require_healthy_locked(self) -> None:
        if self._fault is not None:
            raise GatewayFaultLatchedError(self._fault)

    def _on_candidate_available(self, _gateway: SemanticLinkageGateway) -> None:
        with self._condition:
            self._condition.notify_all()

    def _on_semantic_fault(
        self,
        gateway: SemanticLinkageGateway,
        fault: BaseException,
    ) -> None:
        wrapped = LinkageGatewayError(
            f"{gateway.arm_id.value} gateway receive failed: {fault}"
        )
        self._latch_global_fault(wrapped)

    def _latch_global_fault(self, fault: BaseException) -> None:
        with self._condition:
            if self._fault is not None:
                return
            self._fault = fault
            self._active = False
            self._generation = max(
                self._generation,
                self.left.generation,
                self.right.generation,
            ) + 1
            fault_generation = self._generation
            self._supervisor_stop_event.set()
            self._condition.notify_all()
        self.left._fence_from_owner(generation=fault_generation, fault=fault)
        self.right._fence_from_owner(generation=fault_generation, fault=fault)

    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation

    @property
    def fault(self) -> BaseException | None:
        with self._lock:
            return self._fault

    @property
    def transition_dispatch_receipt(self) -> PairedDispatchReceipt | None:
        """Return the immutable pair submitted at the current transition."""

        with self._condition:
            return self._transition_dispatch_receipt

    @property
    def last_dispatch_receipt(self) -> PairedDispatchReceipt | None:
        """Return the most recent pair fully submitted by the sole send owner."""

        with self._condition:
            return self._last_dispatch_receipt

    def begin_generation(
        self,
        *,
        seed_gripper_efforts: Mapping[ArmId, int | None],
        seed_gripper_angles_um: Mapping[ArmId, int | None] | None = None,
    ) -> int:
        self._synchronize_generation_counters()
        with self._condition:
            self._require_healthy_locked()
            if self._closed:
                raise LinkageGatewayError("bimanual gateway is closed")
            if self._active:
                raise LinkageGatewayError("stop the bimanual gateway before a new generation")
        try:
            generations = tuple(
                gateway.begin_generation(
                    seed_gripper_efforts=seed_gripper_efforts,
                    seed_gripper_angles_um=seed_gripper_angles_um,
                )
                for gateway in (self.left, self.right)
            )
            if len(set(int(value) for value in generations)) != 1:
                raise LinkageGatewayError("left/right gateway generations diverged")
            with self._condition:
                self._generation = int(generations[0])
                self._active = True
                self._automatic_dispatch = False
                self._dispatch_seq = 0
                self._last_dispatched_sequences = {
                    ArmId.SLAVE_LEFT: 0,
                    ArmId.SLAVE_RIGHT: 0,
                }
                self._last_dispatch_receipt = None
                self._transition_dispatch_receipt = None
                self._supervisor_stop_event.clear()
                self._condition.notify_all()
                return self._generation
        except BaseException:
            _parallel_gateways((self.left, self.right), lambda gateway: gateway.stop())
            self._synchronize_generation_counters()
            raise

    def start(self, *, automatic_dispatch: bool = True) -> int:
        with self._condition:
            self._require_healthy_locked()
            if not self._active:
                raise LinkageGatewayError("begin_generation before starting the bimanual gateway")
            existing_thread = self._supervisor_thread
            if existing_thread is not None and existing_thread.is_alive():
                if self._automatic_dispatch != bool(automatic_dispatch):
                    raise LinkageGatewayError("gateway dispatch mode cannot change while running")
                return self._generation
            self._automatic_dispatch = bool(automatic_dispatch)
        try:
            generations = tuple(gateway.start() for gateway in (self.left, self.right))
            if len(set(int(value) for value in generations)) != 1:
                raise LinkageGatewayError("left/right gateway generations diverged")
            if automatic_dispatch:
                with self._condition:
                    self._supervisor_stop_event.clear()
                    self._supervisor_thread = threading.Thread(
                        target=self._automatic_dispatch_loop,
                        name="piper_linkage_supervisor",
                        daemon=True,
                    )
                    self._supervisor_thread.start()
            return self.generation
        except BaseException as exc:
            self._latch_global_fault(exc)
            raise GatewayFaultLatchedError(exc) from exc

    def _candidate_pair_locked(
        self,
        *,
        require_new: bool,
    ) -> tuple[CanonicalTargetCommit, CanonicalTargetCommit] | None:
        with self.left._lock:
            with self.right._lock:
                left_commit = self.left._last_commit
                right_commit = self.right._last_commit
                if (
                    not self.left._active
                    or not self.right._active
                    or left_commit is None
                    or right_commit is None
                ):
                    return None
                if (
                    left_commit.generation != self._generation
                    or right_commit.generation != self._generation
                ):
                    raise LinkageGatewayError(
                        "left/right candidates are outside the active generation"
                    )
                if require_new and (
                    left_commit.frame_group_seq
                    <= self._last_dispatched_sequences[ArmId.SLAVE_LEFT]
                    and right_commit.frame_group_seq
                    <= self._last_dispatched_sequences[ArmId.SLAVE_RIGHT]
                ):
                    return None
                return left_commit, right_commit

    def _wait_for_candidate_pair(
        self,
        *,
        deadline_s: float,
        require_new: bool,
    ) -> tuple[CanonicalTargetCommit, CanonicalTargetCommit]:
        with self._condition:
            while True:
                self._require_healthy_locked()
                if not self._active:
                    raise LinkageGatewayError(
                        "bimanual gateway stopped before a complete candidate pair"
                    )
                try:
                    candidates = self._candidate_pair_locked(require_new=require_new)
                except BaseException as exc:
                    self._latch_global_fault(exc)
                    raise GatewayFaultLatchedError(exc) from exc
                if candidates is not None:
                    return candidates
                remaining_s = deadline_s - time.monotonic()
                if remaining_s <= 0.0:
                    raise GatewayReadyTimeoutError(
                        "left/right gateways did not publish a new fresh complete "
                        "candidate pair before the deadline"
                    )
                self._condition.wait(timeout=remaining_s)

    def _dispatch_candidate_pair(
        self,
        candidates: tuple[CanonicalTargetCommit, CanonicalTargetCommit],
        *,
        deadline_s: float,
        transition_dispatch: bool,
    ) -> PairedDispatchReceipt:
        left_commit, right_commit = candidates
        with self._dispatch_lock:
            with self._condition:
                self._require_healthy_locked()
                if not self._active:
                    raise LinkageGatewayError("bimanual gateway stopped before dispatch")
                try:
                    current = self._candidate_pair_locked(require_new=True)
                except BaseException as exc:
                    self._latch_global_fault(exc)
                    raise GatewayFaultLatchedError(exc) from exc
                if current is None:
                    raise GatewayDispatchInterruptedError(
                        "candidate pair was already dispatched or invalidated"
                    )
                if current != candidates:
                    candidates = current
                    left_commit, right_commit = candidates

            left_messages = left_commit.to_can_messages()
            right_messages = right_commit.to_can_messages()
            try:
                for index in range(max(len(left_messages), len(right_messages))):
                    for gateway, messages in (
                        (self.left, left_messages),
                        (self.right, right_messages),
                    ):
                        if index >= len(messages):
                            continue
                        message = messages[index]
                        try:
                            with self._condition:
                                self._require_healthy_locked()
                                if not self._active:
                                    raise GatewayDispatchInterruptedError(
                                        "bimanual gateway stopped during paired dispatch"
                                    )
                                remaining_s = deadline_s - time.monotonic()
                                if remaining_s <= 0.0:
                                    raise GatewayDispatchTimeoutError(
                                        "paired gateway dispatch exceeded its absolute deadline"
                                    )
                            # The dispatch lock remains the sole send owner.  Do
                            # not hold the state condition across a potentially
                            # blocking CAN send: a bounded stop must be able to
                            # fault-fence the pair if this call exceeds its
                            # advertised timeout.
                            gateway.slave_bus.send(message, timeout=remaining_s)
                        except (GatewayFaultLatchedError, GatewayDispatchInterruptedError):
                            raise
                        except BaseException as exc:
                            raise GatewayTransmitError(
                                gateway.arm_id,
                                int(message.arbitration_id),
                                exc,
                            ) from exc
                        with gateway._lock:
                            gateway.counters.transmitted_frames += 1
            except (GatewayFaultLatchedError, GatewayDispatchInterruptedError):
                raise
            except BaseException as exc:
                self._latch_global_fault(exc)
                raise GatewayFaultLatchedError(exc) from exc

            completed_at_s = time.monotonic()
            with self._condition:
                self._require_healthy_locked()
                self._dispatch_seq += 1
                receipt = PairedDispatchReceipt(
                    generation=self._generation,
                    dispatch_seq=self._dispatch_seq,
                    completed_at_s=completed_at_s,
                    left_commit=left_commit,
                    right_commit=right_commit,
                )
                self._last_dispatched_sequences = {
                    ArmId.SLAVE_LEFT: left_commit.frame_group_seq,
                    ArmId.SLAVE_RIGHT: right_commit.frame_group_seq,
                }
                self._last_dispatch_receipt = receipt
                if transition_dispatch:
                    self._transition_dispatch_receipt = receipt
                self._condition.notify_all()
                return receipt

    def _automatic_dispatch_loop(self) -> None:
        wait_interval_s = max(
            self.left.config.recv_timeout_s,
            self.right.config.recv_timeout_s,
        )
        dispatch_timeout_s = min(
            self.left.config.dispatch_timeout_s,
            self.right.config.dispatch_timeout_s,
        )
        while not self._supervisor_stop_event.is_set():
            try:
                candidates = self._wait_for_candidate_pair(
                    deadline_s=time.monotonic() + wait_interval_s,
                    require_new=True,
                )
            except GatewayReadyTimeoutError:
                continue
            except (GatewayFaultLatchedError, LinkageGatewayError):
                break
            if self._supervisor_stop_event.is_set():
                break
            with self._lock:
                transition_dispatch = self._transition_dispatch_receipt is None
            try:
                self._dispatch_candidate_pair(
                    candidates,
                    deadline_s=time.monotonic() + dispatch_timeout_s,
                    transition_dispatch=transition_dispatch,
                )
            except (GatewayFaultLatchedError, LinkageGatewayError):
                break

    def wait_until_ready(
        self,
        *,
        timeout_s: float,
    ) -> PairedDispatchReceipt:
        if timeout_s <= 0.0:
            raise ValueError("timeout_s must be positive")
        deadline_s = time.monotonic() + timeout_s
        with self._condition:
            if self._automatic_dispatch:
                while self._transition_dispatch_receipt is None:
                    self._require_healthy_locked()
                    if not self._active:
                        raise LinkageGatewayError(
                            "bimanual gateway stopped before transition dispatch"
                        )
                    remaining_s = deadline_s - time.monotonic()
                    if remaining_s <= 0.0:
                        raise GatewayReadyTimeoutError(
                            "automatic gateway supervisor did not complete the transition "
                            "dispatch before the deadline"
                        )
                    self._condition.wait(timeout=remaining_s)
                return self._transition_dispatch_receipt
            existing = self._transition_dispatch_receipt
        if existing is not None:
            return existing
        candidates = self._wait_for_candidate_pair(
            deadline_s=deadline_s,
            require_new=True,
        )
        return self._dispatch_candidate_pair(
            candidates,
            deadline_s=deadline_s,
            transition_dispatch=True,
        )

    def wait_for_target_pair(
        self,
        *,
        timeout_s: float,
    ) -> Mapping[ArmId, CanonicalTargetCommit]:
        """Wait for coherent left/right targets without transmitting them."""

        if timeout_s <= 0.0:
            raise ValueError("timeout_s must be positive")
        left_commit, right_commit = self._wait_for_candidate_pair(
            deadline_s=time.monotonic() + timeout_s,
            require_new=False,
        )
        return {
            ArmId.SLAVE_LEFT: left_commit,
            ArmId.SLAVE_RIGHT: right_commit,
        }

    def dispatch_next_pair(self, *, timeout_s: float | None = None) -> PairedDispatchReceipt:
        with self._condition:
            self._require_healthy_locked()
            if self._automatic_dispatch:
                raise LinkageGatewayError(
                    "dispatch_next_pair is unavailable while automatic dispatch is active"
                )
            if self._transition_dispatch_receipt is None:
                raise LinkageGatewayError("wait_until_ready before stable paired dispatch")
        effective_timeout_s = (
            min(
                self.left.config.dispatch_timeout_s,
                self.right.config.dispatch_timeout_s,
            )
            if timeout_s is None
            else float(timeout_s)
        )
        if effective_timeout_s <= 0.0:
            raise ValueError("timeout_s must be positive")
        candidates = self._wait_for_candidate_pair(
            deadline_s=time.monotonic() + effective_timeout_s,
            require_new=True,
        )
        return self._dispatch_candidate_pair(
            candidates,
            deadline_s=time.monotonic() + effective_timeout_s,
            transition_dispatch=False,
        )

    def require_fresh_family(self) -> Mapping[ArmId, CanonicalTargetCommit]:
        with self._condition:
            self._require_healthy_locked()
            candidates = self._candidate_pair_locked(require_new=False)
            if candidates is None:
                raise LinkageGatewayError(
                    "left/right gateways do not have a fresh complete candidate pair"
                )
            left_commit, right_commit = candidates
            return {
                ArmId.SLAVE_LEFT: left_commit,
                ArmId.SLAVE_RIGHT: right_commit,
            }

    def stop(self, *, timeout_s: float | None = None) -> None:
        """Stop after an already-started paired dispatch reaches its boundary.

        Dynamic authority changes use :meth:`stop_at_cycle_boundary`, whose
        decoder-drain semantics are intentionally separate.  This method is the
        normal shutdown path: it prevents a new automatic dispatch, waits a
        bounded interval for the current immutable pair to finish, and only then
        fences both semantic gateways.  A sender that exceeds the boundary is
        fault-fenced so upper layers can apply their existing hold policy.
        """

        boundary_timeout_s = (
            min(
                self.left.config.dispatch_timeout_s,
                self.right.config.dispatch_timeout_s,
            )
            + max(
                self.left.config.recv_timeout_s,
                self.right.config.recv_timeout_s,
            )
            if timeout_s is None
            else float(timeout_s)
        )
        if boundary_timeout_s <= 0.0:
            raise ValueError("timeout_s must be positive")

        # This event is independent of the condition lock.  Setting it before
        # waiting for the dispatch owner prevents the automatic supervisor from
        # starting another pair while shutdown is queued behind the current one.
        self._supervisor_stop_event.set()
        dispatch_boundary_reached = self._dispatch_lock.acquire(
            timeout=boundary_timeout_s
        )
        if not dispatch_boundary_reached:
            fault = GatewayBoundaryTimeoutError(
                "bimanual gateway normal stop did not reach a paired dispatch "
                f"boundary within {boundary_timeout_s:.3f}s"
            )
            self._latch_global_fault(fault)
            raise fault

        try:
            with self._condition:
                was_running = self._active or (
                    self._supervisor_thread is not None
                    and self._supervisor_thread.is_alive()
                )
                self._active = False
                self._generation = max(
                    self._generation,
                    self.left.generation,
                    self.right.generation,
                )
                if was_running:
                    self._generation += 1
                generation = self._generation
                self._condition.notify_all()
                supervisor_thread = self._supervisor_thread
            self.left._fence_from_owner(generation=generation, fault=None)
            self.right._fence_from_owner(generation=generation, fault=None)
        finally:
            self._dispatch_lock.release()
        if (
            supervisor_thread is not None
            and supervisor_thread is not threading.current_thread()
        ):
            supervisor_thread.join(
                timeout=max(
                    0.1,
                    self.left.config.recv_timeout_s * 3.0,
                    self.right.config.recv_timeout_s * 3.0,
                )
            )
        _parallel_gateways((self.left, self.right), lambda gateway: gateway.stop())
        self._synchronize_generation_counters()
        with self._condition:
            if (
                self._supervisor_thread is supervisor_thread
                and (supervisor_thread is None or not supervisor_thread.is_alive())
            ):
                self._supervisor_thread = None
            self._condition.notify_all()

    def stop_at_cycle_boundary(self, *, timeout_s: float) -> bool:
        if timeout_s <= 0.0:
            raise ValueError("timeout_s must be positive")
        deadline_s = time.monotonic() + timeout_s
        with self._condition:
            self._supervisor_stop_event.set()
            self._condition.notify_all()
            supervisor_thread = self._supervisor_thread
        if (
            supervisor_thread is not None
            and supervisor_thread is not threading.current_thread()
        ):
            supervisor_thread.join(timeout=max(0.0, deadline_s - time.monotonic()))
        remaining_s = deadline_s - time.monotonic()
        if remaining_s <= 0.0:
            fault = GatewayBoundaryTimeoutError(
                "bimanual gateway dispatch did not stop before the boundary deadline"
            )
            self._latch_global_fault(fault)
            raise fault
        dispatch_boundary_reached = self._dispatch_lock.acquire(
            timeout=remaining_s
        )
        if not dispatch_boundary_reached:
            fault = GatewayBoundaryTimeoutError(
                "bimanual gateway dispatch did not stop before the boundary deadline"
            )
            self._latch_global_fault(fault)
            raise fault
        try:
            remaining_s = deadline_s - time.monotonic()
            if remaining_s <= 0.0:
                fault = GatewayBoundaryTimeoutError(
                    "bimanual gateway dispatch did not stop before the boundary deadline"
                )
                self._latch_global_fault(fault)
                raise fault
            try:
                results = _parallel_gateways(
                    (self.left, self.right),
                    lambda gateway: gateway.stop_at_cycle_boundary(
                        timeout_s=remaining_s
                    ),
                )
            except BaseException as exc:
                self._latch_global_fault(exc)
                raise
            if time.monotonic() > deadline_s:
                fault = GatewayBoundaryTimeoutError(
                    "bimanual gateway semantic decoders stopped after the boundary deadline"
                )
                self._latch_global_fault(fault)
                raise fault
        finally:
            self._dispatch_lock.release()
        self._synchronize_generation_counters()
        with self._condition:
            self._active = False
            self._supervisor_thread = None
            self._condition.notify_all()
        return all(bool(result) for result in results)

    def close(self, *, timeout_s: float | None = None) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        stop_error: BaseException | None = None
        try:
            self.stop(timeout_s=timeout_s)
        except BaseException as exc:
            stop_error = exc
        try:
            _parallel_gateways(
                (self.left, self.right),
                lambda gateway: gateway.close(),
            )
        except BaseException:
            if stop_error is None:
                raise
        if stop_error is not None:
            raise stop_error


def _parallel_gateways(
    gateways: Sequence[SemanticLinkageGateway],
    call: Callable[[SemanticLinkageGateway], Any],
) -> tuple[Any, ...]:
    with ThreadPoolExecutor(max_workers=len(gateways), thread_name_prefix="piper_gateway") as executor:
        futures = [executor.submit(call, gateway) for gateway in gateways]
        return tuple(future.result() for future in futures)


__all__ = [
    "BimanualLinkageGateway",
    "GRIPPER_CTRL_ID",
    "GatewayConfig",
    "CanonicalTargetCommit",
    "GatewayCounters",
    "GatewayDispatchTimeoutError",
    "GatewayDispatchInterruptedError",
    "GatewayFaultLatchedError",
    "GatewayBoundaryTimeoutError",
    "GatewayReadyTimeoutError",
    "GatewayTransmitError",
    "JOINT_CTRL_IDS",
    "LINKAGE_INPUT_IDS",
    "LinkageGatewayError",
    "MOTION_CTRL_ID",
    "PairedDispatchReceipt",
    "SemanticLinkageGateway",
]
