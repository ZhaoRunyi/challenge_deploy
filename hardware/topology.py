from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
import time
from typing import Any, Mapping


class ArmId(str, Enum):
    """Stable physical identities; linkage roles never rename an arm."""

    MASTER_LEFT = "master_left"
    MASTER_RIGHT = "master_right"
    SLAVE_LEFT = "slave_left"
    SLAVE_RIGHT = "slave_right"

    @property
    def side(self) -> str:
        return "left" if self in (ArmId.MASTER_LEFT, ArmId.SLAVE_LEFT) else "right"

    @property
    def is_master(self) -> bool:
        return self in (ArmId.MASTER_LEFT, ArmId.MASTER_RIGHT)

    @property
    def is_slave(self) -> bool:
        return not self.is_master


MASTER_ARM_IDS = (ArmId.MASTER_LEFT, ArmId.MASTER_RIGHT)
SLAVE_ARM_IDS = (ArmId.SLAVE_LEFT, ArmId.SLAVE_RIGHT)
ALL_ARM_IDS = (*SLAVE_ARM_IDS, *MASTER_ARM_IDS)


class LinkageRole(IntEnum):
    """Values carried in byte zero of Piper command 0x470."""

    TEACHING_INPUT = 0xFA
    MOTION_OUTPUT = 0xFC

    @property
    def short_name(self) -> str:
        return "FA" if self is LinkageRole.TEACHING_INPUT else "FC"


class StaticMode(str, Enum):
    TELEOP = "teleop"
    ROLLOUT = "rollout"


class PiperTopologyError(RuntimeError):
    """Base class for isolated-topology failures."""


class TopologyConfigError(PiperTopologyError, ValueError):
    pass


class ArmOperationError(PiperTopologyError):
    def __init__(self, arm_id: ArmId, operation: str, cause: BaseException | str) -> None:
        self.arm_id = arm_id
        self.operation = operation
        self.cause = cause
        super().__init__(f"{operation} failed on {arm_id.value}: {cause}")


class BatchOperationError(PiperTopologyError):
    def __init__(self, result: "BatchOperationResult") -> None:
        self.result = result
        failures = ", ".join(
            f"{item.arm_id.value}: {item.error}" for item in result.results if not item.ok
        )
        super().__init__(f"{result.operation} failed ({failures})")


class RoleTransactionError(PiperTopologyError):
    def __init__(
        self,
        message: str,
        *,
        target: LinkageRole,
        attempts: int,
        writes: tuple["RoleWriteResult", ...] = (),
        rollbacks: tuple["RoleWriteResult", ...] = (),
    ) -> None:
        self.target = target
        self.attempts = attempts
        self.writes = writes
        self.rollbacks = rollbacks
        super().__init__(message)


class LatchedFaultError(PiperTopologyError):
    def __init__(self, fault: "FaultRecord") -> None:
        self.fault = fault
        super().__init__(
            f"isolated hardware is fault-latched after {fault.operation}"
            f" on {fault.arm_id.value if fault.arm_id else 'system'}: {fault.message}"
        )


class TransitionInProgressError(PiperTopologyError):
    pass


@dataclass(frozen=True, slots=True)
class ArmOperationResult:
    arm_id: ArmId
    operation: str
    started_at_s: float
    finished_at_s: float
    value: Any = None
    error: BaseException | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass(frozen=True, slots=True)
class BatchOperationResult:
    operation: str
    results: tuple[ArmOperationResult, ...]

    @property
    def ok(self) -> bool:
        return all(result.ok for result in self.results)

    @property
    def failures(self) -> tuple[ArmOperationResult, ...]:
        return tuple(result for result in self.results if not result.ok)

    def require_success(self) -> "BatchOperationResult":
        if not self.ok:
            raise BatchOperationError(self)
        return self


@dataclass(frozen=True, slots=True)
class RoleObservation:
    arm_id: ArmId
    role: LinkageRole | None
    observed_at_s: float
    evidence: str
    fresh: bool

    @classmethod
    def unknown(cls, arm_id: ArmId, evidence: str) -> "RoleObservation":
        return cls(
            arm_id=arm_id,
            role=None,
            observed_at_s=time.time(),
            evidence=evidence,
            fresh=False,
        )


@dataclass(frozen=True, slots=True)
class RoleWriteResult:
    arm_id: ArmId
    previous: LinkageRole | None
    target: LinkageRole
    written: bool
    verified: bool
    observation: RoleObservation
    error: BaseException | None = None
    motion_output_seeded: bool = False
    pending: bool = False

    @property
    def ok(self) -> bool:
        return self.error is None and (
            (not self.written and self.observation.role is self.target and self.observation.fresh)
            or (self.written and self.verified and self.observation.role is self.target)
        )

    @property
    def accepted(self) -> bool:
        """Return whether the write is confirmed or awaiting event-driven FA traffic."""

        return self.ok or (self.error is None and self.pending)


@dataclass(frozen=True, slots=True)
class RoleTransactionResult:
    target: LinkageRole
    attempts: int
    writes: tuple[RoleWriteResult, ...]
    rollbacks: tuple[RoleWriteResult, ...] = ()

    @property
    def ok(self) -> bool:
        # ``writes`` retains failed first-attempt evidence for diagnostics.  A
        # successful retry is judged by the final whole-pair attempt, not by
        # the deliberately retained history.
        return len(self.writes) >= 2 and all(result.ok for result in self.writes[-2:])

    @property
    def accepted(self) -> bool:
        return len(self.writes) >= 2 and all(
            result.accepted for result in self.writes[-2:]
        )


@dataclass(frozen=True, slots=True)
class FaultRecord:
    operation: str
    message: str
    arm_id: ArmId | None = None
    occurred_at_s: float = field(default_factory=time.time)
    partial_result: BatchOperationResult | None = None


@dataclass(frozen=True, slots=True)
class PhysicalArmStates:
    states: Mapping[ArmId, Any]
    captured_at_s: float

    def __getitem__(self, arm_id: ArmId) -> Any:
        return self.states[arm_id]

    @property
    def slave_left(self) -> Any:
        return self.states[ArmId.SLAVE_LEFT]

    @property
    def slave_right(self) -> Any:
        return self.states[ArmId.SLAVE_RIGHT]


@dataclass(frozen=True, slots=True)
class LifecycleResult:
    mode: StaticMode | str
    generation: int
    operations: tuple[BatchOperationResult | RoleTransactionResult, ...]
    physical_states: PhysicalArmStates | None = None


def _can_names_for(
    robot_config: Mapping[str, Any],
    arm_ids: tuple[ArmId, ...],
) -> dict[ArmId, str]:
    names: dict[ArmId, str] = {}
    missing: list[str] = []
    for arm_id in arm_ids:
        arm_section = robot_config.get(arm_id.value)
        can_name = arm_section.get("can_name") if isinstance(arm_section, Mapping) else None
        if not isinstance(can_name, str) or not can_name.strip():
            missing.append(f"robot.{arm_id.value}.can_name")
            continue
        names[arm_id] = can_name.strip()
    if missing:
        raise TopologyConfigError("missing isolated CAN configuration: " + ", ".join(missing))

    duplicates = sorted({name for name in names.values() if list(names.values()).count(name) > 1})
    if duplicates:
        raise TopologyConfigError(
            f"isolated topology requires {len(arm_ids)} distinct CAN interfaces; duplicated: "
            + ", ".join(duplicates)
        )
    return names


def isolated_slave_can_names(robot_config: Mapping[str, Any]) -> dict[ArmId, str]:
    return _can_names_for(robot_config, SLAVE_ARM_IDS)


def isolated_can_names(robot_config: Mapping[str, Any]) -> dict[ArmId, str]:
    """Read only the explicit four-arm isolated schema.

    ``robot_config`` is the mapping below the top-level ``robot`` key.  The
    legacy ``left``/``right`` aliases are intentionally not consulted here.
    """

    return _can_names_for(robot_config, ALL_ARM_IDS)
