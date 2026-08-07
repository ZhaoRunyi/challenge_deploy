"""Single topology-selection boundary for Piper hardware construction.

Callers must invoke :func:`build_hardware` before constructing cameras.  The
factory selects one registered topology builder exactly once; the shared path
keeps the deployed two-bus behavior, while the isolated path validates adapter
identity before opening a Piper or SocketCAN object.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from .config import (
    CanInterfaceIdentity,
    discover_can_interfaces,
    validate_can_interface_serials,
    validate_config,
)
from .constants import PIPER_ARM_IDS
from .isolated import IsolatedFourArmSystem, IsolatedSlaveSystem
from .linkage_gateway import BimanualLinkageGateway, GatewayConfig, SemanticLinkageGateway
from .piper import DualPiperSystem
from .topology import ArmId


SLAVE_ARM_IDS: tuple[str, str] = ("slave_left", "slave_right")


class HardwareFactoryError(ValueError):
    """Raised when a requested topology/session combination is unsupported."""


@dataclass(frozen=True, slots=True)
class HardwareAssembly:
    robot: Any
    topology: str
    constructed_arm_ids: tuple[str, ...]
    gateway: Any | None = None


def _default_semantic_gateway_factory(
    *,
    arm_id: ArmId,
    master_can_name: str,
    slave_can_name: str,
    config: GatewayConfig,
) -> SemanticLinkageGateway:
    return SemanticLinkageGateway.from_socketcan(
        arm_id=arm_id,
        master_can_name=master_can_name,
        slave_can_name=slave_can_name,
        config=config,
    )


@dataclass(frozen=True, slots=True)
class HardwareFactoryDependencies:
    """Injectable hardware edges used by deterministic, zero-hardware tests."""

    discovery_factory: Callable[[], Mapping[str, str | CanInterfaceIdentity]] = (
        discover_can_interfaces
    )
    shared_robot_factory: Callable[..., Any] = DualPiperSystem
    isolated_arm_factory: Callable[[ArmId, str, bool], Any] | None = None
    semantic_gateway_factory: Callable[..., Any] = _default_semantic_gateway_factory
    bimanual_gateway_factory: Callable[[Any, Any], Any] = BimanualLinkageGateway


@dataclass(frozen=True, slots=True)
class _BuildRequest:
    config: Mapping[str, Any]
    intervention: bool
    commands_enabled: bool
    prefer_joint_ctrl: bool
    name: str
    dependencies: HardwareFactoryDependencies


def _build_shared(request: _BuildRequest) -> HardwareAssembly:
    if request.intervention:
        raise HardwareFactoryError(
            "intervention requires can_topology='isolated'; refusing before any "
            "CAN or camera construction"
        )

    # Golden path: no discovery, serial gate, role write, or gateway.
    robot_config = request.config["robot"]
    robot = request.dependencies.shared_robot_factory(
        left_can_name=robot_config["slave_left"]["can_name"],
        right_can_name=robot_config["slave_right"]["can_name"],
        commands_enabled=request.commands_enabled,
        prefer_joint_ctrl=request.prefer_joint_ctrl,
        name=request.name,
    )
    return HardwareAssembly(
        robot=robot,
        topology="shared",
        constructed_arm_ids=SLAVE_ARM_IDS,
        gateway=None,
    )


def _discover_and_validate_isolated(
    request: _BuildRequest,
    required_arm_ids: tuple[str, ...],
) -> None:
    discovered = request.dependencies.discovery_factory()
    validate_can_interface_serials(
        request.config,
        required_arm_ids=required_arm_ids,
        discovered=discovered,
    )


def _build_isolated_slave(request: _BuildRequest) -> HardwareAssembly:
    required = SLAVE_ARM_IDS
    validate_config(request.config, required_arm_ids=required)
    _discover_and_validate_isolated(request, required)
    robot = IsolatedSlaveSystem.from_config(
        request.config,
        commands_enabled=request.commands_enabled,
        arm_factory=request.dependencies.isolated_arm_factory,
    )
    return HardwareAssembly(
        robot=robot,
        topology="isolated",
        constructed_arm_ids=required,
        gateway=None,
    )


def gateway_config_for_side(
    config: Mapping[str, Any],
    side: str,
) -> GatewayConfig:
    if side not in ("left", "right"):
        raise ValueError("Piper side must be 'left' or 'right'")
    robot = config["robot"]
    slave_id = f"slave_{side}"
    return GatewayConfig(
        gripper_effort=int(robot[slave_id].get("gripper_effort", 1000)),
    )


def _close_gateway_quietly(gateway: Any) -> None:
    try:
        gateway.close()
    except Exception:
        pass


def _build_bimanual_gateway(
    request: _BuildRequest,
    configs: tuple[GatewayConfig, GatewayConfig],
) -> Any:
    robot = request.config["robot"]
    left = request.dependencies.semantic_gateway_factory(
        arm_id=ArmId.SLAVE_LEFT,
        master_can_name=robot["master_left"]["can_name"],
        slave_can_name=robot["slave_left"]["can_name"],
        config=configs[0],
    )
    try:
        right = request.dependencies.semantic_gateway_factory(
            arm_id=ArmId.SLAVE_RIGHT,
            master_can_name=robot["master_right"]["can_name"],
            slave_can_name=robot["slave_right"]["can_name"],
            config=configs[1],
        )
    except BaseException:
        _close_gateway_quietly(left)
        raise
    try:
        return request.dependencies.bimanual_gateway_factory(left, right)
    except BaseException:
        _close_gateway_quietly(left)
        _close_gateway_quietly(right)
        raise


def _build_isolated_intervention(request: _BuildRequest) -> HardwareAssembly:
    required = tuple(PIPER_ARM_IDS)
    validate_config(request.config, required_arm_ids=required)
    # Build protocol settings before discovery or any CAN constructor.
    gateway_configs = (
        gateway_config_for_side(request.config, "left"),
        gateway_config_for_side(request.config, "right"),
    )
    _discover_and_validate_isolated(request, required)
    robot = IsolatedFourArmSystem.from_config(
        request.config,
        commands_enabled=request.commands_enabled,
        arm_factory=request.dependencies.isolated_arm_factory,
    )
    try:
        gateway = _build_bimanual_gateway(request, gateway_configs)
    except BaseException:
        robot.abort_construction()
        raise
    return HardwareAssembly(
        robot=robot,
        topology="isolated",
        constructed_arm_ids=required,
        gateway=gateway,
    )


_ISOLATED_BUILDERS: Mapping[bool, Callable[[_BuildRequest], HardwareAssembly]] = {
    False: _build_isolated_slave,
    True: _build_isolated_intervention,
}


def _build_isolated(request: _BuildRequest) -> HardwareAssembly:
    return _ISOLATED_BUILDERS[request.intervention](request)


_TOPOLOGY_BUILDERS: Mapping[str, Callable[[_BuildRequest], HardwareAssembly]] = {
    "shared": _build_shared,
    "isolated": _build_isolated,
}


def build_hardware(
    config: Mapping[str, Any],
    *,
    intervention: bool = False,
    commands_enabled: bool = True,
    prefer_joint_ctrl: bool = False,
    name: str = "dual_piper",
    dependencies: HardwareFactoryDependencies | None = None,
) -> HardwareAssembly:
    """Validate, select one topology builder, and construct its hardware graph."""

    validate_config(config, required_arm_ids=())
    topology = str(config["can_topology"])
    request = _BuildRequest(
        config=config,
        intervention=bool(intervention),
        commands_enabled=bool(commands_enabled),
        prefer_joint_ctrl=bool(prefer_joint_ctrl),
        name=str(name),
        dependencies=dependencies or HardwareFactoryDependencies(),
    )
    return _TOPOLOGY_BUILDERS[topology](request)


__all__ = [
    "HardwareAssembly",
    "HardwareFactoryDependencies",
    "HardwareFactoryError",
    "SLAVE_ARM_IDS",
    "build_hardware",
    "gateway_config_for_side",
]
