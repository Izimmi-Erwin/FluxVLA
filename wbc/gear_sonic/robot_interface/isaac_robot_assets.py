"""Isaac robot asset selection for Sonic G1 validation.

The WBC bridge controls the 29 body joints named in ``SONIC_BODY_JOINT_NAMES``.
Robot USD selection is centralized here so launchers do not hard-code a USD,
articulation root, or collision-layer compatibility independently.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from gear_sonic.robot_interface.joint_mapping import SONIC_BODY_JOINT_NAMES


GEAR_SONIC_ROOT = Path(__file__).resolve().parents[1]
G1_ROBOT_ROOT = GEAR_SONIC_ROOT / "data" / "robots" / "g1"


@dataclass(frozen=True)
class IsaacRobotAsset:
    name: str
    usd_path: Path
    prim_path: str
    articulation_root: str
    root_body_prim_path: str
    body_joint_names: tuple[str, ...]
    left_hand_joint_names: tuple[str, ...]
    right_hand_joint_names: tuple[str, ...]
    expected_body_dof: int
    expected_total_dof: int
    foot_prim_paths: tuple[str, ...]
    imu_prim_paths: tuple[str, ...]
    collision_layer_path: Path | None
    collision_calibration_compatible: bool
    notes: str = ""

    @property
    def hand_joint_names(self) -> tuple[str, ...]:
        return (*self.left_hand_joint_names, *self.right_hand_joint_names)

    @property
    def hand_dof_available(self) -> bool:
        return bool(self.hand_joint_names)

    def resolved(self) -> "IsaacRobotAsset":
        return IsaacRobotAsset(
            name=self.name,
            usd_path=self.usd_path.expanduser().resolve(),
            prim_path=self.prim_path,
            articulation_root=self.articulation_root,
            root_body_prim_path=self.root_body_prim_path,
            body_joint_names=self.body_joint_names,
            left_hand_joint_names=self.left_hand_joint_names,
            right_hand_joint_names=self.right_hand_joint_names,
            expected_body_dof=self.expected_body_dof,
            expected_total_dof=self.expected_total_dof,
            foot_prim_paths=self.foot_prim_paths,
            imu_prim_paths=self.imu_prim_paths,
            collision_layer_path=self.collision_layer_path.expanduser().resolve()
            if self.collision_layer_path is not None
            else None,
            collision_calibration_compatible=self.collision_calibration_compatible,
            notes=self.notes,
        )

    def as_dict(self) -> dict[str, object]:
        resolved = self.resolved()
        return {
            "name": resolved.name,
            "usd_path": str(resolved.usd_path),
            "prim_path": resolved.prim_path,
            "articulation_root": resolved.articulation_root,
            "root_body_prim_path": resolved.root_body_prim_path,
            "body_joint_count": len(resolved.body_joint_names),
            "expected_body_dof": resolved.expected_body_dof,
            "expected_total_dof": resolved.expected_total_dof,
            "hand_dof_available": resolved.hand_dof_available,
            "left_hand_joint_names": list(resolved.left_hand_joint_names),
            "right_hand_joint_names": list(resolved.right_hand_joint_names),
            "foot_prim_paths": list(resolved.foot_prim_paths),
            "imu_prim_paths": list(resolved.imu_prim_paths),
            "collision_layer_path": str(resolved.collision_layer_path)
            if resolved.collision_layer_path is not None
            else None,
            "collision_calibration_compatible": resolved.collision_calibration_compatible,
            "notes": resolved.notes,
        }


OFFICIAL_G1_29DOF = IsaacRobotAsset(
    name="official_g1_29dof",
    usd_path=G1_ROBOT_ROOT / "g1_29dof_rev_1_0" / "g1_29dof_rev_1_0.usd",
    prim_path="/World/G1",
    articulation_root="/World/G1/pelvis",
    root_body_prim_path="/World/G1/pelvis",
    body_joint_names=SONIC_BODY_JOINT_NAMES,
    left_hand_joint_names=(),
    right_hand_joint_names=(),
    expected_body_dof=29,
    expected_total_dof=29,
    foot_prim_paths=(
        "/World/G1/left_ankle_roll_link",
        "/World/G1/right_ankle_roll_link",
    ),
    imu_prim_paths=(
        "/World/G1/pelvis/imu_in_pelvis",
        "/World/G1/torso_link/imu_in_torso",
    ),
    collision_layer_path=None,
    collision_calibration_compatible=False,
    notes=(
        "Official Unitree G1 29DOF USD present in the local unitree_model layout. "
        "No Isaac-only collision calibration layer is applied by default."
    ),
)

SONIC_G1_43DOF = IsaacRobotAsset(
    name="sonic_g1_43dof",
    usd_path=G1_ROBOT_ROOT / "g1_sonic_usd" / "g1_43dof_main.usd",
    prim_path="/World/G1",
    articulation_root="/World/G1/pelvis",
    root_body_prim_path="/World/G1/pelvis",
    body_joint_names=SONIC_BODY_JOINT_NAMES,
    left_hand_joint_names=(),
    right_hand_joint_names=(),
    expected_body_dof=29,
    expected_total_dof=29,
    foot_prim_paths=(
        "/World/G1/left_ankle_roll_link",
        "/World/G1/right_ankle_roll_link",
    ),
    imu_prim_paths=(
        "/World/G1/imu_in_pelvis",
        "/World/G1/imu_in_torso",
    ),
    collision_layer_path=None,
    collision_calibration_compatible=False,
    notes=(
        "Compatibility asset. Despite the filename, Isaac exposes 29 controllable "
        "body DOFs in the current articulation audit. No Isaac-only collision "
        "calibration layer is applied by default."
    ),
)

ISAAC_ROBOT_ASSETS: Mapping[str, IsaacRobotAsset] = {
    OFFICIAL_G1_29DOF.name: OFFICIAL_G1_29DOF,
    SONIC_G1_43DOF.name: SONIC_G1_43DOF,
}

DEFAULT_ROBOT_MODEL = OFFICIAL_G1_29DOF.name


def available_robot_models() -> tuple[str, ...]:
    return tuple(ISAAC_ROBOT_ASSETS)


def get_robot_asset(name: str | None) -> IsaacRobotAsset:
    key = name or DEFAULT_ROBOT_MODEL
    try:
        return ISAAC_ROBOT_ASSETS[key].resolved()
    except KeyError as exc:
        raise ValueError(f"unknown robot model {key!r}; expected one of {available_robot_models()}") from exc
