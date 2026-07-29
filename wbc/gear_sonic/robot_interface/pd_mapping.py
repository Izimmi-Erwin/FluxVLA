"""PD/drive parameter mapping for Sonic G1 Isaac control calibration.

This module is simulator-import free. It maps named Sonic body joints to the
PD gains used by the GR00T lowcmd policy. The MuJoCo XML itself uses torque
motors and does not specify actuator-level kp/kd or explicit force ranges; the
torque motor ctrlrange is used as the Isaac max-effort bound for this pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from gear_sonic.robot_interface.joint_mapping import SONIC_BODY_JOINT_NAMES, build_joint_mapping


@dataclass(frozen=True)
class MujocoPDParam:
    joint: str
    actuator: str
    kp: float | None
    kd: float | None
    passive_damping: float
    ctrl_range: tuple[float, float]
    force_range: tuple[float, float]
    source: str


@dataclass(frozen=True)
class IsaacDriveParam:
    joint: str
    stiffness: float
    damping: float
    max_effort: float
    source: str


@dataclass(frozen=True)
class PDMappingRow:
    mujoco_joint: str
    isaac_joint: str
    sonic_index: int
    isaac_index: int
    kp: float
    kd: float
    effort_limit: float
    mujoco_kp: float | None
    mujoco_kd: float | None
    passive_damping: float
    source: str


_KP_5020 = 14.250623098688912
_KD_5020 = 0.9072228433183532
_KP_7520_14 = 40.17923847366998
_KD_7520_14 = 2.5578897651010553
_KP_7520_22 = 99.098427782326
_KD_7520_22 = 6.3088018536769574
_KP_4010 = 16.77832748185191
_KD_4010 = 1.0681415022205298


SONIC_STANDING_POSE: tuple[float, ...] = (
    -0.312,
    0.0,
    0.0,
    0.669,
    -0.363,
    0.0,
    -0.312,
    0.0,
    0.0,
    0.669,
    -0.363,
    0.0,
    0.0,
    0.0,
    0.0,
    0.2,
    0.2,
    0.0,
    0.6,
    0.0,
    0.0,
    0.0,
    0.2,
    -0.2,
    0.0,
    0.6,
    0.0,
    0.0,
    0.0,
)


def _param(
    joint: str,
    actuator: str,
    ctrl: float,
    *,
    passive_damping: float = 0.05,
    source: str = "g1_29dof_with_hand.xml torque motor",
) -> MujocoPDParam:
    return MujocoPDParam(
        joint=joint,
        actuator=actuator,
        kp=None,
        kd=None,
        passive_damping=passive_damping,
        ctrl_range=(-ctrl, ctrl),
        force_range=(-ctrl, ctrl),
        source=source,
    )


MUJOCO_PD_PARAMS: dict[str, MujocoPDParam] = {
    "left_hip_pitch_joint": _param("left_hip_pitch_joint", "left_hip_pitch", 88.0),
    "left_hip_roll_joint": _param("left_hip_roll_joint", "left_hip_roll", 88.0),
    "left_hip_yaw_joint": _param("left_hip_yaw_joint", "left_hip_yaw", 88.0),
    "left_knee_joint": _param("left_knee_joint", "left_knee", 139.0),
    "left_ankle_pitch_joint": _param("left_ankle_pitch_joint", "left_ankle_pitch", 50.0),
    "left_ankle_roll_joint": _param("left_ankle_roll_joint", "left_ankle_roll", 50.0),
    "right_hip_pitch_joint": _param("right_hip_pitch_joint", "right_hip_pitch", 88.0),
    "right_hip_roll_joint": _param("right_hip_roll_joint", "right_hip_roll", 88.0),
    "right_hip_yaw_joint": _param("right_hip_yaw_joint", "right_hip_yaw", 88.0),
    "right_knee_joint": _param("right_knee_joint", "right_knee", 139.0),
    "right_ankle_pitch_joint": _param("right_ankle_pitch_joint", "right_ankle_pitch", 50.0),
    "right_ankle_roll_joint": _param("right_ankle_roll_joint", "right_ankle_roll", 50.0),
    "waist_yaw_joint": _param("waist_yaw_joint", "waist_yaw", 88.0),
    "waist_roll_joint": _param("waist_roll_joint", "waist_roll", 50.0),
    "waist_pitch_joint": _param("waist_pitch_joint", "waist_pitch", 50.0),
    "left_shoulder_pitch_joint": _param("left_shoulder_pitch_joint", "left_shoulder_pitch", 25.0),
    "left_shoulder_roll_joint": _param("left_shoulder_roll_joint", "left_shoulder_roll", 25.0),
    "left_shoulder_yaw_joint": _param("left_shoulder_yaw_joint", "left_shoulder_yaw", 25.0),
    "left_elbow_joint": _param("left_elbow_joint", "left_elbow", 25.0),
    "left_wrist_roll_joint": _param("left_wrist_roll_joint", "left_wrist_roll", 25.0),
    "left_wrist_pitch_joint": _param("left_wrist_pitch_joint", "left_wrist_pitch", 5.0, passive_damping=0.05),
    "left_wrist_yaw_joint": _param("left_wrist_yaw_joint", "left_wrist_yaw", 5.0, passive_damping=0.05),
    "right_shoulder_pitch_joint": _param("right_shoulder_pitch_joint", "right_shoulder_pitch", 25.0),
    "right_shoulder_roll_joint": _param("right_shoulder_roll_joint", "right_shoulder_roll", 25.0),
    "right_shoulder_yaw_joint": _param("right_shoulder_yaw_joint", "right_shoulder_yaw", 25.0),
    "right_elbow_joint": _param("right_elbow_joint", "right_elbow", 25.0),
    "right_wrist_roll_joint": _param("right_wrist_roll_joint", "right_wrist_roll", 25.0),
    "right_wrist_pitch_joint": _param("right_wrist_pitch_joint", "right_wrist_pitch", 5.0, passive_damping=0.05),
    "right_wrist_yaw_joint": _param("right_wrist_yaw_joint", "right_wrist_yaw", 5.0, passive_damping=0.05),
}


def _drive(joint: str, kp: float, kd: float) -> IsaacDriveParam:
    effort = max(abs(v) for v in MUJOCO_PD_PARAMS[joint].force_range)
    return IsaacDriveParam(
        joint=joint,
        stiffness=kp,
        damping=kd,
        max_effort=effort,
        source="GR00T policy_parameters.hpp kp/kd + MuJoCo torque motor ctrlrange as max effort",
    )


ISAAC_DRIVE_PARAMS: dict[str, IsaacDriveParam] = {
    "left_hip_pitch_joint": _drive("left_hip_pitch_joint", _KP_7520_22, _KD_7520_22),
    "left_hip_roll_joint": _drive("left_hip_roll_joint", _KP_7520_22, _KD_7520_22),
    "left_hip_yaw_joint": _drive("left_hip_yaw_joint", _KP_7520_14, _KD_7520_14),
    "left_knee_joint": _drive("left_knee_joint", _KP_7520_22, _KD_7520_22),
    "left_ankle_pitch_joint": _drive("left_ankle_pitch_joint", 2.0 * _KP_5020, 2.0 * _KD_5020),
    "left_ankle_roll_joint": _drive("left_ankle_roll_joint", 2.0 * _KP_5020, 2.0 * _KD_5020),
    "right_hip_pitch_joint": _drive("right_hip_pitch_joint", _KP_7520_22, _KD_7520_22),
    "right_hip_roll_joint": _drive("right_hip_roll_joint", _KP_7520_22, _KD_7520_22),
    "right_hip_yaw_joint": _drive("right_hip_yaw_joint", _KP_7520_14, _KD_7520_14),
    "right_knee_joint": _drive("right_knee_joint", _KP_7520_22, _KD_7520_22),
    "right_ankle_pitch_joint": _drive("right_ankle_pitch_joint", 2.0 * _KP_5020, 2.0 * _KD_5020),
    "right_ankle_roll_joint": _drive("right_ankle_roll_joint", 2.0 * _KP_5020, 2.0 * _KD_5020),
    "waist_yaw_joint": _drive("waist_yaw_joint", _KP_7520_14, _KD_7520_14),
    "waist_roll_joint": _drive("waist_roll_joint", 2.0 * _KP_5020, 2.0 * _KD_5020),
    "waist_pitch_joint": _drive("waist_pitch_joint", 2.0 * _KP_5020, 2.0 * _KD_5020),
    "left_shoulder_pitch_joint": _drive("left_shoulder_pitch_joint", _KP_5020, _KD_5020),
    "left_shoulder_roll_joint": _drive("left_shoulder_roll_joint", _KP_5020, _KD_5020),
    "left_shoulder_yaw_joint": _drive("left_shoulder_yaw_joint", _KP_5020, _KD_5020),
    "left_elbow_joint": _drive("left_elbow_joint", _KP_5020, _KD_5020),
    "left_wrist_roll_joint": _drive("left_wrist_roll_joint", _KP_5020, _KD_5020),
    "left_wrist_pitch_joint": _drive("left_wrist_pitch_joint", _KP_4010, _KD_4010),
    "left_wrist_yaw_joint": _drive("left_wrist_yaw_joint", _KP_4010, _KD_4010),
    "right_shoulder_pitch_joint": _drive("right_shoulder_pitch_joint", _KP_5020, _KD_5020),
    "right_shoulder_roll_joint": _drive("right_shoulder_roll_joint", _KP_5020, _KD_5020),
    "right_shoulder_yaw_joint": _drive("right_shoulder_yaw_joint", _KP_5020, _KD_5020),
    "right_elbow_joint": _drive("right_elbow_joint", _KP_5020, _KD_5020),
    "right_wrist_roll_joint": _drive("right_wrist_roll_joint", _KP_5020, _KD_5020),
    "right_wrist_pitch_joint": _drive("right_wrist_pitch_joint", _KP_4010, _KD_4010),
    "right_wrist_yaw_joint": _drive("right_wrist_yaw_joint", _KP_4010, _KD_4010),
}


def build_pd_mapping(
    isaac_joint_names: Iterable[str],
    *,
    kp_scale: float = 1.0,
    kd_scale: float = 1.0,
) -> list[PDMappingRow]:
    """Build Sonic body joint PD rows in Sonic body order."""

    isaac_names = list(isaac_joint_names)
    sonic_to_isaac = build_joint_mapping(
        source_joint_names=isaac_names,
        target_joint_names=SONIC_BODY_JOINT_NAMES,
    )

    rows: list[PDMappingRow] = []
    for sonic_index, joint_name in enumerate(SONIC_BODY_JOINT_NAMES):
        mujoco = MUJOCO_PD_PARAMS[joint_name]
        drive = ISAAC_DRIVE_PARAMS[joint_name]
        rows.append(
            PDMappingRow(
                mujoco_joint=joint_name,
                isaac_joint=isaac_names[sonic_to_isaac[sonic_index]],
                sonic_index=sonic_index,
                isaac_index=sonic_to_isaac[sonic_index],
                kp=drive.stiffness * kp_scale,
                kd=drive.damping * kd_scale,
                effort_limit=drive.max_effort,
                mujoco_kp=mujoco.kp,
                mujoco_kd=mujoco.kd,
                passive_damping=mujoco.passive_damping,
                source=drive.source,
            )
        )
    return rows
