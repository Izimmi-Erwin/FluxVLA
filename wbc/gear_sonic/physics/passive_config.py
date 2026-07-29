"""Passive dynamics calibration helpers for Isaac G1 balance tests.

The values mirror the MuJoCo XML defaults as runtime Isaac authoring. They do
not modify source USD assets, MuJoCo XML, DDS messages, or WBC code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from gear_sonic.robot_interface.joint_mapping import SONIC_BODY_JOINT_NAMES, build_joint_mapping


@dataclass(frozen=True)
class PassiveJointParam:
    joint: str
    passive_damping: float
    armature: float
    frictionloss: float
    source: str


@dataclass(frozen=True)
class PassiveDynamicsConfig:
    name: str
    damping_scale: float = 1.0
    armature_scale: float = 1.0
    friction_scale: float = 1.0
    apply_drive_damping_offset: bool = True
    apply_joint_armature: bool = True
    apply_joint_friction: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "damping_scale": self.damping_scale,
            "armature_scale": self.armature_scale,
            "friction_scale": self.friction_scale,
            "apply_drive_damping_offset": self.apply_drive_damping_offset,
            "apply_joint_armature": self.apply_joint_armature,
            "apply_joint_friction": self.apply_joint_friction,
        }


@dataclass(frozen=True)
class PassiveDynamicsMappingRow:
    sonic_joint: str
    isaac_joint: str
    sonic_index: int
    isaac_index: int
    damping_offset: float
    armature: float
    joint_friction: float
    source: str


def _frictionloss_for_joint(joint: str) -> float:
    if "wrist" in joint or "finger" in joint:
        return 0.1
    return 0.2


MUJOCO_PASSIVE_PARAMS: dict[str, PassiveJointParam] = {
    joint: PassiveJointParam(
        joint=joint,
        passive_damping=0.05,
        armature=0.01,
        frictionloss=_frictionloss_for_joint(joint),
        source="g1_29dof_with_hand.xml default joint damping/armature/frictionloss",
    )
    for joint in SONIC_BODY_JOINT_NAMES
}


DISABLED_PASSIVE_CONFIG = PassiveDynamicsConfig(
    name="disabled",
    damping_scale=0.0,
    armature_scale=0.0,
    friction_scale=0.0,
    apply_drive_damping_offset=False,
    apply_joint_armature=False,
    apply_joint_friction=False,
)


MUJOCO_PASSIVE_CONFIG = PassiveDynamicsConfig(name="mujoco_passive")


PASSIVE_PRESETS: dict[str, PassiveDynamicsConfig] = {
    "disabled": DISABLED_PASSIVE_CONFIG,
    "mujoco": MUJOCO_PASSIVE_CONFIG,
}


def build_passive_config(
    *,
    preset: str = "disabled",
    damping_scale: float | None = None,
    armature_scale: float | None = None,
    friction_scale: float | None = None,
    apply_drive_damping_offset: bool | None = None,
    apply_joint_armature: bool | None = None,
    apply_joint_friction: bool | None = None,
) -> PassiveDynamicsConfig:
    if preset not in PASSIVE_PRESETS:
        raise ValueError(f"unknown passive preset {preset!r}; options={sorted(PASSIVE_PRESETS)}")
    base = PASSIVE_PRESETS[preset]
    return PassiveDynamicsConfig(
        name=base.name,
        damping_scale=base.damping_scale if damping_scale is None else float(damping_scale),
        armature_scale=base.armature_scale if armature_scale is None else float(armature_scale),
        friction_scale=base.friction_scale if friction_scale is None else float(friction_scale),
        apply_drive_damping_offset=(
            base.apply_drive_damping_offset
            if apply_drive_damping_offset is None
            else bool(apply_drive_damping_offset)
        ),
        apply_joint_armature=(
            base.apply_joint_armature if apply_joint_armature is None else bool(apply_joint_armature)
        ),
        apply_joint_friction=(
            base.apply_joint_friction if apply_joint_friction is None else bool(apply_joint_friction)
        ),
    )


def build_passive_mapping(
    isaac_joint_names: Iterable[str],
    *,
    config: PassiveDynamicsConfig | None = None,
) -> list[PassiveDynamicsMappingRow]:
    isaac_names = list(isaac_joint_names)
    sonic_to_isaac = build_joint_mapping(
        source_joint_names=isaac_names,
        target_joint_names=SONIC_BODY_JOINT_NAMES,
    )
    cfg = config or MUJOCO_PASSIVE_CONFIG
    rows: list[PassiveDynamicsMappingRow] = []
    for sonic_index, joint_name in enumerate(SONIC_BODY_JOINT_NAMES):
        param = MUJOCO_PASSIVE_PARAMS[joint_name]
        isaac_index = sonic_to_isaac[sonic_index]
        rows.append(
            PassiveDynamicsMappingRow(
                sonic_joint=joint_name,
                isaac_joint=isaac_names[isaac_index],
                sonic_index=sonic_index,
                isaac_index=isaac_index,
                damping_offset=param.passive_damping * cfg.damping_scale,
                armature=param.armature * cfg.armature_scale,
                joint_friction=param.frictionloss * cfg.friction_scale,
                source=param.source,
            )
        )
    return rows
