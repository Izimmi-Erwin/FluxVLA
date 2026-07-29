"""Joint order contracts for Sonic G1 simulator migration.

This module freezes the names and index mappings that Isaac Sim must reproduce.
It intentionally contains no simulator, DDS, or Isaac imports.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable


SONIC_BODY_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

LEFT_HAND_JOINT_NAMES = (
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
)

RIGHT_HAND_JOINT_NAMES = (
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
)

SONIC_43_JOINT_NAMES = (
    *SONIC_BODY_JOINT_NAMES,
    *LEFT_HAND_JOINT_NAMES,
    *RIGHT_HAND_JOINT_NAMES,
)

# Confirmed from GR00T C++ policy_parameters.hpp.
# Meaning: values_in_mujoco_order = [values_in_isaaclab_order[i] for i in ISAACLAB_TO_MUJOCO_29]
ISAACLAB_TO_MUJOCO_29 = (
    0,
    3,
    6,
    9,
    13,
    17,
    1,
    4,
    7,
    10,
    14,
    18,
    2,
    5,
    8,
    11,
    15,
    19,
    21,
    23,
    25,
    27,
    12,
    16,
    20,
    22,
    24,
    26,
    28,
)

# Confirmed inverse mapping from GR00T C++ policy_parameters.hpp.
# Meaning: values_in_isaaclab_order = [values_in_mujoco_order[i] for i in MUJOCO_TO_ISAACLAB_29]
MUJOCO_TO_ISAACLAB_29 = (
    0,
    6,
    12,
    1,
    7,
    13,
    2,
    8,
    14,
    3,
    9,
    15,
    22,
    4,
    10,
    16,
    23,
    5,
    11,
    17,
    24,
    18,
    25,
    19,
    26,
    20,
    27,
    21,
    28,
)


def _duplicates(names: list[str]) -> list[str]:
    counts = Counter(names)
    return sorted(name for name, count in counts.items() if count > 1)


def build_joint_mapping(
    source_joint_names: Iterable[str],
    target_joint_names: Iterable[str],
) -> list[int]:
    """Build source indices needed to produce target order.

    The returned mapping is suitable for:

        target_values = [source_values[i] for i in mapping]

    Both lists must contain the same unique joint names. Errors are explicit
    because a wrong joint order can make the robot unstable.
    """

    source = list(source_joint_names)
    target = list(target_joint_names)

    if len(source) != len(target):
        raise ValueError(f"joint count mismatch: source={len(source)} target={len(target)}")

    duplicate_source = _duplicates(source)
    if duplicate_source:
        raise ValueError(f"duplicate source joint names: {duplicate_source}")

    duplicate_target = _duplicates(target)
    if duplicate_target:
        raise ValueError(f"duplicate target joint names: {duplicate_target}")

    source_index = {name: idx for idx, name in enumerate(source)}
    missing = [name for name in target if name not in source_index]
    if missing:
        raise ValueError(f"missing source joints required by target: {missing}")

    extra = [name for name in source if name not in set(target)]
    if extra:
        raise ValueError(f"source joints not present in target: {extra}")

    return [source_index[name] for name in target]


def validate_contracts() -> None:
    """Validate static Sonic joint contract sizes and uniqueness."""

    if len(SONIC_BODY_JOINT_NAMES) != 29:
        raise ValueError(f"SONIC_BODY_JOINT_NAMES must have 29 joints, got {len(SONIC_BODY_JOINT_NAMES)}")
    if len(LEFT_HAND_JOINT_NAMES) != 7:
        raise ValueError(f"LEFT_HAND_JOINT_NAMES must have 7 joints, got {len(LEFT_HAND_JOINT_NAMES)}")
    if len(RIGHT_HAND_JOINT_NAMES) != 7:
        raise ValueError(f"RIGHT_HAND_JOINT_NAMES must have 7 joints, got {len(RIGHT_HAND_JOINT_NAMES)}")
    if len(SONIC_43_JOINT_NAMES) != 43:
        raise ValueError(f"SONIC_43_JOINT_NAMES must have 43 joints, got {len(SONIC_43_JOINT_NAMES)}")

    duplicates = _duplicates(list(SONIC_43_JOINT_NAMES))
    if duplicates:
        raise ValueError(f"duplicate Sonic 43 joint names: {duplicates}")

    if len(ISAACLAB_TO_MUJOCO_29) != 29:
        raise ValueError("ISAACLAB_TO_MUJOCO_29 must have 29 entries")
    if len(MUJOCO_TO_ISAACLAB_29) != 29:
        raise ValueError("MUJOCO_TO_ISAACLAB_29 must have 29 entries")

    for isaac_idx, mujoco_idx in enumerate(ISAACLAB_TO_MUJOCO_29):
        if MUJOCO_TO_ISAACLAB_29[mujoco_idx] != isaac_idx:
            raise ValueError(
                "body mapping is not reversible: "
                f"isaac_idx={isaac_idx} mujoco_idx={mujoco_idx}"
            )


validate_contracts()
