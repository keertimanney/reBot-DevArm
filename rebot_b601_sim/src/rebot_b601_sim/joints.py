from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

JOINT_NAMES: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_yaw",
    "wrist_roll",
    "gripper",
)

SIM_JOINT_NAMES: tuple[str, ...] = JOINT_NAMES[:6]


def strip_pos_suffix(action: Mapping[str, float]) -> dict[str, float]:
    """Normalize LeRobot-style action keys into plain joint names."""
    out: dict[str, float] = {}
    for key, value in action.items():
        if key.endswith(".pos"):
            out[key.removesuffix(".pos")] = float(value)
        else:
            out[key] = float(value)
    return out


def degrees_action_to_q(
    action: Mapping[str, float],
    neutral_q: Sequence[float],
    joint_order: Sequence[str] = SIM_JOINT_NAMES,
) -> list[float]:
    """Convert degree joint commands to a Pinocchio q vector by joint order."""
    positions = strip_pos_suffix(action)
    q = list(neutral_q)
    for idx, joint_name in enumerate(joint_order):
        if idx >= len(q):
            break
        if joint_name in positions:
            q[idx] = math.radians(positions[joint_name])
    return q


def zero_action(joint_order: Sequence[str] = JOINT_NAMES) -> dict[str, float]:
    return {f"{joint}.pos": 0.0 for joint in joint_order}

