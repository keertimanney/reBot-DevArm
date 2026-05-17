from __future__ import annotations

import numpy as np

from rebot_b601_sim.joints import SIM_JOINT_NAMES

# Conservative first-cut observe pose, radians.
# Intent: lift the arm out of the top camera view while pointing the wrist area
# generally toward the workspace. This should be tuned once cameras are mounted.
Q_OBSERVE = np.array([0.0, -1.05, -1.35, 0.65, 0.0, 0.0], dtype=float)

JOINT_ORDER = SIM_JOINT_NAMES

JOINT_LIMITS_RAD = {
    "shoulder_pan": (-2.5307, 2.5307),
    "shoulder_lift": (-2.9671, 0.0175),
    "elbow_flex": (-3.4907, 0.0175),
    "wrist_flex": (-1.3963, 1.5708),
    "wrist_yaw": (-1.5708, 1.5708),
    "wrist_roll": (-1.5708, 1.5708),
}

GRIPPER_CLOSED_DEG = 0.0
GRIPPER_OPEN_DEG = -270.0
DEFAULT_GRIPPER_MAX_WIDTH_M = 0.08

