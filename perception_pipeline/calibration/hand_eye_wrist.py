"""Eye-in-hand calibration: solve T_tool_wrist.

Setup:
  - AprilTag fixed on the table at an arbitrary unknown pose.
  - Wrist camera mounted on the gripper.
  - Robot moves through ~20 poses viewing the tag from different angles.

For each pose i:
  - capture rgb (detect tag → T_wrist_tag_i)
  - record T_base_tool_i (from FK)

Solve:
    T_base_tool_i @ T_tool_wrist @ T_wrist_tag_i = T_base_tag   (constant)

This is the classic eye-in-hand formulation; cv2.calibrateHandEye solves it
directly with `method=cv2.CALIB_HAND_EYE_PARK` or `_DANIILIDIS`.

Usage:
    python calibration/hand_eye_wrist.py --capture
    python calibration/hand_eye_wrist.py --solve data/wrist_calib/
"""
from __future__ import annotations

import argparse
import sys

# TODO: implement, mirror structure of hand_eye_top.py.
# For eye-in-hand pass:
#   R_gripper2base = T_base_tool (yes, exactly that)
#   T_target2cam   = T_wrist_tag
# and the function returns R_cam2gripper, t_cam2gripper = T_tool_wrist.

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--capture", action="store_true")
    p.add_argument("--solve", type=str, default=None)
    p.add_argument("--tag-size", type=float, default=0.04)
    p.add_argument("--tag-family", default="tag36h11")
    p.add_argument("--out", default="configs/extrinsics.yaml")
    args = p.parse_args()
    raise NotImplementedError("Implement capture + solve. See module docstring.")


if __name__ == "__main__":
    sys.exit(main())
