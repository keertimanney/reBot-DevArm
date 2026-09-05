"""Eye-to-hand calibration: solve T_base_top.

Setup:
  - Fixed top camera, looking down at the workspace.
  - An AprilTag (or ArUco marker) rigidly attached to the gripper at a known
    transform T_tool_tag.
  - Robot moves through N (~20) joint configurations covering varied positions
    and orientations within the top camera's FoV.

For each pose i, we capture:
  - rgb image
  - T_base_tool_i  (from FK)

We detect the tag in each image to get T_top_tag_i, then solve:
    T_base_top @ T_top_tag_i = T_base_tool_i @ T_tool_tag
=> T_base_top = T_base_tool_i @ T_tool_tag @ inv(T_top_tag_i)

This is AX=ZB in standard form; use cv2.calibrateHandEye(method=DANIILIDIS).

Usage:
    python calibration/hand_eye_top.py --capture            # interactive capture
    python calibration/hand_eye_top.py --solve data/top_calib/
"""
from __future__ import annotations

import argparse
import sys

# TODO: implement.
# Two subcommands:
#   --capture: open camera, for each robot pose key-press save image + current
#              T_base_tool (you supply T_base_tool via a small client to your
#              motion stack, or read from a log)
#   --solve:   load saved (image, T_base_tool) pairs, detect AprilTag, run
#              cv2.calibrateHandEye(R_gripper2base, t_gripper2base,
#                                   R_target2cam,  t_target2cam,
#                                   method=cv2.CALIB_HAND_EYE_DANIILIDIS)
#              write T_base_top to configs/extrinsics.yaml
#
# Important: cv2.calibrateHandEye has confusing argument naming. For eye-to-hand
# (camera fixed, target moves with gripper), pass:
#   R_gripper2base = inv(T_base_tool)
#   T_target2cam   = T_cam_tag
# Read the OpenCV docs carefully before trusting any output.

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--capture", action="store_true")
    p.add_argument("--solve", type=str, default=None, help="path to captured data dir")
    p.add_argument("--tag-size", type=float, default=0.04, help="AprilTag edge length, meters")
    p.add_argument("--tag-family", default="tag36h11")
    p.add_argument("--out", default="configs/extrinsics.yaml")
    args = p.parse_args()
    raise NotImplementedError("Implement capture + solve. See module docstring.")


if __name__ == "__main__":
    sys.exit(main())
