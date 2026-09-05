"""Sanity check the extrinsics.

Place an AprilTag at a known, measured position on the table. Move the arm to
the observe pose. Run this script — it estimates the tag pose via both cameras
and compares to ground truth in base frame.

A 5mm extrinsic error masquerades as a 5mm FoundationPose error. Validate
before trusting downstream pose estimates.

Usage:
    python calibration/validate_extrinsics.py \\
        --ground-truth-xyz 0.30 0.10 0.000 \\
        --observe-pose configs/default.yaml
"""
from __future__ import annotations

import argparse
import sys

# TODO: implement.
#   1. load extrinsics
#   2. command robot to observe pose, capture both images
#   3. detect AprilTag in each → T_top_tag, T_wrist_tag
#   4. lift to base frame:
#       T_base_tag_top   = T_base_top @ T_top_tag
#       T_base_tag_wrist = T_base_tool @ T_tool_wrist @ T_wrist_tag
#   5. print:
#       - position error from each view vs ground truth
#       - rotation error from each view vs ground truth (axis-angle magnitude)
#       - disagreement between top and wrist views
#   6. PASS / FAIL with thresholds (3mm, 1deg by default)

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ground-truth-xyz", nargs=3, type=float, required=True)
    p.add_argument("--ground-truth-rpy", nargs=3, type=float, default=[0, 0, 0])
    p.add_argument("--observe-pose", default="configs/default.yaml")
    p.add_argument("--extrinsics", default="configs/extrinsics.yaml")
    p.add_argument("--pos-tol-m", type=float, default=0.003)
    p.add_argument("--rot-tol-deg", type=float, default=1.0)
    args = p.parse_args()
    raise NotImplementedError("Implement. See module docstring.")


if __name__ == "__main__":
    sys.exit(main())
