"""CLI: load two images and a joint state, run pose estimation, print results.

Useful for offline debugging without the robot connected.

Usage:
    python scripts/run_estimate.py \\
        --top  data/sample/top.png \\
        --wrist data/sample/wrist.png \\
        --joints "0 -30 60 0 60 0" \\
        --query "red 2x4 brick" \\
        --candidates brick_2x4_red brick_2x4_blue brick_2x2_red
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

# Add parent to path so this script works without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pose_pipeline import PoseEstimator                 # noqa: E402
from pose_pipeline.frames import load_extrinsics, make_fk  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--top", required=True)
    p.add_argument("--wrist", required=True)
    p.add_argument("--joints", required=True, help="space-separated joint angles in degrees")
    p.add_argument("--query", required=True)
    p.add_argument("--candidates", nargs="+", required=True)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--extrinsics", default="configs/extrinsics.yaml")
    p.add_argument("--manifest", default="cad_library/manifest.yaml")
    p.add_argument("--urdf", default=None, help="URDF for Pinocchio FK")
    p.add_argument("--tool-link", default="gripper_tool")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))
    ext = load_extrinsics(args.extrinsics)

    estimator = PoseEstimator(
        cad_library_path=args.manifest,
        T_base_top=ext["T_base_top"],
        T_tool_wrist=ext["T_tool_wrist"],
        K_top=ext["K_top"],
        K_wrist=ext["K_wrist"],
        config=cfg,
    )

    rgb_top = cv2.cvtColor(cv2.imread(args.top), cv2.COLOR_BGR2RGB)
    rgb_wrist = cv2.cvtColor(cv2.imread(args.wrist), cv2.COLOR_BGR2RGB)
    q_deg = np.array([float(x) for x in args.joints.split()])
    q_rad = np.deg2rad(q_deg)

    if args.urdf:
        fk = make_fk(args.urdf, tool_link=args.tool_link)
        T_base_tool = fk(q_rad)
    else:
        print("WARN: no --urdf passed; using identity for T_base_tool. Results will be wrong.")
        T_base_tool = np.eye(4)

    dets = estimator.estimate(
        rgb_top=rgb_top,
        rgb_wrist=rgb_wrist,
        T_base_tool=T_base_tool,
        query=args.query,
        candidate_cad_ids=args.candidates,
    )

    if not dets:
        print("No detections.")
        return 1
    for d in dets:
        print(d)
        print(d.T_base_obj)
    return 0


if __name__ == "__main__":
    sys.exit(main())
