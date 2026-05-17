#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def add_control_package_to_path(repo_root: Path) -> None:
    control_repo = repo_root / "reBotArm_control_py"
    sys.path.insert(0, str(control_repo))


def read_pose(arm: Any, joint_to_pose: Any) -> dict[str, Any]:
    q, vel, torq = arm.get_state()
    q = np.asarray(q, dtype=np.float64)
    vel = np.asarray(vel, dtype=np.float64)
    torq = np.asarray(torq, dtype=np.float64)
    pos, rpy = joint_to_pose(q)

    return {
        "joints_rad": [float(v) for v in q],
        "joints_deg": [float(v) for v in np.degrees(q)],
        "velocity_rad_s": [float(v) for v in vel],
        "torque": [float(v) for v in torq],
        "end_effector": {
            "xyz": [float(v) for v in pos],
            "rpy_rad": [float(v) for v in rpy],
            "rpy_deg": [float(v) for v in np.degrees(rpy)],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Record current follower-arm joint angles and Pinocchio FK poses "
            "under labels such as A, B, C."
        )
    )
    parser.add_argument(
        "--arm-config",
        default=None,
        help="Optional reBotArm_control_py arm YAML config path.",
    )
    parser.add_argument(
        "--output",
        default="recorded_arm_positions.yaml",
        help="YAML output file.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=2,
        help="Number of positions to record.",
    )
    parser.add_argument(
        "--settle",
        type=float,
        default=0.2,
        help="Seconds to wait after pressing enter before reading state.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    add_control_package_to_path(repo_root)

    try:
        from reBotArm_control_py.actuator import RobotArm
        from reBotArm_control_py.kinematics import joint_to_pose
    except ImportError as exc:
        raise SystemExit(
            "Could not import reBotArm_control_py. Activate .venv first:\n"
            "  source .venv/bin/activate"
        ) from exc

    arm = RobotArm(cfg_path=args.arm_config)
    records: dict[str, Any] = {}

    print("Connecting to follower arm for read-only position recording...")
    print("This script reads joint state and FK pose; it does not enable or move motors.")
    print("Move the arm to each desired pose manually, then press Enter to record it.")

    try:
        arm.connect()
        for idx in range(args.count):
            while True:
                label = input(f"\nLabel for position {idx + 1}/{args.count} (e.g. A): ").strip()
                if not label:
                    print("Label cannot be empty.")
                    continue
                if label in records:
                    print(f"Label {label!r} is already recorded.")
                    continue
                break

            input(f"Move the arm to position {label}, then press Enter to capture...")
            time.sleep(args.settle)
            record = read_pose(arm, joint_to_pose)
            records[label] = record

            xyz = record["end_effector"]["xyz"]
            joints_deg = record["joints_deg"]
            print(
                f"Recorded {label}: "
                f"xyz=({xyz[0]:+.4f}, {xyz[1]:+.4f}, {xyz[2]:+.4f}) m | "
                f"joints_deg={[round(v, 2) for v in joints_deg]}"
            )
    finally:
        arm.disconnect()

    output = {
        "metadata": {
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "arm_config": args.arm_config,
            "frame": "end_link",
            "units": {
                "joint_angles": "rad",
                "xyz": "m",
                "rpy": "rad",
            },
        },
        "positions": records,
    }

    output_path = Path(args.output)
    output_path.write_text(yaml.safe_dump(output, sort_keys=False), encoding="utf-8")
    print(f"\nSaved {len(records)} position(s) to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

