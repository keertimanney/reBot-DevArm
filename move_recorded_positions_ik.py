#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class RecordedPose:
    label: str
    xyz: list[float]
    rpy_rad: list[float]


def add_control_package_to_path(repo_root: Path) -> None:
    sys.path.insert(0, str(repo_root / "reBotArm_control_py"))


def load_recorded_poses(path: Path) -> dict[str, RecordedPose]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw_positions = data.get("positions") or {}
    poses: dict[str, RecordedPose] = {}

    for label, record in raw_positions.items():
        ee = (record or {}).get("end_effector") or {}
        xyz = ee.get("xyz")
        rpy = ee.get("rpy_rad")
        if not isinstance(xyz, list) or len(xyz) != 3:
            raise ValueError(f"Recorded position {label!r} is missing end_effector.xyz")
        if not isinstance(rpy, list) or len(rpy) != 3:
            raise ValueError(f"Recorded position {label!r} is missing end_effector.rpy_rad")
        poses[str(label)] = RecordedPose(
            label=str(label),
            xyz=[float(v) for v in xyz],
            rpy_rad=[float(v) for v in rpy],
        )

    if not poses:
        raise ValueError(f"No recorded poses found in {path}")
    return poses


def wait_until_done(controller: Any, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while getattr(controller, "_moving", False):
        if time.monotonic() > deadline:
            raise TimeoutError("Timed out waiting for IK trajectory to finish")
        time.sleep(0.05)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Move through recorded end-effector poses using ArmEndPos IK."
    )
    parser.add_argument(
        "--recorded",
        default="recorded_arm_positions.yaml",
        help="YAML file created by record_arm_positions.py.",
    )
    parser.add_argument(
        "--arm-config",
        default=None,
        help="Optional reBotArm_control_py arm YAML config path.",
    )
    parser.add_argument(
        "--sequence",
        nargs="+",
        default=["zero", "B", "A"],
        help="Labels to move through. Use 'zero' for the neutral/home pose.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=2.0,
        help="Seconds per IK trajectory.",
    )
    parser.add_argument(
        "--no-step",
        action="store_true",
        help="Run the sequence without waiting for Enter before each target.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    add_control_package_to_path(repo_root)

    try:
        from reBotArm_control_py.actuator import RobotArm
        from reBotArm_control_py.controllers import ArmEndPos
        from reBotArm_control_py.kinematics import joint_to_pose
    except ImportError as exc:
        raise SystemExit(
            "Could not import reBotArm_control_py. Activate .venv first:\n"
            "  source .venv/bin/activate"
        ) from exc

    poses = load_recorded_poses(Path(args.recorded))
    zero_xyz, zero_rpy = joint_to_pose(__import__("numpy").zeros(6))
    poses["zero"] = RecordedPose(
        label="zero",
        xyz=[float(v) for v in zero_xyz],
        rpy_rad=[float(v) for v in zero_rpy],
    )

    missing = [label for label in args.sequence if label not in poses]
    if missing:
        raise SystemExit(f"Missing recorded pose(s): {missing}. Available: {sorted(poses)}")

    print("IK sequence:", " -> ".join(args.sequence))
    print("This script uses recorded end-effector XYZ/RPY poses and solves IK for each move.")
    print("Keep the workspace clear. Press Ctrl+C to abort.")

    arm = RobotArm(cfg_path=args.arm_config)
    controller = ArmEndPos(arm)

    try:
        controller.start()
        for label in args.sequence:
            pose = poses[label]
            x, y, z = pose.xyz
            roll, pitch, yaw = pose.rpy_rad
            print(
                f"\nTarget {label}: "
                f"xyz=({x:+.4f}, {y:+.4f}, {z:+.4f}) "
                f"rpy=({roll:+.4f}, {pitch:+.4f}, {yaw:+.4f})"
            )
            if not args.no_step:
                input("Press Enter to move to this target...")

            ok = controller.move_to_traj(
                x=x,
                y=y,
                z=z,
                roll=roll,
                pitch=pitch,
                yaw=yaw,
                duration=args.duration,
            )
            print("IK accepted" if ok else "IK failed")
            if not ok:
                return 1
            wait_until_done(controller, timeout=max(10.0, args.duration + 5.0))
    finally:
        controller.end()

    print("\nSequence complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

