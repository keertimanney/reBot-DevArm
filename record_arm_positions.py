#!/usr/bin/env python3
from __future__ import annotations

import argparse
import select
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


def read_pose(arm: Any, joint_to_pose: Any, gripper: Any = None) -> dict[str, Any]:
    q = arm.get_positions(request=True)
    vel = arm.get_velocities(request=False)
    torq = arm.get_torques(request=False)
    q = np.asarray(q, dtype=np.float64)
    vel = np.asarray(vel, dtype=np.float64)
    torq = np.asarray(torq, dtype=np.float64)
    pos, rpy = joint_to_pose(q)

    record: dict[str, Any] = {
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
    if gripper is not None:
        record["gripper_rad"] = float(gripper.get_position(request=True))
    return record


def _zero_pose_record(joint_to_pose: Any, n_joints: int) -> dict[str, Any]:
    """Compute the zero-configuration pose from FK without touching hardware."""
    q = np.zeros(n_joints)
    pos, rpy = joint_to_pose(q)
    return {
        "joints_rad": [0.0] * n_joints,
        "joints_deg": [0.0] * n_joints,
        "velocity_rad_s": [0.0] * n_joints,
        "torque": [0.0] * n_joints,
        "end_effector": {
            "xyz": [float(v) for v in pos],
            "rpy_rad": [float(v) for v in rpy],
            "rpy_deg": [float(v) for v in np.degrees(rpy)],
        },
    }


def _make_gripper_handle(arm: Any) -> Any:
    """Return a lightweight gripper handle sharing the arm's serial controller.

    Registers the gripper motor (0x07) on the arm's existing Controller so no
    second serial connection is opened.  Returns None if registration fails.
    """
    class _GripperHandle:
        def __init__(self, ctrl: Any, mot: Any) -> None:
            self._ctrl = ctrl
            self._mot  = mot

        def get_position(self, request: bool = True) -> float:
            if request:
                try:
                    self._mot.request_feedback()
                except Exception:
                    pass
                try:
                    self._ctrl.poll_feedback_once()
                except Exception:
                    pass
            try:
                st = self._mot.get_state()
                return float(st.pos) if st is not None else 0.0
            except Exception:
                return 0.0

        def disconnect(self) -> None:
            pass  # lifecycle managed by the shared arm controller

    try:
        ctrl = list(arm._ctrl_map.values())[0]
        mot  = ctrl.add_damiao_motor(0x07, 0x17, "4310")
        return _GripperHandle(ctrl, mot)
    except Exception as exc:
        print(f"[warn] Gripper motor not registered — gripper position won't be recorded. ({exc})")
        return None


def _live_capture(arm: Any, label: str, settle: float, gripper: Any = None) -> None:
    """Stream live joint angles (and gripper position) until the user presses Enter."""
    print(f"  Move arm to '{label}' — press Enter to capture.")
    while True:
        q = arm.get_positions(request=True)
        deg = [f"{v:+6.1f}" for v in np.degrees(np.asarray(q, dtype=np.float64))]
        line = f"\r  joints_deg: [{', '.join(deg)}]"
        if gripper is not None:
            g_rad = gripper.get_position(request=True)
            line += f"  gripper: {g_rad:+.4f} rad"
        line += "  "
        sys.stdout.write(line)
        sys.stdout.flush()
        if select.select([sys.stdin], [], [], 0.1)[0]:
            sys.stdin.readline()   # consume the newline
            break
    sys.stdout.write("\n")
    time.sleep(settle)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Record follower-arm joint angles and Pinocchio FK poses under labels "
            "such as A, B, C.  A 'zero' pose (all joints = 0) is saved automatically."
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
        default=6,
        help="Number of user-defined positions to record (in addition to zero).",
    )
    parser.add_argument(
        "--settle",
        type=float,
        default=0.2,
        help="Seconds to wait after pressing Enter before reading state.",
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

    # Register the gripper motor on the arm's existing controller so we don't
    # open a second connection to the same serial port.
    gripper = _make_gripper_handle(arm)
    records: dict[str, Any] = {}

    # Auto-record the zero pose from FK — no hardware read needed.
    zero = _zero_pose_record(joint_to_pose, n_joints=arm.num_joints)
    records["zero_act"] = zero
    xyz0 = zero["end_effector"]["xyz"]
    print(
        f"[auto] Recorded 'zero_act': "
        f"xyz=({xyz0[0]:+.4f}, {xyz0[1]:+.4f}, {xyz0[2]:+.4f}) m | "
        f"joints_deg={zero['joints_deg']}"
    )

    print("\nConnecting to arm (read-only — motors are not enabled or commanded).")
    print("Joint angles stream live while you position the arm.\n")

    try:
        arm.connect()
        for idx in range(args.count):
            while True:
                label = input(f"Label for position {idx + 1}/{args.count} (e.g. A): ").strip()
                if not label:
                    print("  Label cannot be empty.")
                    continue
                if label in records:
                    print(f"  Label {label!r} is already recorded.")
                    continue
                break

            _live_capture(arm, label, args.settle, gripper=gripper)
            record = read_pose(arm, joint_to_pose, gripper=gripper)
            records[label] = record

            xyz = record["end_effector"]["xyz"]
            g_info = f"  gripper={record['gripper_rad']:+.4f} rad" if "gripper_rad" in record else ""
            print(
                f"  Captured '{label}': "
                f"xyz=({xyz[0]:+.4f}, {xyz[1]:+.4f}, {xyz[2]:+.4f}) m | "
                f"joints_deg={[round(v, 2) for v in record['joints_deg']]}"
                f"{g_info}\n"
            )
    finally:
        arm.disconnect()   # closes the shared serial controller (gripper included)

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
    print(f"Saved {len(records)} position(s) to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
