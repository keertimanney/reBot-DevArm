#!/usr/bin/env python3
"""Replay positions recorded by record_arm_positions.py on the real robot.

Uses joint angles directly (not IK from xyz) for faithful, reliable replay.

Usage:
    python replay_arm_positions.py                       # replay A then B
    python replay_arm_positions.py --positions B A B    # custom sequence
    python replay_arm_positions.py --file my.yaml --duration 4.0
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent / "reBotArm_control_py"))

from reBotArm_control_py.actuator import RobotArm
from reBotArm_control_py.controllers import ArmEndPos


_DEFAULT_SPEED = 0.5   # rad/s, used to auto-compute duration from max joint delta
_MIN_DURATION  = 2.0   # seconds minimum per move
_DT            = 0.02  # trajectory sample period (s)


def _min_jerk(tau: float) -> float:
    """Min-jerk scalar profile: smooth S-curve, zero vel/acc at endpoints."""
    return 10*tau**3 - 15*tau**4 + 6*tau**5


def _move_joints(
    controller: ArmEndPos,
    arm: RobotArm,
    q_target: np.ndarray,
    duration: float,
) -> None:
    """Drive arm to q_target via min-jerk joint interpolation.

    Modifies controller._q_target on a background thread at _DT intervals.
    The existing 500Hz _loop_cb picks up each update and sends pos_vel to motors.
    Blocks until the trajectory finishes.
    """
    q_start, _, _ = arm.get_state()
    n = max(2, int(duration / _DT))

    # Stop any in-progress trajectory from move_to_traj
    controller._stop_send.set()
    if controller._send_thread is not None:
        controller._send_thread.join(timeout=1.0)
    controller._stop_send.clear()

    done = threading.Event()

    def _send():
        interval = duration / n
        for i in range(n):
            tau = i / (n - 1)
            s = _min_jerk(tau)
            controller._q_target[:] = q_start + s * (q_target - q_start)
            time.sleep(interval)
        controller._q_target[:] = q_target
        done.set()

    t = threading.Thread(target=_send, daemon=True)
    t.start()
    done.wait(timeout=duration + 2.0)


def _auto_duration(q_from: np.ndarray, q_to: np.ndarray, speed: float) -> float:
    max_delta = float(np.max(np.abs(q_to - q_from)))
    return max(_MIN_DURATION, max_delta / speed)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay recorded arm positions on the real robot using joint angles.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python replay_arm_positions.py
  python replay_arm_positions.py --positions B A B
  python replay_arm_positions.py --file my_positions.yaml --duration 4.0
  python replay_arm_positions.py --speed 0.3
        """,
    )
    parser.add_argument(
        "--file", default="recorded_arm_positions.yaml",
        help="YAML file produced by record_arm_positions.py.",
    )
    parser.add_argument(
        "--positions", nargs="+", default=None, metavar="LABEL",
        help="Labels to replay in order (default: all in file order).",
    )
    parser.add_argument(
        "--duration", type=float, default=None,
        help="Fixed move duration in seconds (auto from joint distance when omitted).",
    )
    parser.add_argument(
        "--speed", type=float, default=_DEFAULT_SPEED,
        help=f"Max joint speed rad/s for auto-duration (default: {_DEFAULT_SPEED}).",
    )
    parser.add_argument(
        "--cfg", type=str, default=None,
        help="Path to arm.yaml (auto-detected when omitted).",
    )
    args = parser.parse_args()

    data = yaml.safe_load(Path(args.file).read_text())
    stored = data["positions"]

    labels = args.positions if args.positions else list(stored.keys())
    for lbl in labels:
        if lbl not in stored:
            sys.exit(
                f"ERROR: {lbl!r} not in {args.file}. "
                f"Available: {list(stored.keys())}"
            )

    arm = RobotArm(cfg_path=args.cfg)
    controller = ArmEndPos(arm)

    print("Connecting and enabling motors …")
    controller.start()
    print(f"Ready. Replaying: {labels}\n")

    try:
        q_now, _, _ = arm.get_state()

        for i, lbl in enumerate(labels):
            rec = stored[lbl]
            q_target = np.array(rec["joints_rad"], dtype=np.float64)
            xyz = rec["end_effector"]["xyz"]

            dur = (
                args.duration
                if args.duration is not None
                else _auto_duration(q_now, q_target, args.speed)
            )

            print(
                f"[{i+1}/{len(labels)}] → {lbl!r}  "
                f"joints_deg={[round(v, 1) for v in np.degrees(q_target).tolist()]}  "
                f"xyz=({xyz[0]:+.3f}, {xyz[1]:+.3f}, {xyz[2]:+.3f})  "
                f"dur={dur:.2f}s"
            )

            _move_joints(controller, arm, q_target, dur)
            q_now, _, _ = arm.get_state()
            print(f"  done.")

    finally:
        print("\nReturning to home …")
        controller.end()
        print("Done.")


if __name__ == "__main__":
    main()
