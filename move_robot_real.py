#!/usr/bin/env python3
"""Move the real reBot arm through waypoints using the hardware controller.

Same waypoint format as the sim script:

    python move_robot_real.py --waypoints 0.35,0.0,0.15  0.40,0.10,0.10
    python move_robot_real.py --waypoints 0.35,0.0,0.15  0.40,0.10,0.10 0.35,0.0,0.15 0.40,0.10,0.10

Each waypoint is  x,y,z  or  x,y,z,yaw_rad  (metres / radians).
EEF is kept pointing straight down (pitch = π/2) throughout.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

# Make reBotArm_control_py importable from the repo layout.
sys.path.insert(0, str(Path(__file__).resolve().parent / "reBotArm_control_py"))

from reBotArm_control_py.actuator import RobotArm
from reBotArm_control_py.controllers import ArmEndPos


_DEFAULT_SPEED = 0.10   # m/s, same default as sim
_MIN_DURATION  = 1.0    # seconds


def _parse_waypoint(s: str) -> tuple:
    parts = s.split(",")
    if len(parts) not in (3, 4):
        raise argparse.ArgumentTypeError(
            f"Waypoint must be x,y,z or x,y,z,yaw_rad — got: {s!r}"
        )
    try:
        return tuple(float(v) for v in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Move the real reBot arm through waypoints (EEF pointing down).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python move_robot_real.py --waypoints 0.35,0.0,0.15  0.40,0.10,0.10
  python move_robot_real.py --waypoints 0.35,0.0,0.15 --duration 3.0
  python move_robot_real.py --waypoints 0.35,0.0,0.15 --speed 0.05
        """,
    )
    p.add_argument(
        "--waypoints", nargs="+", type=_parse_waypoint, metavar="x,y,z[,yaw]",
        required=True,
        help="Sequence of waypoints. Each is x,y,z or x,y,z,yaw_rad.",
    )
    p.add_argument(
        "--duration", type=float, default=None,
        help="Fixed move duration in seconds per waypoint (auto from distance when omitted).",
    )
    p.add_argument(
        "--speed", type=float, default=_DEFAULT_SPEED,
        help=f"Linear speed in m/s for auto-duration (default: {_DEFAULT_SPEED}).",
    )
    p.add_argument(
        "--cfg", type=str, default=None,
        help="Path to arm.yaml config (auto-detected when omitted).",
    )
    return p


def _auto_duration(speed: float, from_xyz, to_xyz) -> float:
    dist = math.sqrt(sum((b - a) ** 2 for a, b in zip(from_xyz, to_xyz)))
    return max(_MIN_DURATION, dist / speed)


def main() -> None:
    args = _build_parser().parse_args()

    cfg_path = args.cfg  # None → RobotArm uses its own default

    arm = RobotArm(cfg_path=cfg_path)
    controller = ArmEndPos(arm)

    print("Connecting to robot and enabling motors …")
    controller.start()
    print("Ready.\n")

    # Track current position so we can estimate duration between waypoints.
    # Start from the home position (all-zero joint angles → EEF at its neutral pose).
    # We read actual position after enable.
    try:
        q_now, _, _ = arm.get_state()
    except Exception:
        q_now = None

    # Use a rough starting xyz of (0, 0, 0) for the first duration estimate;
    # the controller computes FK internally, but we only need distance for timing.
    current_xyz = (0.0, 0.0, 0.0)

    try:
        for i, wp in enumerate(args.waypoints):
            x, y, z = float(wp[0]), float(wp[1]), float(wp[2])
            yaw = float(wp[3]) if len(wp) >= 4 else 0.0

            if args.duration is not None:
                dur = args.duration
            else:
                dur = _auto_duration(args.speed, current_xyz, (x, y, z))

            print(f"Waypoint {i+1}/{len(args.waypoints)}: "
                  f"({x:+.3f}, {y:+.3f}, {z:+.3f})  yaw={math.degrees(yaw):.1f}°  "
                  f"duration={dur:.2f}s …")

            ok = controller.move_to_traj(
                x=x, y=y, z=z,
                roll=0.0, pitch=math.pi / 2, yaw=yaw,
                duration=dur,
            )

            if not ok:
                print(f"  IK/planning failed — skipping waypoint {i+1}.", file=sys.stderr)
            else:
                # Wait for the trajectory to finish before commanding the next waypoint.
                time.sleep(dur + 0.1)
                print("  done.")

            current_xyz = (x, y, z)

    finally:
        print("\nReturning to home position …")
        controller.end()
        print("Done.")


if __name__ == "__main__":
    main()
