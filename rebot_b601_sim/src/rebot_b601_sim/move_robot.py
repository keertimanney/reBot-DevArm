"""CLI entry point for pose-to-pose robot motion in simulation.

See ``robot_mover.py`` for the Python API (RobotMover class).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path


def _parse_waypoint(s: str) -> tuple:
    parts = s.split(",")
    if len(parts) not in (3, 4):
        raise argparse.ArgumentTypeError(
            f"Waypoint must be  x,y,z  or  x,y,z,yaw_rad  — got: {s!r}"
        )
    try:
        return tuple(float(v) for v in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Move reBot arm in simulation keeping EEF pointing down.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  rebot-b601-move --to 0.35 0.0 0.15
  rebot-b601-move --waypoints 0.3,0.1,0.15  0.4,-0.1,0.10  0.35,0.0,0.20
  rebot-b601-move --backend mujoco --waypoints 0.3,0.1,0.15 0.4,-0.1,0.10
  rebot-b601-move --interactive
        """,
    )
    p.add_argument(
        "--backend", choices=["meshcat", "mujoco"], default="meshcat",
        help="Simulation backend (default: meshcat)",
    )
    p.add_argument(
        "--to", nargs=3, type=float, metavar=("X", "Y", "Z"),
        help="Move to a single XYZ target (metres).",
    )
    p.add_argument(
        "--yaw", type=float, default=0.0,
        help="Gripper yaw about world Z in radians (default: 0).",
    )
    p.add_argument(
        "--waypoints", nargs="+", type=_parse_waypoint, metavar="x,y,z[,yaw]",
        help="Sequence of waypoints. Each is  x,y,z  or  x,y,z,yaw_rad.",
    )
    p.add_argument(
        "--duration", type=float, default=None,
        help="Fixed move duration in seconds (auto from distance when omitted).",
    )
    p.add_argument(
        "--gripper", type=float, default=0.0,
        help="Gripper command in degrees: 0=closed, -270=fully open.",
    )
    p.add_argument(
        "--interactive", action="store_true",
        help="Interactive prompt mode (meshcat backend only).",
    )
    p.add_argument(
        "--workcell", type=Path, default=None,
        help="Path to a workcell JSON to load into the scene.",
    )
    p.add_argument(
        "--no-browser", action="store_true",
        help="Do not open the MeshCAT browser tab automatically.",
    )
    p.add_argument(
        "--speed", type=float, default=None,
        help="Linear speed in m/s for auto-duration (default: 0.10).",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()

    from .robot_mover import RobotMover, MoveParams

    params = MoveParams()
    if args.speed is not None:
        params.linear_speed = args.speed

    mover = RobotMover(
        backend=args.backend,
        open_browser=not args.no_browser,
        workcell=args.workcell,
        params=params,
    )

    if args.interactive:
        if args.backend != "meshcat":
            print("--interactive only works with --backend meshcat", file=sys.stderr)
            sys.exit(1)
        _run_interactive(mover)
        return

    if args.to:
        x, y, z = args.to
        ok = mover.move_to(x, y, z, args.yaw, args.duration, args.gripper)
        if not ok:
            sys.exit(1)

    if args.waypoints:
        mover.run_waypoints(args.waypoints, args.yaw, args.duration, args.gripper)

    if args.backend == "mujoco":
        mover.run_mujoco()


def _run_interactive(mover) -> None:
    import numpy as np
    import pinocchio as pin

    print("Interactive mode — MeshCAT viewer open. Commands:")
    print("  x y z            move to position in metres (EEF down)")
    print("  x y z yaw_deg    move with gripper yaw in degrees")
    print("  home             return to neutral configuration")
    print("  pose             print current EEF pose")
    print("  q / quit         exit\n")

    while True:
        pos, rpy = mover.current_pose()
        yaw_deg = math.degrees(rpy[2])
        prompt = f"[{pos[0]:.3f} {pos[1]:.3f} {pos[2]:.3f} yaw={yaw_deg:.1f}°] > "
        try:
            line = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not line:
            continue
        if line in ("q", "quit", "exit"):
            break
        if line == "home":
            mover.home()
            continue
        if line == "pose":
            print(
                f"  pos=[{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]  "
                f"rpy_deg={np.degrees(rpy).round(2).tolist()}"
            )
            continue

        parts = line.split()
        try:
            vals = [float(v) for v in parts]
        except ValueError:
            print("  Format: x y z [yaw_deg]")
            continue
        if len(vals) < 3:
            print("  Format: x y z [yaw_deg]")
            continue

        x, y, z = vals[0], vals[1], vals[2]
        yaw_rad = math.radians(vals[3]) if len(vals) >= 4 else 0.0
        ok = mover.move_to(x, y, z, yaw=yaw_rad)
        if ok:
            p, r = mover.current_pose()
            print(
                f"  → pos=[{p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}]  "
                f"rpy_deg={np.degrees(r).round(1).tolist()}"
            )

    mover.home()
    print("Done.")


if __name__ == "__main__":
    main()
