#!/usr/bin/env python3
"""Interactive REPL — type a position name and the arm moves there.

Supports both realtime_positions.yaml (xyz → IK) and
recorded_arm_positions.yaml (joints_rad → joint-space replay).

Usage:
    python move_repl.py
    python move_repl.py --positions-file recorded_arm_positions.yaml
    python move_repl.py --duration 3.0
"""

from __future__ import annotations

import argparse
import math
import sys
import threading
import time
from pathlib import Path

import numpy as np
import yaml

_REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_ROOT / "reBotArm_control_py"))

from reBotArm_control_py.actuator import RobotArm
from reBotArm_control_py.controllers import ArmEndPos


_DEFAULT_SPEED = 0.5   # rad/s for joint-space moves
_MIN_DURATION  = 2.0


# ── joint-space trajectory (same as replay_arm_positions.py) ──────────────────

def _min_jerk(tau: float) -> float:
    return 10 * tau**3 - 15 * tau**4 + 6 * tau**5


def _move_joints(ctrl: ArmEndPos, arm: RobotArm, q_target: np.ndarray, duration: float) -> None:
    q_start, _, _ = arm.get_state()
    n = max(2, int(duration / 0.02))

    ctrl._stop_send.set()
    if ctrl._send_thread is not None:
        ctrl._send_thread.join(timeout=1.0)
    ctrl._stop_send.clear()

    done = threading.Event()

    def _send() -> None:
        interval = duration / n
        for i in range(n):
            if ctrl._stop_send.is_set():
                break
            tau = i / (n - 1)
            s = _min_jerk(tau)
            ctrl._q_target[:] = q_start + s * (q_target - q_start)
            time.sleep(interval)
        ctrl._q_target[:] = q_target
        done.set()

    t = threading.Thread(target=_send, daemon=True)
    ctrl._send_thread = t
    t.start()
    done.wait(timeout=duration + 3.0)


def _go_to_zero(ctrl: ArmEndPos, arm: RobotArm) -> None:
    q_now, _, _ = arm.get_state()
    max_delta = float(np.max(np.abs(q_now)))
    duration = max(_MIN_DURATION, max_delta / _DEFAULT_SPEED)
    _move_joints(ctrl, arm, np.zeros(arm.num_joints), duration)


# ── config loader ──────────────────────────────────────────────────────────────

def load_positions(path: Path) -> tuple[dict, dict]:
    data = yaml.safe_load(path.read_text()) or {}
    raw = data.get("positions") or {}
    if not raw:
        raise ValueError(f"No 'positions' found in {path}")
    return raw, data.get("motion") or {}


# ── REPL ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive REPL: type a position name to move the arm there.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  <name>       move to that position (e.g. A, B, center)
  home         return to zero_act
  list         show all positions
  q / quit     home and exit
        """,
    )
    parser.add_argument(
        "--positions-file", default="realtime_positions.yaml",
        help="YAML with positions (default: realtime_positions.yaml).",
    )
    parser.add_argument(
        "--duration", type=float, default=None,
        help="Move duration in seconds (auto from joint distance when omitted).",
    )
    parser.add_argument(
        "--arm-cfg", default=None,
        help="Path to arm.yaml (auto-detected when omitted).",
    )
    args = parser.parse_args()

    positions_path = Path(args.positions_file)
    if not positions_path.is_absolute():
        positions_path = _REPO_ROOT / positions_path
    if not positions_path.exists():
        sys.exit(f"Positions file not found: {positions_path}")

    raw_positions, motion_cfg = load_positions(positions_path)
    roll    = float(motion_cfg.get("roll",  0.0))
    pitch   = float(motion_cfg.get("pitch", math.pi / 2))
    cfg_dur = float(motion_cfg.get("duration", _MIN_DURATION))

    # Build a normalised lookup: lower-case name → raw entry
    lookup: dict[str, tuple[str, dict]] = {
        name.lower(): (name, entry)
        for name, entry in raw_positions.items()
    }

    print(f"Loaded {len(lookup)} positions from {positions_path.name}:")
    for name, entry in raw_positions.items():
        if "xyz" in entry:
            xyz = entry["xyz"]
            print(f"  {name}: xyz={xyz}")
        elif "joints_rad" in entry:
            deg = [round(math.degrees(v), 1) for v in entry["joints_rad"]]
            print(f"  {name}: joints_deg={deg}")

    arm  = RobotArm(cfg_path=args.arm_cfg)
    ctrl = ArmEndPos(arm)

    print("\nConnecting and enabling motors …")
    ctrl.start()

    print("Moving to zero_act (home) …")
    _go_to_zero(ctrl, arm)
    print("At home. Ready.\n")

    try:
        while True:
            try:
                cmd = input("move> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not cmd:
                continue

            low = cmd.lower()

            if low in ("q", "quit", "exit"):
                break

            if low in ("home", "zero", "zero_act"):
                print("Going home …")
                _go_to_zero(ctrl, arm)
                print("Done.")
                continue

            if low == "list":
                for name, entry in raw_positions.items():
                    if "xyz" in entry:
                        print(f"  {name}: xyz={entry['xyz']}")
                    elif "joints_rad" in entry:
                        deg = [round(math.degrees(v), 1) for v in entry["joints_rad"]]
                        print(f"  {name}: joints_deg={deg}")
                continue

            if low not in lookup:
                print(f"Unknown position '{cmd}'. Type 'list' to see options.")
                continue

            canonical, entry = lookup[low]

            if "xyz" in entry:
                xyz = entry["xyz"]
                yaw = float(entry.get("yaw", 0.0))
                dur = args.duration or cfg_dur
                print(f"→ {canonical}  xyz={xyz}  dur={dur:.1f}s …")
                ok = ctrl.move_to_traj(
                    x=xyz[0], y=xyz[1], z=xyz[2],
                    roll=roll, pitch=pitch, yaw=yaw,
                    duration=dur,
                )
                if ok:
                    time.sleep(dur + 0.1)
                    print("  done.")
                else:
                    print(f"  IK failed for '{canonical}'.")

            elif "joints_rad" in entry:
                q_target = np.array(entry["joints_rad"], dtype=np.float64)
                q_now, _, _ = arm.get_state()
                max_delta = float(np.max(np.abs(q_target - q_now)))
                dur = args.duration or max(_MIN_DURATION, max_delta / _DEFAULT_SPEED)
                deg = [round(math.degrees(v), 1) for v in q_target]
                print(f"→ {canonical}  joints_deg={deg}  dur={dur:.1f}s …")
                _move_joints(ctrl, arm, q_target, dur)
                print("  done.")

            else:
                print(f"  '{canonical}' has neither 'xyz' nor 'joints_rad' — skipping.")

    finally:
        print("Returning home and disconnecting …")
        ctrl.end()
        print("Done.")


if __name__ == "__main__":
    main()
