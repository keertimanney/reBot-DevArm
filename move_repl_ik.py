#!/usr/bin/env python3
"""Interactive REPL — moves to positions using Cartesian IK.

Loads recorded_arm_positions.yaml. On startup it drives to zero_act using
joint-space replay (reliable). All subsequent moves use the recorded
end_effector xyz + rpy solved through ArmEndPos.move_to_traj (IK).

Commands:
  <name>            move to a named position via IK  (e.g. A, B, center)
  rel dx dy dz      relative Cartesian move from current EEF (metres)
  open              move gripper to _OPEN_GRIPPER_POSE
  close             move gripper to _CLOSE_GRIPPER_POSE
  gripper <rad>     move gripper to an exact motor angle in radians
  home              return to zero_act via joint-space replay
  list              show all positions and their Cartesian coords
  pose              print current EEF xyz and rpy
  q / quit          home and exit

Usage:
    python move_repl_ik.py
    python move_repl_ik.py --positions-file recorded_arm_positions.yaml
    python move_repl_ik.py --duration 3.0
"""

from __future__ import annotations

import argparse
import math
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

_REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_ROOT / "reBotArm_control_py"))

from reBotArm_control_py.actuator import RobotArm
from reBotArm_control_py.controllers import ArmEndPos
from reBotArm_control_py.kinematics import joint_to_pose


# ── gripper motor angles (radians) — edit to match your physical gripper ──────
# These are position targets sent directly to the gripper motor via pos_vel.

_OPEN_GRIPPER_POSE  = 4.0    # rad — fully open
_CLOSE_GRIPPER_POSE = 1.0    # rad — fully closed

# ── motion constants ───────────────────────────────────────────────────────────

_DEFAULT_SPEED = 0.5   # rad/s for joint-space home moves
_MIN_DURATION  = 2.0   # seconds minimum per IK move
_HOME_LABEL    = "zero_act"


# ── gripper handle (shares the arm's serial controller) ───────────────────────

def _make_gripper_handle(arm: Any) -> Any:
    """Register the gripper motor on the arm's existing controller.

    Runs a dedicated 100 Hz MIT-mode loop so the motor actively holds its
    commanded position — a one-shot send is unreliable without this.
    Returns None if registration fails.
    """
    class _GripperHandle:
        _RATE = 100.0

        def __init__(self, ctrl: Any, mot: Any) -> None:
            self._ctrl    = ctrl
            self._mot     = mot
            self._target  : float | None = None
            self._lock    = threading.Lock()
            self._running = False
            self._thread  : threading.Thread | None = None

        def start(self) -> None:
            from motorbridge import Mode
            try:
                self._mot.clear_error()
            except Exception as e:
                print(f"[gripper] clear_error: {e}")
            time.sleep(0.2)
            try:
                self._ctrl.enable_all()
            except Exception as e:
                print(f"[gripper] enable: {e}")
            time.sleep(0.5)
            for attempt in range(10):
                try:
                    self._mot.ensure_mode(Mode.FORCE_POS, 2000)
                    print(f"[gripper] FORCE_POS mode OK (attempt {attempt+1})")
                    break
                except Exception as e:
                    print(f"[gripper] FORCE_POS attempt {attempt+1}: {e}")
                    time.sleep(0.05)
            time.sleep(0.2)

            # Seed target from current position — no sudden jerk on start.
            try:
                self._mot.request_feedback()
                self._ctrl.poll_feedback_once()
                st = self._mot.get_state()
                self._target = float(st.pos) if st is not None else 0.0
            except Exception:
                self._target = 0.0
            print(f"[gripper] ready — current={self._target:.4f} rad")

            self._running = True
            self._thread  = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

        def _loop(self) -> None:
            dt = 1.0 / self._RATE
            while self._running:
                t0 = time.perf_counter()
                with self._lock:
                    target = self._target
                if target is not None:
                    try:
                        # FORCE_POS: pos_rad, vlim_rad_s, torque_ratio
                        self._mot.send_force_pos(target, 2.0, 0.1)
                        self._mot.request_feedback()
                        self._ctrl.poll_feedback_once()
                    except Exception:
                        pass
                elapsed = time.perf_counter() - t0
                sleep = dt - elapsed
                if sleep > 0:
                    time.sleep(sleep)

        def move_to(self, pos: float) -> None:
            with self._lock:
                self._target = pos

        def pos_vel(self, pos: float) -> None:
            self.move_to(pos)

        def get_position(self, request: bool = True) -> float:
            if request:
                try: self._mot.request_feedback()
                except Exception: pass
                try: self._ctrl.poll_feedback_once()
                except Exception: pass
            try:
                st = self._mot.get_state()
                return float(st.pos) if st is not None else 0.0
            except Exception:
                return 0.0

        def disconnect(self) -> None:
            self._running = False
            if self._thread is not None:
                self._thread.join(timeout=1.0)

    try:
        ctrl = list(arm._ctrl_map.values())[0]
        mot  = ctrl.add_damiao_motor(0x07, 0x17, "4310")
        return _GripperHandle(ctrl, mot)
    except Exception as exc:
        print(f"[warn] Gripper not available — gripper commands will be skipped. ({exc})")
        return None


# ── joint-space trajectory (reliable, no IK) ──────────────────────────────────

def _min_jerk(tau: float) -> float:
    return 10 * tau**3 - 15 * tau**4 + 6 * tau**5


def _move_joints(ctrl: ArmEndPos, arm: RobotArm, q_target: np.ndarray, duration: float) -> None:
    """Min-jerk joint-space trajectory. Blocks until done."""
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


# ── IK move helpers ────────────────────────────────────────────────────────────

def _ik_move(
    ctrl: ArmEndPos,
    x: float, y: float, z: float,
    roll: float, pitch: float, yaw: float,
    duration: float,
    label: str = "",
) -> bool:
    tag = f" [{label}]" if label else ""
    print(
        f"→{tag}  xyz=[{x:+.4f}, {y:+.4f}, {z:+.4f}]  "
        f"rpy_deg=[{math.degrees(roll):+.1f}, {math.degrees(pitch):+.1f}, "
        f"{math.degrees(yaw):+.1f}]  dur={duration:.1f}s …"
    )
    ok = ctrl.move_to_traj(x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw, duration=duration)
    if ok:
        time.sleep(duration + 0.1)
        print("  done.")
    else:
        print("  IK failed.")
    return ok


def _current_pose(arm: RobotArm) -> tuple[list[float], list[float]]:
    """Return current EEF (xyz, rpy_rad) via FK."""
    q, _, _ = arm.get_state()
    pos, rpy = joint_to_pose(np.asarray(q, dtype=np.float64))
    return list(pos), list(rpy)


# ── REPL ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="IK-based REPL for moving the arm to Cartesian positions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--positions-file", default="recorded_arm_positions.yaml",
        help="YAML from record_arm_positions.py (default: recorded_arm_positions.yaml).",
    )
    parser.add_argument(
        "--duration", type=float, default=None,
        help="IK move duration in seconds (default: %(default)s → uses _MIN_DURATION).",
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

    data = yaml.safe_load(positions_path.read_text()) or {}
    raw  = data.get("positions") or {}
    if not raw:
        sys.exit(f"No 'positions' found in {positions_path}")
    if _HOME_LABEL not in raw:
        sys.exit(
            f"'{_HOME_LABEL}' not found in {positions_path}. "
            "Re-run record_arm_positions.py to capture it."
        )

    lookup = {name.lower(): (name, entry) for name, entry in raw.items()}

    print(f"Loaded {len(raw)} positions from {positions_path.name}:")
    for name, entry in raw.items():
        ee  = entry.get("end_effector", {})
        xyz = [round(v, 4) for v in ee.get("xyz", [])]
        rpy = [round(math.degrees(v), 1) for v in ee.get("rpy_rad", [])]
        print(f"  {name}: xyz={xyz}  rpy_deg={rpy}")

    print(f"\nGripper motor angles (edit _OPEN/_CLOSE_GRIPPER_POSE in script to change):")
    print(f"  open : {_OPEN_GRIPPER_POSE} rad")
    print(f"  close: {_CLOSE_GRIPPER_POSE} rad")

    arm     = RobotArm(cfg_path=args.arm_cfg)
    ctrl    = ArmEndPos(arm)
    gripper = _make_gripper_handle(arm)

    print("\nConnecting and enabling motors …")
    ctrl.start()
    if gripper is not None:
        gripper.start()
        print(f"Gripper open={_OPEN_GRIPPER_POSE} rad  close={_CLOSE_GRIPPER_POSE} rad")

    # ── home via joint-space on startup ────────────────────────────────────────
    q_home   = np.array(raw[_HOME_LABEL]["joints_rad"], dtype=np.float64)
    q_now, _, _ = arm.get_state()
    home_dur = max(_MIN_DURATION, float(np.max(np.abs(q_home - q_now))) / _DEFAULT_SPEED)
    print(f"Moving to '{_HOME_LABEL}' (joint-space, {home_dur:.1f}s) …")
    _move_joints(ctrl, arm, q_home, home_dur)
    print("At home. Ready.\n")

    default_dur = args.duration or _MIN_DURATION

    try:
        while True:
            try:
                cmd = input("move> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not cmd:
                continue

            parts = cmd.split()
            low   = parts[0].lower()

            # ── quit ──────────────────────────────────────────────────────────
            if low in ("q", "quit", "exit"):
                break

            # ── home ──────────────────────────────────────────────────────────
            if low in ("home", "zero", "zero_act"):
                q_now, _, _ = arm.get_state()
                dur = max(_MIN_DURATION, float(np.max(np.abs(q_home - q_now))) / _DEFAULT_SPEED)
                print(f"Going home (joint-space, {dur:.1f}s) …")
                _move_joints(ctrl, arm, q_home, dur)
                print("Done.")
                continue

            # ── list ──────────────────────────────────────────────────────────
            if low == "list":
                for name, entry in raw.items():
                    ee  = entry.get("end_effector", {})
                    xyz = [round(v, 4) for v in ee.get("xyz", [])]
                    rpy = [round(math.degrees(v), 1) for v in ee.get("rpy_rad", [])]
                    print(f"  {name}: xyz={xyz}  rpy_deg={rpy}")
                continue

            # ── pose ──────────────────────────────────────────────────────────
            if low == "pose":
                pos, rpy = _current_pose(arm)
                print(
                    f"  xyz=[{pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f}]  "
                    f"rpy_deg=[{math.degrees(rpy[0]):+.1f}, "
                    f"{math.degrees(rpy[1]):+.1f}, "
                    f"{math.degrees(rpy[2]):+.1f}]"
                )
                continue

            # ── rel dx dy dz ───────────────────────────────────────────────────
            if low == "rel":
                if len(parts) != 4:
                    print("  Usage: rel <dx> <dy> <dz>  (metres)")
                    continue
                try:
                    dx, dy, dz = float(parts[1]), float(parts[2]), float(parts[3])
                except ValueError:
                    print("  Usage: rel <dx> <dy> <dz>  (metres)")
                    continue
                pos, rpy = _current_pose(arm)
                _ik_move(
                    ctrl,
                    x=pos[0] + dx, y=pos[1] + dy, z=pos[2] + dz,
                    roll=rpy[0], pitch=rpy[1], yaw=rpy[2],
                    duration=default_dur,
                    label=f"rel {dx:+.3f} {dy:+.3f} {dz:+.3f}",
                )
                continue

            # ── open / close gripper ───────────────────────────────────────────
            if low == "open":
                print(f"Gripper → open ({_OPEN_GRIPPER_POSE} rad) …")
                gripper.pos_vel(_OPEN_GRIPPER_POSE)
                print("  done.")
                continue

            if low == "close":
                print(f"Gripper → close ({_CLOSE_GRIPPER_POSE} rad) …")
                gripper.pos_vel(_CLOSE_GRIPPER_POSE)
                print("  done.")
                continue

            # ── gripper <rad> ─────────────────────────────────────────────────
            if low == "gripper":
                if len(parts) != 2:
                    print("  Usage: gripper <rad>   e.g. gripper -2.0")
                    continue
                try:
                    target_rad = float(parts[1])
                except ValueError:
                    print("  Usage: gripper <rad>   e.g. gripper -2.0")
                    continue
                if gripper is None:
                    print("  Gripper not available.")
                else:
                    print(f"Gripper → {target_rad:.4f} rad …")
                    gripper.move_to(target_rad)
                continue

            # ── named position ─────────────────────────────────────────────────
            if low not in lookup:
                print(f"  Unknown command or position '{cmd}'. "
                      "Commands: <name>, rel dx dy dz, open, close, home, list, pose, q")
                continue

            canonical, entry = lookup[low]
            ee  = entry.get("end_effector", {})
            xyz = ee.get("xyz")
            rpy = ee.get("rpy_rad")

            if not xyz or not rpy:
                print(f"  '{canonical}' has no end_effector data — skipping.")
                continue

            _ik_move(
                ctrl,
                x=xyz[0], y=xyz[1], z=xyz[2],
                roll=rpy[0], pitch=rpy[1], yaw=rpy[2],
                duration=default_dur,
                label=canonical,
            )

            g_rad = entry.get("gripper_rad")
            if g_rad is not None and gripper is not None:
                print(f"  Gripper → {g_rad:.4f} rad …")
                gripper.pos_vel(g_rad)
            # if no gripper_rad recorded, leave gripper at its current position

    finally:
        print("Returning home and disconnecting …")
        _move_joints(ctrl, arm, q_home, home_dur)
        ctrl.end()
        gripper.disconnect()
        print("Done.")


if __name__ == "__main__":
    main()
