#!/usr/bin/env python3
"""Interactive REPL — moves to XY positions with a separate Z modifier.

Named positions use recorded XY + orientation; Z is supplied per-command.
zero / zero_act always use joint-space replay.

Commands:
  <name>              move to recorded XY using the file's Z
  <name> up           move to recorded XY at _Z_UP height
  <name> down         move to recorded XY at _Z_DOWN height
  <name> 0.12         move to recorded XY at Z = 0.12 m
  rel dx dy dz        relative Cartesian move from current EEF
  open                gripper to _OPEN_GRIPPER_POSE
  close               gripper to _CLOSE_GRIPPER_POSE
  gripper <rad>       gripper to exact motor angle
  zero_gripper        clear gripper error and command to 0.0 rad
  grab_face           open → lower (yaw=0°) → close → raise
  grab_edge           rotate 45° → open → lower → close → raise
  drop_face           lower (yaw=0°) → open → raise
  drop_edge           rotate 45° → lower → open → raise
  home / zero_act     return home via joint-space
  grid <i> <j>        move to centre of grid cell (i, j) from grid_config.yaml
  list                show all positions
  pose                print current EEF xyz and rpy
  q / quit            home and exit

Usage:
    python move_repl_xy.py
    python move_repl_xy.py --duration 2.5
    python move_repl_xy.py --grid-file grid_config.yaml
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

from compute_grid import compute_grid_map


# ── Z presets — edit to match your table setup ────────────────────────────────

_Z_UP   = 0.15    # m — raised/transit height
_Z_DOWN = 0.075   # m — table/contact height

# ── grab / drop Z heights (tune per object) ───────────────────────────────────

_GRAB_FACE_UP_Z   = 0.10    # m — lift height after face grab
_GRAB_FACE_DOWN_Z = 0.045   # m — contact height for face grab

_GRAB_EDGE_UP_Z   = 0.12    # m — lift height after edge grab
_GRAB_EDGE_DOWN_Z = 0.065   # m — contact height for edge grab

_DROP_FACE_UP_Z   = 0.10    # m — lift height after face drop
_DROP_FACE_DOWN_Z = 0.040   # m — release height for face drop

_DROP_EDGE_UP_Z   = 0.12    # m — lift height after edge drop
_DROP_EDGE_DOWN_Z = 0.055   # m — release height for edge drop

_GRAB_EDGE_YAW    = math.pi / 4   # rad — 45° yaw for edge grab/drop

_GRIPPER_SETTLE    = 0.8    # s — wait after each gripper command

# ── per-function gripper poses (tune open/close independently) ────────────────
# Scale: 0.0 = fully open, 6.5 = fully closed

_GRAB_FACE_OPEN  = 2.5    # rad — open before face grab
_GRAB_FACE_CLOSE = 3.8000    # rad — close after face grab

_GRAB_EDGE_OPEN  = 2.5    # rad — open before edge grab
_GRAB_EDGE_CLOSE = 4.65    # rad — close after edge grab

_DROP_FACE_OPEN  = 2.5    # rad — open to release (face drop)
_DROP_EDGE_OPEN  = 2.5    # rad — open to release (edge drop)

# ── manual open/close command angles ──────────────────────────────────────────

_OPEN_GRIPPER_POSE  = 0.0   # rad — `open`  command
_CLOSE_GRIPPER_POSE = 1.5   # rad — `close` command

# ── motion constants ───────────────────────────────────────────────────────────

_DEFAULT_SPEED = 0.5   # rad/s for joint-space moves
_MIN_DURATION  = 2.0
_HOME_LABEL    = "zero_act"
_JOINT_LABELS  = {"zero", "zero_act"}   # always use joint-space for these


# ── gripper handle ─────────────────────────────────────────────────────────────

def _make_gripper_handle(arm: Any) -> Any:
    class _GripperHandle:
        _RATE = 100.0

        def __init__(self, ctrl: Any, mot: Any) -> None:
            self._ctrl   = ctrl
            self._mot    = mot
            self._target : float | None = None
            self._lock   = threading.Lock()
            self._running = False
            self._thread : threading.Thread | None = None

        def start(self) -> None:
            from motorbridge import Mode
            try:
                self._mot.clear_error()
            except Exception:
                pass
            time.sleep(0.2)
            try:
                self._ctrl.enable_all()
            except Exception:
                pass
            time.sleep(0.5)
            for attempt in range(10):
                try:
                    self._mot.ensure_mode(Mode.FORCE_POS, 2000)
                    print(f"[gripper] FORCE_POS OK (attempt {attempt+1})")
                    break
                except Exception as e:
                    print(f"[gripper] FORCE_POS attempt {attempt+1}: {e}")
                    time.sleep(0.05)
            time.sleep(0.2)
            try:
                self._mot.request_feedback()
                self._ctrl.poll_feedback_once()
                st = self._mot.get_state()
                self._target = float(st.pos) if st is not None else 0.0
            except Exception:
                self._target = 0.0
            print(f"[gripper] ready — current={self._target:.4f} rad")
            self._running = True
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

        def _loop(self) -> None:
            dt = 1.0 / self._RATE
            while self._running:
                t0 = time.perf_counter()
                with self._lock:
                    target = self._target
                if target is not None:
                    try:
                        self._mot.send_force_pos(target, 2.0, 0.1)
                        self._mot.request_feedback()
                        self._ctrl.poll_feedback_once()
                    except Exception:
                        pass
                rem = dt - (time.perf_counter() - t0)
                if rem > 0:
                    time.sleep(rem)

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
        print(f"[warn] Gripper not available ({exc})")
        return None


# ── joint-space trajectory ─────────────────────────────────────────────────────

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
            s = _min_jerk(i / (n - 1))
            ctrl._q_target[:] = q_start + s * (q_target - q_start)
            time.sleep(interval)
        ctrl._q_target[:] = q_target
        done.set()

    t = threading.Thread(target=_send, daemon=True)
    ctrl._send_thread = t
    t.start()
    done.wait(timeout=duration + 3.0)


# ── IK move ────────────────────────────────────────────────────────────────────

def _ik_move(ctrl, x, y, z, roll, pitch, yaw, duration, label="") -> bool:
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
    q, _, _ = arm.get_state()
    pos, rpy = joint_to_pose(np.asarray(q, dtype=np.float64))
    return list(pos), list(rpy)


def zero_gripper(gripper: Any) -> None:
    """Clear gripper error, re-enable FORCE_POS mode, then command to 0.0 rad.

    clear_error() resets the motor out of FORCE_POS, so we must re-run the
    same enable + ensure_mode sequence that start() uses before moving.
    """
    if gripper is None:
        print("[zero_gripper] no gripper attached")
        return
    from motorbridge import Mode
    try:
        gripper._mot.clear_error()
        print("[zero_gripper] error cleared")
    except Exception as exc:
        print(f"[zero_gripper] clear_error: {exc}")
    time.sleep(0.2)
    try:
        gripper._ctrl.enable_all()
    except Exception:
        pass
    time.sleep(0.3)
    for attempt in range(5):
        try:
            gripper._mot.ensure_mode(Mode.FORCE_POS, 2000)
            print(f"[zero_gripper] FORCE_POS OK (attempt {attempt+1})")
            break
        except Exception as exc:
            print(f"[zero_gripper] FORCE_POS attempt {attempt+1}: {exc}")
            time.sleep(0.05)
    time.sleep(0.2)
    gripper.move_to(0.0)
    print("[zero_gripper] → 0.0 rad")


def grab_face(ctrl: Any, arm: Any, gripper: Any,
              up_z: float = _GRAB_FACE_UP_Z,
              down_z: float = _GRAB_FACE_DOWN_Z,
              duration: float = _MIN_DURATION) -> None:
    """Open → lower (yaw=0) → close → raise."""
    pos, _ = _current_pose(arm)
    x, y = pos[0], pos[1]
    if gripper:
        print("  [grab_face] open …")
        gripper.move_to(_GRAB_FACE_OPEN)
        time.sleep(_GRIPPER_SETTLE)
    print(f"  [grab_face] lower z={down_z:.3f} m …")
    _ik_move(ctrl, x, y, down_z, roll=0.0, pitch=math.pi / 2, yaw=0.0,
             duration=duration, label="face:down")
    if gripper:
        print("  [grab_face] close …")
        gripper.move_to(_GRAB_FACE_CLOSE)
        time.sleep(_GRIPPER_SETTLE)
    print(f"  [grab_face] raise z={up_z:.3f} m …")
    _ik_move(ctrl, x, y, up_z, roll=0.0, pitch=math.pi / 2, yaw=0.0,
             duration=duration, label="face:up")


def grab_edge(ctrl: Any, arm: Any, gripper: Any,
              up_z: float = _GRAB_EDGE_UP_Z,
              down_z: float = _GRAB_EDGE_DOWN_Z,
              duration: float = _MIN_DURATION) -> None:
    """Rotate 45° → open → lower → close → raise."""
    pos, _ = _current_pose(arm)
    x, y, z_now = pos[0], pos[1], pos[2]
    print(f"  [grab_edge] rotate yaw={math.degrees(_GRAB_EDGE_YAW):.0f}° …")
    _ik_move(ctrl, x, y, z_now, roll=0.0, pitch=math.pi / 2, yaw=_GRAB_EDGE_YAW,
             duration=duration, label="edge:rotate")
    if gripper:
        print("  [grab_edge] open …")
        gripper.move_to(_GRAB_EDGE_OPEN)
        time.sleep(_GRIPPER_SETTLE)
    print(f"  [grab_edge] lower z={down_z:.3f} m …")
    _ik_move(ctrl, x, y, down_z, roll=0.0, pitch=math.pi / 2, yaw=_GRAB_EDGE_YAW,
             duration=duration, label="edge:down")
    if gripper:
        print("  [grab_edge] close …")
        gripper.move_to(_GRAB_EDGE_CLOSE)
        time.sleep(_GRIPPER_SETTLE)
    print(f"  [grab_edge] raise z={up_z:.3f} m …")
    _ik_move(ctrl, x, y, up_z, roll=0.0, pitch=math.pi / 2, yaw=_GRAB_EDGE_YAW,
             duration=duration, label="edge:up")


def drop_face(ctrl: Any, arm: Any, gripper: Any,
              up_z: float = _DROP_FACE_UP_Z,
              down_z: float = _DROP_FACE_DOWN_Z,
              duration: float = _MIN_DURATION) -> None:
    """Lower (yaw=0) → open → raise."""
    pos, _ = _current_pose(arm)
    x, y = pos[0], pos[1]
    print(f"  [drop_face] lower z={down_z:.3f} m …")
    _ik_move(ctrl, x, y, down_z, roll=0.0, pitch=math.pi / 2, yaw=0.0,
             duration=duration, label="face:down")
    if gripper:
        print("  [drop_face] open …")
        gripper.move_to(_DROP_FACE_OPEN)
        time.sleep(_GRIPPER_SETTLE)
    print(f"  [drop_face] raise z={up_z:.3f} m …")
    _ik_move(ctrl, x, y, up_z, roll=0.0, pitch=math.pi / 2, yaw=0.0,
             duration=duration, label="face:up")


def drop_edge(ctrl: Any, arm: Any, gripper: Any,
              up_z: float = _DROP_EDGE_UP_Z,
              down_z: float = _DROP_EDGE_DOWN_Z,
              duration: float = _MIN_DURATION) -> None:
    """Rotate 45° → lower → open → raise."""
    pos, _ = _current_pose(arm)
    x, y, z_now = pos[0], pos[1], pos[2]
    print(f"  [drop_edge] rotate yaw={math.degrees(_GRAB_EDGE_YAW):.0f}° …")
    _ik_move(ctrl, x, y, z_now, roll=0.0, pitch=math.pi / 2, yaw=_GRAB_EDGE_YAW,
             duration=duration, label="edge:rotate")
    print(f"  [drop_edge] lower z={down_z:.3f} m …")
    _ik_move(ctrl, x, y, down_z, roll=0.0, pitch=math.pi / 2, yaw=_GRAB_EDGE_YAW,
             duration=duration, label="edge:down")
    if gripper:
        print("  [drop_edge] open …")
        gripper.move_to(_DROP_EDGE_OPEN)
        time.sleep(_GRIPPER_SETTLE)
    print(f"  [drop_edge] raise z={up_z:.3f} m …")
    _ik_move(ctrl, x, y, up_z, roll=0.0, pitch=math.pi / 2, yaw=_GRAB_EDGE_YAW,
             duration=duration, label="edge:up")


# ── Agent tool descriptions & dispatcher ──────────────────────────────────────

AGENT_TOOLS = [
    {
        "name": "move_to_pickup_zone",
        "description": (
            "Move the arm to the block pickup zone (recorded position 'A') at raised "
            "transit height (_Z_UP). Call this to position above the block before grab_block."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "grab_block",
        "description": (
            "Grab the block at the current XY position. Lowers the arm, closes the gripper, "
            "then raises. 'face' descends straight down (yaw=0°); "
            "'edge' rotates the wrist 45° first to grip a side edge."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "grip_style": {
                    "type": "string",
                    "enum": ["face", "edge"],
                    "description": (
                        "'face' — straight-down descent, good for flat or loose objects.  "
                        "'edge' — 45° wrist rotation before descent, good for thin upright blocks."
                    ),
                },
            },
            "required": ["grip_style"],
        },
    },
    {
        "name": "move_to_grid_cell",
        "description": (
            "Move the arm above grid cell (i, j) at transit height, ready to drop a block. "
            "Grid layout is defined in grid_config.yaml."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "i": {"type": "integer", "description": "Column index (0-based, left→right)."},
                "j": {"type": "integer", "description": "Row index (0-based, front→back)."},
            },
            "required": ["i", "j"],
        },
    },
    {
        "name": "drop_block",
        "description": (
            "Drop the held block at the current XY position. Lowers the arm, opens the gripper, "
            "then raises. grip_style must match how the block was grabbed."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "grip_style": {
                    "type": "string",
                    "enum": ["face", "edge"],
                    "description": "Must match the grip_style used in the grab_block call.",
                },
            },
            "required": ["grip_style"],
        },
    },
    {
        "name": "home",
        "description": (
            "Return the arm to the zero/home position (zero_act) using joint-space motion. "
            "Use this whenever the user asks to reset, go home, or return to zero."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "move_to_named_position_up",
        "description": (
            "Move the arm to a named recorded position's XY at raised transit height (_Z_UP) "
            "with the gripper pointing straight down. Use for transit moves above known spots."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "position_name": {
                    "type": "string",
                    "description": (
                        "Name of a recorded XY position (case-insensitive, e.g. 'A'). "
                        "Do not use 'zero' or 'zero_act' — those are joint-space only."
                    ),
                },
            },
            "required": ["position_name"],
        },
    },
]


def _tool_move_to_pickup_zone(
    ctrl: Any, lookup: dict, duration: float
) -> str:
    if "a" not in lookup:
        return "error: position 'A' not found in recorded positions"
    _, entry = lookup["a"]
    ee  = entry.get("end_effector", {})
    xyz = ee.get("xyz")
    if not xyz:
        return "error: position 'A' has no end_effector XY data"
    ok = _ik_move(ctrl, xyz[0], xyz[1], _Z_UP,
                  roll=0.0, pitch=math.pi / 2, yaw=0.0,
                  duration=duration, label="pickup_zone(A up)")
    return "moved to pickup zone (A up)" if ok else "IK failed — pickup zone unreachable"


def _tool_grab_block(
    style: str, ctrl: Any, arm: Any, gripper: Any, duration: float
) -> str:
    if style == "face":
        grab_face(ctrl, arm, gripper, duration=duration)
        return "grabbed block (face grip)"
    if style == "edge":
        grab_edge(ctrl, arm, gripper, duration=duration)
        return "grabbed block (edge grip)"
    return f"error: unknown grip_style '{style}' — use 'face' or 'edge'"


def _tool_move_to_grid_cell(
    i: int, j: int,
    ctrl: Any,
    grid_map: dict,
    grid_nx: int, grid_ny: int,
    grid_z: float,
    duration: float,
) -> str:
    if not grid_map:
        return "error: no grid loaded — pass --grid-file"
    if (i, j) not in grid_map:
        return f"error: cell ({i},{j}) out of range — grid is {grid_nx}×{grid_ny}"
    gx, gy = grid_map[(i, j)]
    ok = _ik_move(ctrl, gx, gy, grid_z,
                  roll=0.0, pitch=math.pi / 2, yaw=0.0,
                  duration=duration, label=f"grid({i},{j})")
    return f"moved to grid cell ({i},{j})" if ok else f"IK failed — grid({i},{j}) unreachable"


def _tool_drop_block(
    style: str, ctrl: Any, arm: Any, gripper: Any, duration: float
) -> str:
    if style == "face":
        drop_face(ctrl, arm, gripper, duration=duration)
        return "dropped block (face grip)"
    if style == "edge":
        drop_edge(ctrl, arm, gripper, duration=duration)
        return "dropped block (edge grip)"
    return f"error: unknown grip_style '{style}' — use 'face' or 'edge'"


def _tool_move_to_named_up(
    pos_name: str, ctrl: Any, lookup: dict, duration: float
) -> str:
    low = pos_name.lower()
    if low in _JOINT_LABELS:
        return f"error: '{pos_name}' is a joint-space position — use the home command"
    if low not in lookup:
        return f"error: position '{pos_name}' not found in recorded positions"
    _, entry = lookup[low]
    ee  = entry.get("end_effector", {})
    xyz = ee.get("xyz")
    if not xyz:
        return f"error: position '{pos_name}' has no end_effector XY data"
    ok = _ik_move(ctrl, xyz[0], xyz[1], _Z_UP,
                  roll=0.0, pitch=math.pi / 2, yaw=0.0,
                  duration=duration, label=f"{pos_name} up")
    return f"moved to {pos_name} up" if ok else f"IK failed — {pos_name} up unreachable"


def _tool_home(ctrl: Any, arm: Any, lookup: dict, duration: float) -> str:
    _, entry = lookup[_HOME_LABEL.lower()]
    q_target = np.array(entry["joints_rad"], dtype=np.float64)
    q_now, _, _ = arm.get_state()
    dur = max(_MIN_DURATION, float(np.max(np.abs(q_target - q_now))) / _DEFAULT_SPEED)
    _move_joints(ctrl, arm, q_target, dur)
    return "arm returned to zero/home position"


def dispatch_tool(
    name: str,
    tool_input: dict,
    *,
    ctrl: Any,
    arm: Any,
    gripper: Any,
    grid_map: dict,
    grid_nx: int,
    grid_ny: int,
    grid_z: float,
    lookup: dict,
    duration: float,
) -> str:
    """Execute an agent tool call; returns a plain-text result string."""
    if name == "move_to_pickup_zone":
        return _tool_move_to_pickup_zone(ctrl, lookup, duration)
    if name == "grab_block":
        return _tool_grab_block(
            tool_input.get("grip_style", "face"), ctrl, arm, gripper, duration
        )
    if name == "move_to_grid_cell":
        return _tool_move_to_grid_cell(
            int(tool_input["i"]), int(tool_input["j"]),
            ctrl, grid_map, grid_nx, grid_ny, grid_z, duration,
        )
    if name == "drop_block":
        return _tool_drop_block(
            tool_input.get("grip_style", "face"), ctrl, arm, gripper, duration
        )
    if name == "home":
        return _tool_home(ctrl, arm, lookup, duration)
    if name == "move_to_named_position_up":
        return _tool_move_to_named_up(
            tool_input.get("position_name", ""), ctrl, lookup, duration
        )
    return f"error: unknown tool '{name}'"


def _resolve_z(modifier: str | None, recorded_z: float) -> float | None:
    """Convert a Z modifier token to a Z value in metres.

    Returns None if the token is not a valid Z specifier.
    """
    if modifier is None:
        return recorded_z
    if modifier.lower() == "up":
        return _Z_UP
    if modifier.lower() == "down":
        return _Z_DOWN
    try:
        return float(modifier)
    except ValueError:
        return None


# ── REPL ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="XY+Z REPL — supply Z per-command; zero/zero_act use joint-space.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--positions-file", default="recorded_arm_positions.yaml")
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--arm-cfg", default=None)
    parser.add_argument("--grid-file", default="grid_config.yaml")
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
        sys.exit(f"'{_HOME_LABEL}' not in {positions_path}. Re-run record_arm_positions.py.")

    lookup = {name.lower(): (name, entry) for name, entry in raw.items()}

    print(f"Loaded {len(raw)} positions from {positions_path.name}:")
    for name, entry in raw.items():
        if name.lower() in _JOINT_LABELS:
            print(f"  {name}: [joint-space]")
        else:
            ee  = entry.get("end_effector", {})
            xyz = [round(v, 4) for v in ee.get("xyz", [])]
            rpy = [round(math.degrees(v), 1) for v in ee.get("rpy_rad", [])]
            print(f"  {name}: xyz={xyz}  rpy_deg={rpy}")

    print(f"\nZ presets:  up={_Z_UP} m   down={_Z_DOWN} m")
    print(f"Gripper:    open={_OPEN_GRIPPER_POSE} rad   close={_CLOSE_GRIPPER_POSE} rad\n")

    grid_map: dict[tuple[int, int], tuple[float, float]] = {}
    grid_z: float = _Z_UP
    grid_nx = grid_ny = 0
    grid_path = Path(args.grid_file)
    if not grid_path.is_absolute():
        grid_path = _REPO_ROOT / grid_path
    if grid_path.exists():
        grid_map, grid_z = compute_grid_map(grid_path)
        _gcfg = yaml.safe_load(grid_path.read_text())
        grid_nx = _gcfg["grid"]["nx"]
        grid_ny = _gcfg["grid"]["ny"]
        print(f"Grid loaded: {grid_nx}×{grid_ny} from {grid_path.name}  z={grid_z} m")
    else:
        print(f"[grid] No grid file at {grid_path} — 'grid' command unavailable.")

    arm     = RobotArm(cfg_path=args.arm_cfg)
    ctrl    = ArmEndPos(arm)
    gripper = _make_gripper_handle(arm)

    print("Connecting and enabling motors …")
    ctrl.start()
    if gripper is not None:
        gripper.start()

    q_home   = np.array(raw[_HOME_LABEL]["joints_rad"], dtype=np.float64)
    q_now, _, _ = arm.get_state()
    home_dur = max(_MIN_DURATION, float(np.max(np.abs(q_home - q_now))) / _DEFAULT_SPEED)
    print(f"Moving to '{_HOME_LABEL}' ({home_dur:.1f}s) …")
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

            # ── home / joint-space labels ──────────────────────────────────────
            if low in {"home"} | _JOINT_LABELS:
                if low in lookup:
                    _, entry = lookup[low]
                    q_target = np.array(entry["joints_rad"], dtype=np.float64)
                else:
                    q_target = q_home
                q_now, _, _ = arm.get_state()
                dur = max(_MIN_DURATION, float(np.max(np.abs(q_target - q_now))) / _DEFAULT_SPEED)
                print(f"Going to '{low}' (joint-space, {dur:.1f}s) …")
                _move_joints(ctrl, arm, q_target, dur)
                print("Done.")
                continue

            # ── list ──────────────────────────────────────────────────────────
            if low == "list":
                for name, entry in raw.items():
                    if name.lower() in _JOINT_LABELS:
                        print(f"  {name}: [joint-space]")
                    else:
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
                    x=pos[0]+dx, y=pos[1]+dy, z=pos[2]+dz,
                    roll=rpy[0], pitch=rpy[1], yaw=rpy[2],
                    duration=default_dur,
                    label=f"rel {dx:+.3f} {dy:+.3f} {dz:+.3f}",
                )
                continue

            # ── open / close gripper ───────────────────────────────────────────
            if low == "open":
                if gripper:
                    gripper.pos_vel(_OPEN_GRIPPER_POSE)
                    print(f"  Gripper → {_OPEN_GRIPPER_POSE} rad")
                continue

            if low == "close":
                if gripper:
                    gripper.pos_vel(_CLOSE_GRIPPER_POSE)
                    print(f"  Gripper → {_CLOSE_GRIPPER_POSE} rad")
                continue

            # ── gripper <rad> ─────────────────────────────────────────────────
            if low == "gripper":
                if len(parts) != 2:
                    print("  Usage: gripper <rad>")
                    continue
                try:
                    target_rad = float(parts[1])
                except ValueError:
                    print("  Usage: gripper <rad>")
                    continue
                if gripper:
                    gripper.move_to(target_rad)
                    print(f"  Gripper → {target_rad:.4f} rad")
                continue

            # ── zero_gripper ──────────────────────────────────────────────
            if low == "zero_gripper":
                zero_gripper(gripper)
                continue

            # ── grab_face / grab_edge ─────────────────────────────────────────
            if low == "grab_face":
                grab_face(ctrl, arm, gripper, duration=default_dur)
                continue

            if low == "grab_edge":
                grab_edge(ctrl, arm, gripper, duration=default_dur)
                continue

            if low == "drop_face":
                drop_face(ctrl, arm, gripper, duration=default_dur)
                continue

            if low == "drop_edge":
                drop_edge(ctrl, arm, gripper, duration=default_dur)
                continue

            # ── grid <i> <j> ──────────────────────────────────────────────────
            if low == "grid":
                if not grid_map:
                    print("  No grid loaded. Pass --grid-file or add grid_config.yaml.")
                    continue
                if len(parts) != 3:
                    print(f"  Usage: grid <i> <j>   (i: 0..{grid_nx-1}, j: 0..{grid_ny-1})")
                    continue
                try:
                    gi, gj = int(parts[1]), int(parts[2])
                except ValueError:
                    print(f"  Usage: grid <i> <j>   (i: 0..{grid_nx-1}, j: 0..{grid_ny-1})")
                    continue
                if (gi, gj) not in grid_map:
                    print(f"  Cell ({gi}, {gj}) out of range. Grid is {grid_nx}×{grid_ny}.")
                    continue
                gx, gy = grid_map[(gi, gj)]
                _ik_move(
                    ctrl,
                    x=gx, y=gy, z=grid_z,
                    roll=0.0, pitch=math.pi / 2, yaw=0.0,
                    duration=default_dur,
                    label=f"grid({gi},{gj})",
                )
                continue

            # ── named position: <name> [up|down|z_m] ─────────────────────────
            if low not in lookup:
                print(
                    f"  Unknown: '{cmd}'.\n"
                    f"  Commands: <name> [up|down|z_m], rel dx dy dz, "
                    f"open, close, gripper <rad>, home, list, pose, q"
                )
                continue

            canonical, entry = lookup[low]

            # Joint-space labels are already handled above; skip here.
            if low in _JOINT_LABELS:
                continue

            ee  = entry.get("end_effector", {})
            xyz = ee.get("xyz")
            rpy = ee.get("rpy_rad")

            if not xyz or not rpy:
                print(f"  '{canonical}' has no end_effector data — skipping.")
                continue

            # Resolve Z modifier (second token, optional)
            z_tok = parts[1] if len(parts) >= 2 else None
            print("Z_tok: ", z_tok)
            z = _resolve_z(z_tok, recorded_z=xyz[2])

            if z is None:
                print(f"  Bad Z modifier '{z_tok}'. Use: up / down / <metres>")
                continue

            if z_tok is not None:
                # Z is overridden → gripper straight down, no rotation.
                # roll=0, pitch=π/2, yaw=0 gives a consistent pointing-down
                # orientation so up/down moves are pure Z translations.
                # Use `<name>` with no modifier to replay the recorded yaw.
                use_roll  = 0.0
                use_pitch = math.pi / 2
                use_yaw   = 0.0
            else:
                # No modifier → replay the exact recorded orientation.
                use_roll, use_pitch, use_yaw = rpy[0], rpy[1], rpy[2]

            z_label = z_tok if z_tok else "file"
            _ik_move(
                ctrl,
                x=xyz[0], y=xyz[1], z=z,
                roll=use_roll, pitch=use_pitch, yaw=use_yaw,
                duration=default_dur,
                label=f"{canonical} z={z_label}",
            )

            g_rad = entry.get("gripper_rad")
            if g_rad is not None and gripper is not None:
                print(f"  Gripper → {g_rad:.4f} rad …")
                gripper.pos_vel(g_rad)

    finally:
        print("Returning home and disconnecting …")
        _move_joints(ctrl, arm, q_home, home_dur)
        ctrl.end()
        if gripper is not None:
            gripper.disconnect()
        print("Done.")


if __name__ == "__main__":
    main()
