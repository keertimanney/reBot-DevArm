# IK-Based Motion Guide — reBot Arm B601

A complete reference for moving the reBot arm between poses using inverse kinematics,
SE(3) geodesic trajectory planning, and CLIK tracking in both MeshCAT and MuJoCo simulation.

---

## Table of Contents

1. [How It Works](#how-it-works)
2. [Install](#install)
3. [Quick Start](#quick-start)
4. [End-Effector Orientation](#end-effector-orientation)
5. [Reachable Workspace](#reachable-workspace)
6. [CLI Reference](#cli-reference)
7. [Python API Reference](#python-api-reference)
8. [Backends](#backends)
9. [Waypoint Format](#waypoint-format)
10. [Under the Hood](#under-the-hood)
11. [Troubleshooting](#troubleshooting)

---

## How It Works

```
Target pose (x, y, z, yaw)
        │
        ▼
  ┌─────────────┐
  │ EEF-down    │  pitch=π/2 → gripper X-axis points to world [0,0,-1]
  │ orientation │  yaw rotates about world Z while staying down
  └──────┬──────┘
         │  pin.SE3 target
         ▼
  ┌──────────────���──────────────┐
  │  Pinocchio IK (DLS CLIK)    │  Damped Least Squares, adaptive damping,
  │  solve_ik_with_retry        │  line-search backtracking, random restarts
  └──────┬──────────────────────┘
         │  q_end (joint config)
         ▼
  ┌──────────────────────────────────────┐
  │  SE(3) Geodesic Trajectory Sampler  │  interpolates on the Lie group
  │  plan_cartesian_geodesic_trajectory  │  min-jerk time profile, dt=20 ms
  └──────┬───────────────────────────────┘
         │  CartesianTrajectory (SE3 waypoints)
         ▼
  ┌────────────────────────────────────────┐
  │  CLIK Tracker (track_trajectory)      │  tracks each Cartesian waypoint
  │  DLS + null-space joint-limit proj.   │  with joint-limit avoidance
  └──────┬─────────────────────────────────┘
         │  JointTrajectory (q, t per step)
         ▼
  ┌─────────────┐   ┌──────────────────────────────────────┐
  │  MeshCAT    │   │  MuJoCo                              │
  │  display_q  │   │  PD control: τ = kp(q_ref−q)−kd·dq  │
  │  FK viz     │   │  physics: gravity, contacts, inertia │
  └─────────────┘   └──────────────────────────────────────┘
```

---

## Install

From the repo root, with your conda environment active:

```bash
conda activate rebot

# Install the sim package (includes MeshCAT backend)
pip install -e rebot_b601_sim

# Add MuJoCo support
pip install -e 'rebot_b601_sim[mujoco]'
```

Verify:

```bash
python -c "from rebot_b601_sim.robot_mover import RobotMover; print('OK')"
```

---

## Quick Start

### MeshCAT (browser viewer, no physics)

```bash
# Move through two poses, viewer opens in browser automatically
python rebot_b601_sim/scripts/move_robot.py \
  --waypoints 0.35,0.0,0.15  0.40,0.10,0.10

# Interactive prompt — type poses, see them live
python rebot_b601_sim/scripts/move_robot.py --interactive
```

### MuJoCo (physics simulation)

On macOS the viewer requires `mjpython`:

```bash
mjpython rebot_b601_sim/scripts/move_robot.py --backend mujoco \
  --waypoints 0.35,0.0,0.15  0.40,0.10,0.10  0.30,-0.10,0.05
```

On Linux, plain `python` works:

```bash
python rebot_b601_sim/scripts/move_robot.py --backend mujoco \
  --waypoints 0.35,0.0,0.15  0.40,0.10,0.10
```

### Python API (two lines)

```python
from rebot_b601_sim.robot_mover import RobotMover

mover = RobotMover()                          # MeshCAT by default
mover.move_to(0.35, 0.0, 0.15)               # arm moves, browser updates
```

---

## End-Effector Orientation

### The "pointing down" convention

The B601's gripper extends along the **local X-axis** of the `end_link` frame
(confirmed by the MeshCAT overlay: palm and fingers are placed at `+X` offsets from
the end link). "Pointing down" means that local X aligns with world `[0, 0, -1]`.

This is achieved with `pitch = π/2` in ZYX Euler convention:

```
R_down = Rz(yaw) · Ry(π/2) · Rx(0)
```

| Column of R_down | World direction |
|---|---|
| X (approach) | [0, 0, −1] — pointing at the floor |
| Y (left) | depends on yaw |
| Z (up of tool) | depends on yaw |

At the robot's neutral pose (`q = 0`), the EEF is at `[0.26, 0, 0.19]` with X
pointing forward. The `pitch = π/2` rotation tilts the gripper so its mouth faces
straight down.

### Yaw

`yaw` spins the gripper about the world Z-axis while keeping it pointed down. This
lets you orient the gripper fingers for different object approach angles.

```python
mover.move_to(0.35, 0.0, 0.10, yaw=0.0)      # fingers aligned with world X
mover.move_to(0.35, 0.0, 0.10, yaw=1.57)     # fingers rotated 90° about Z
mover.move_to(0.35, 0.0, 0.10, yaw=-0.79)    # fingers at −45°
```

---

## Reachable Workspace

With the EEF-down constraint (`pitch = π/2`, `yaw = 0`), the reachable positions are:

```
        z=0.05  z=0.10  z=0.15  z=0.20
x=0.20    ✓       ✓       ✓       —
x=0.25    ✓       ✓       ✓       —
x=0.30    ✓       ✓       ✓       —
x=0.35    ✓       ✓       ✓       —
x=0.40    ✓       ✓       ✓       —
x=0.45    ✓       ✓       —       —
```

*(y = 0 in the table above. Lateral reach ±0.15 m is typical at these depths.)*

Rules of thumb:
- **Safe pick zone**: `x ∈ [0.20, 0.40]`, `z ∈ [0.05, 0.15]`
- **z > 0.15 m** at most x values hits the arm's backward reach limit
- **x > 0.45 m** extends past the arm's forward reach
- Changing `yaw` slightly shifts the reachable region but does not dramatically change it

---

## CLI Reference

```
python rebot_b601_sim/scripts/move_robot.py [OPTIONS]
```

or, if the package is installed:

```
rebot-b601-move [OPTIONS]
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--backend` | `meshcat` \| `mujoco` | `meshcat` | Simulation backend |
| `--to X Y Z` | floats | — | Move to single XYZ target (metres) |
| `--yaw` | float | `0.0` | Gripper yaw about world Z (radians) |
| `--waypoints x,y,z ...` | list | — | Sequence of poses (see format below) |
| `--duration` | float | auto | Move duration in seconds |
| `--speed` | float | `0.10` | m/s for auto-duration calculation |
| `--gripper` | float | `0.0` | 0 = closed, −270 = fully open (degrees) |
| `--interactive` | flag | off | Interactive prompt (meshcat only) |
| `--workcell` | path | default cubes | Workcell JSON to load |
| `--no-browser` | flag | off | Skip auto-opening MeshCAT browser tab |

### Examples

```bash
# Move to one position
rebot-b601-move --to 0.35 0.0 0.15

# Move with open gripper
rebot-b601-move --to 0.35 0.0 0.10 --gripper -270

# Sequence of waypoints
rebot-b601-move --waypoints 0.3,0.1,0.12  0.4,-0.1,0.08  0.35,0.0,0.15

# Waypoints with per-pose yaw (radians)
rebot-b601-move --waypoints 0.35,0.0,0.15,0.0  0.40,0.10,0.10,1.57

# Slow motion (0.05 m/s)
rebot-b601-move --speed 0.05 --waypoints 0.35,0.0,0.15  0.40,0.10,0.10

# MuJoCo, macOS
mjpython rebot_b601_sim/scripts/move_robot.py --backend mujoco \
  --waypoints 0.35,0.0,0.15  0.40,0.10,0.10
```

### Interactive mode

```bash
rebot-b601-move --interactive
```

Prompt shows current EEF position. Commands:

| Input | Action |
|-------|--------|
| `x y z` | move to position (metres) |
| `x y z yaw_deg` | move with yaw in degrees |
| `home` | return to neutral configuration |
| `pose` | print current EEF position and RPY |
| `q` / `quit` | exit |

---

## Python API Reference

### `RobotMover(backend, control_repo, open_browser, workcell, params)`

```python
from rebot_b601_sim.robot_mover import RobotMover, MoveParams

mover = RobotMover(
    backend="meshcat",   # "meshcat" or "mujoco"
    open_browser=True,   # open MeshCAT tab on init (meshcat only)
    workcell=None,       # Path to workcell JSON, optional
    params=MoveParams(), # see MoveParams below
)
```

### `move_to(x, y, z, yaw=0.0, duration=None, gripper=0.0) → bool`

Move the end effector to `(x, y, z)` with EEF pointing down.

```python
ok = mover.move_to(0.35, 0.0, 0.15)
ok = mover.move_to(0.35, 0.0, 0.15, yaw=1.57)          # with yaw
ok = mover.move_to(0.35, 0.0, 0.15, duration=3.0)       # fixed 3s move
ok = mover.move_to(0.35, 0.0, 0.05, gripper=-270)       # open gripper
```

Returns `True` if IK found a solution. Returns `False` and prints a warning if
the position is unreachable.

For `mujoco` backend: queues the move; call `run_mujoco()` to execute.

### `run_waypoints(waypoints, yaw=0.0, duration=None, gripper=0.0)`

Execute a list of poses in sequence.

```python
mover.run_waypoints([
    (0.30,  0.10, 0.12),
    (0.40, -0.10, 0.08),
    (0.35,  0.00, 0.10),
])

# Per-pose yaw (4-tuple)
mover.run_waypoints([
    (0.35, 0.0, 0.15, 0.0),
    (0.40, 0.1, 0.10, 1.57),
])
```

### `home()`

Smooth geodesic trajectory back to the zero/neutral configuration.

```python
mover.home()
```

### `run_mujoco()`

Launch the MuJoCo viewer and execute all queued moves. **Blocking** — returns when
the viewer window is closed.

On macOS the script must be run under `mjpython` (the function re-execs
automatically when invoked via the CLI).

```python
mover = RobotMover(backend="mujoco")
mover.run_waypoints([(0.35, 0.0, 0.15), (0.40, 0.10, 0.10)])
mover.run_mujoco()    # opens physics viewer, blocks until window closed
```

### `current_pose() → (position, rpy)`

Return the current end-effector position and orientation.

```python
pos, rpy = mover.current_pose()
# pos: np.ndarray shape (3,)  — [x, y, z] in metres
# rpy: np.ndarray shape (3,)  — [roll, pitch, yaw] in radians
```

### `target_pose(x, y, z, yaw=0.0) → pin.SE3`

Build a `pinocchio.SE3` pose with EEF-down orientation at `(x, y, z)`.
Useful for custom IK calls or visualisation.

```python
T = mover.target_pose(0.35, 0.0, 0.15)
```

### `MoveParams`

All trajectory and controller knobs:

```python
@dataclass
class MoveParams:
    dt:            float = 0.02    # trajectory sample period (s)
    linear_speed:  float = 0.10   # m/s for auto-duration
    min_duration:  float = 1.0    # minimum move time (s)
    ik_max_iter:   int   = 200
    ik_tolerance:  float = 1e-4
    ik_damping:    float = 1e-6
    ik_step_size:  float = 0.8
    null_gain:     float = 0.10   # null-space joint-limit avoidance
    kp:            float = 35.0   # MuJoCo PD stiffness
    kd:            float = 2.0    # MuJoCo PD damping
```

Usage:

```python
from rebot_b601_sim.robot_mover import RobotMover, MoveParams

params = MoveParams(linear_speed=0.05, null_gain=0.2)
mover = RobotMover(params=params)
```

---

## Backends

### MeshCAT (default)

- Loads URDF + meshes via Pinocchio, renders in a browser tab
- **No physics** — pure FK visualisation, no gravity or collisions
- `move_to()` plays back each trajectory step in real time and returns
- Best for: rapid iteration, debugging poses, interactive exploration
- Gripper overlay is visual only (not a kinematic joint)

### MuJoCo

- Full rigid-body physics: gravity, joint limits, contact forces
- PD joint controller tracks the IK-planned trajectory
- `move_to()` queues moves; `run_mujoco()` opens the viewer and runs them
- Best for: verifying trajectories against physics, pick-and-place demos
- On macOS must run under `mjpython` (automatic re-exec from CLI)
- Gripper: synthetic sliding finger boxes under `end_link` — visual + collision, not CAD-accurate

---

## Waypoint Format

Waypoints on the CLI are comma-separated with no spaces:

```
x,y,z           # metres, EEF down, yaw=0
x,y,z,yaw       # metres + yaw in radians
```

```bash
--waypoints 0.35,0.0,0.15  0.40,0.10,0.10  0.30,-0.10,0.05
--waypoints 0.35,0.0,0.15,0.0  0.40,0.10,0.10,1.57
```

In Python, pass a list of tuples:

```python
mover.run_waypoints([
    (0.35,  0.00, 0.15),           # 3-tuple: yaw defaults to 0
    (0.40,  0.10, 0.10, 1.57),     # 4-tuple: explicit yaw
    (0.30, -0.10, 0.05, -0.79),
])
```

---

## Under the Hood

### IK Solver — Damped Least Squares CLIK

File: [`reBotArm_control_py/kinematics/inverse_kinematics.py`](reBotArm_control_py/reBotArm_control_py/kinematics/inverse_kinematics.py)

The IK uses the closed-loop inverse kinematics (CLIK) algorithm:

```
dq = α · Jᵀ · (J Jᵀ + λI)⁻¹ · err
q  ← clamp(integrate(q, dq), joint_limits)
```

Where:
- `err` = `log6(T_current⁻¹ · T_target)` — 6D SE(3) error (local frame)
- `J` = LOCAL frame Jacobian via `pinocchio.getFrameJacobian`
- `λ = damping · max(1, ‖err‖ · 10)` — adaptive Levenberg–Marquardt damping
- `α` = step size, halved up to 4× if error increases (line search)

If the first attempt fails, `solve_ik_with_retry` samples up to 8 random seeds
within joint limits and returns the best solution found.

### Trajectory Planner — SE(3) Geodesic + CLIK Tracker

Files:
- [`reBotArm_control_py/trajectory/sampler.py`](reBotArm_control_py/reBotArm_control_py/trajectory/sampler.py)
- [`reBotArm_control_py/trajectory/clik_tracker.py`](reBotArm_control_py/reBotArm_control_py/trajectory/clik_tracker.py)

**Sampler** interpolates on the SE(3) Lie group:

```
T(s) = T_start · exp(s · log(T_start⁻¹ · T_end))   s ∈ [0, 1]
```

with a minimum-jerk time profile `s = 10t³ − 15t⁴ + 6t⁵`. The trajectory is
sampled at `dt = 20 ms` (50 Hz).

**CLIK Tracker** then follows each SE(3) waypoint with the same DLS solver,
plus a null-space term that pushes joints away from their limits:

```
dq = dq_task + null_gain · (I − Jᵀ(JJᵀ)⁻¹J) · ∇g(q)
```

where `∇g(q)` is the joint-limit gradient.

### MuJoCo PD Controller

The MuJoCo backend tracks the joint trajectory with a simple PD controller
applied as `qfrc_applied`:

```
τᵢ = kp · (q_ref,i − q_i) − kd · dq_i
```

Defaults: `kp = 35 N·m/rad`, `kd = 2 N·m·s/rad`. Adjust via `MoveParams`.

### Joint Limits

| Joint | Min (°) | Max (°) |
|-------|---------|---------|
| joint1 (shoulder_pan) | −160 | +160 |
| joint2 (shoulder_lift) | −180 | 0 |
| joint3 (elbow_flex) | −180 | 0 |
| joint4 (wrist_flex) | −107 | +90 |
| joint5 (wrist_yaw) | −90 | +90 |
| joint6 (wrist_roll) | −180 | +180 |

---

## Troubleshooting

### "IK failed for pos=(x, y, z)"

The position is outside the reachable workspace with EEF-down orientation.

- Check the [Reachable Workspace](#reachable-workspace) table
- Keep `x ∈ [0.20, 0.40]`, `z ∈ [0.05, 0.15]`
- Try reducing z slightly (the arm can't easily point down at high z)
- Increase `ik_max_iter` in `MoveParams` for stubborn cases

### MuJoCo viewer doesn't open on macOS

The viewer requires `mjpython`:

```bash
# Wrong
python rebot_b601_sim/scripts/move_robot.py --backend mujoco ...

# Correct
mjpython rebot_b601_sim/scripts/move_robot.py --backend mujoco ...
```

Check that `mjpython` is available:

```bash
which mjpython     # should print a path
```

If missing, re-install MuJoCo:

```bash
pip install mujoco
```

### "No module named reBotArm_control_py.kinematics"

The package auto-detects its sibling directory. If the repo layout is non-standard,
set the environment variable:

```bash
export REBOTARM_CONTROL_PY=/path/to/reBotArm_control_py
```

### Jerky motion or IK diverging mid-trajectory

- Increase `null_gain` (default 0.10) to better avoid joint limits: `MoveParams(null_gain=0.2)`
- Reduce `linear_speed` to give the IK more time per step: `MoveParams(linear_speed=0.05)`
- Reduce `dt` for a denser trajectory: `MoveParams(dt=0.01)`

### MuJoCo arm oscillates or overshoots

Reduce PD gains:

```python
params = MoveParams(kp=20.0, kd=3.0)
mover = RobotMover(backend="mujoco", params=params)
```

---

## File Map

```
reBot-DevArm/
├── IK_MOTION.md                          ← this file
│
├── rebot_b601_sim/
│   ├── scripts/
│   │   └── move_robot.py                 ← CLI entry point
│   └── src/rebot_b601_sim/
│       ├── robot_mover.py                ← RobotMover class (Python API)
│       ├── move_robot.py                 ← CLI main() + interactive mode
│       ├── pinocchio_meshcat.py          ← B601MeshcatSim (MeshCAT backend)
│       └── mujoco_sim.py                 ← MuJoCo physics backend helpers
│
└── reBotArm_control_py/reBotArm_control_py/
    ├── kinematics/
    │   ├── robot_model.py                ← URDF loader
    │   ├── forward_kinematics.py         ← compute_fk()
    │   └── inverse_kinematics.py         ← solve_ik(), solve_ik_with_retry()
    └── trajectory/
        ├── sampler.py                    ← SE(3) geodesic sampler
        ├── clik_tracker.py               ← CLIK trajectory tracker
        └── trajectory_planner.py         ← plan_joint_space_trajectory()
```
