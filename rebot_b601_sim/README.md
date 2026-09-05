# reBot B601-DM Simulation

This directory is a small companion layer for the Seeed B601-DM Pinocchio/MeshCat workflow documented at:

- https://wiki.seeedstudio.com/rebot_arm_b601_dm_pinocchio_meshcat/
- https://github.com/vectorBH6/reBotArm_control_py

It covers three workflows:

1. Start a Pinocchio/MeshCat robot simulation.
2. Read the reBot Arm 102 leader and move the simulated B601 follower.
3. Load modular workcell geometry from JSON. The starter config places two STL cubes in the workspace.
4. Start a MuJoCo dynamics sim that uses the same workcell JSON with free-moving objects.
5. Read the reBot Arm 102 leader and move the MuJoCo robot in the physics scene.

## Install

From the repo root:

```bash
conda activate rebot
python -m pip install -e rebot_b601_sim
```

Use the same conda environment for all related packages. If you run the command
from `base`, install the local LeRobot packages into `base`; if you run from
`rebot`, install the sim package into `rebot`.

Leader-to-sim also needs:

```bash
python -m pip install -e ./lerobot -e ./lerobot-teleoperator-rebot-arm-102
```

For the MuJoCo backend:

```bash
python -m pip install -e 'rebot_b601_sim[mujoco]'
```

The robot URDF and meshes are expected from Seeed's Pinocchio project. Fetch it beside this checkout:

```bash
rebot-b601-setup-sim --fetch-upstream
```

Or point to an existing checkout:

```bash
export REBOTARM_CONTROL_PY=/Users/kanishk/rebot_lerobot/reBotArm_control_py
```

Use the outer `reBotArm_control_py` checkout here. The nested
`reBotArm_control_py/reBotArm_control_py` directory is the Python package; it does
not contain the `urdf/` meshes this simulator needs.

## 1. Start The Simulation

```bash
rebot-b601-sim
```

To set an initial pose in degrees:

```bash
rebot-b601-sim --q-deg 0 -30 -60 20 0 0
```

## 2. Leader To Simulated Follower

Plug in the reBot Arm 102 leader, then run:

```bash
rebot-b601-leader-sim --leader-port /dev/cu.usbserial-0001
```

This uses the existing `RebotArm102Leader` integration from `lerobot-teleoperator-rebot-arm-102`, including its calibration, joint direction, unwrapping, and clamp logic. The gripper action is read but ignored by the 6-DOF Pinocchio arm model.

The upstream URDF has a fixed `end_link` rather than a real actuated gripper.
This simulator draws a lightweight visual gripper overlay in MeshCat and drives
that overlay from `gripper.pos`, so leader gripper motion is visible. It is not
a physical gripper and it does not create contact forces.

## 3. Modular Geometry JSON

The default workcell is [configs/workcell_two_cubes.json](configs/workcell_two_cubes.json). Each object has:

- `name`: MeshCat node name.
- `mesh`: STL path, relative to the JSON file.
- `color`: Hex color.
- `pose.xyz`: Position in meters.
- `pose.rpy`: Roll, pitch, yaw in radians. `quat_xyzw` is also supported.
- `mass`: Object mass in kg. This is used by MuJoCo and ignored by MeshCat.

Example:

```json
{
  "name": "red_cube",
  "mesh": "../assets/cube_10cm.stl",
  "color": "0xd94b3d",
  "pose": {
    "xyz": [0.32, -0.12, 0.05],
    "rpy": [0.0, 0.0, 0.0]
  }
}
```

Load a different scene:

```bash
rebot-b601-sim --workcell rebot_b601_sim/configs/workcell_two_cubes.json
```

## 4. MuJoCo Dynamics Sim

Run the contact/dynamics backend:

```bash
rebot-b601-mujoco
```

On macOS, MuJoCo's viewer has to run under `mjpython`. The command auto-detects
that and re-launches itself with the environment's `mjpython` when a viewer is
requested.

Headless smoke test:

```bash
rebot-b601-mujoco --no-viewer --duration 2
```

Set an initial robot pose in degrees:

```bash
rebot-b601-mujoco --q-deg 0 -30 -60 20 0 0
```

Test the synthetic MuJoCo gripper without the leader:

```bash
rebot-b601-mujoco --gripper-deg -270
```

Drive MuJoCo from the physical leader:

```bash
rebot-b601-leader-mujoco --leader-port /dev/cu.usbserial-10
```

Headless leader-to-MuJoCo run:

```bash
rebot-b601-leader-mujoco --leader-port /dev/cu.usbserial-10 --no-viewer
```

This backend builds a temporary MuJoCo-loadable URDF from the same robot URDF and
workcell JSON. The JSON objects are added as free bodies with mass and mesh
collision, so gravity and contacts are active.

Current limitation: the upstream B601 URDF only has a simple fixed end link, not
a complete two-finger gripper model with actuated fingers. The cubes can collide
with the robot and floor, but reliable grasping will need a richer gripper
description or a MuJoCo-native MJCF model with finger joints, collision geoms,
and actuators.

To make leader gripper commands visible and contact-capable in MuJoCo, the
generated MuJoCo scene adds a simple synthetic palm and two sliding finger boxes
under `end_link`. This is not the real B601 gripper CAD, but it gives the
simulator movable gripper collision geometry while a proper gripper model is
missing upstream.

## Notes

- Seeed's wiki recommends Python 3.10+, Ubuntu 22.04+, `uv sync`, Pinocchio, and MeshCat. This repo currently runs on your macOS `rebot` conda workflow, but hardware permissions and CAN setup still follow your local drivers.
- The setup script does not copy the upstream repo into this package. It only checks/clones a sibling checkout so the URDF/mesh assets stay externally owned.
