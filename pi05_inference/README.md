# Pi0.5 Zero-Shot Inference on reBot Arm

Run Physical Intelligence's π0.5 foundation model on the reBot B601 DM arm **without fine-tuning**, using an end-effector pose translation layer to bridge the embodiment gap between Franka (DROID) and reBot.

## Architecture

```
┌──────────────┐         ┌──────────────────┐         ┌──────────────┐
│  reBot Arm   │◄───────►│  Control Loop    │◄───────►│  pi0.5 Server│
│  (6 DOF)     │  CAN    │  + Translator    │  WS     │  (RTX 4090)  │
└──────────────┘         └──────────────────┘         └──────────────┘
```

## The Embodiment Translation Problem

π0.5 was trained on Franka Panda data (7-DOF, 855mm reach). The reBot arm is a different robot (6-DOF, 650mm reach). Joint-level actions don't transfer directly — a 0.1 rad delta on Franka joint 3 means something completely different on reBot joint 3 because the kinematics are different.

**Solution:** Use end-effector (Cartesian) pose as a shared coordinate space.

## Translation Pipeline

### Observation (reBot → pi0.5)

The model needs to "see" the arm in Franka joint space:

```
reBot joint positions (6)
    │
    ▼  reBot Forward Kinematics
EE pose (position + orientation)
    │
    ▼  Scale workspace (650mm → 855mm)
Scaled EE pose in Franka workspace
    │
    ▼  Franka Inverse Kinematics
Franka joint positions (7)  ──→  pi0.5 input
```

### Action (pi0.5 → reBot)

The model outputs Franka joint deltas that we convert back:

```
pi0.5 output: Franka joint deltas (7) + gripper (1)
    │
    ▼  Accumulate deltas onto Franka state
Franka joint positions (7)
    │
    ▼  Franka Forward Kinematics
EE pose (position + orientation)
    │
    ▼  Scale workspace (855mm → 650mm)
Scaled EE pose in reBot workspace
    │
    ▼  reBot Inverse Kinematics
reBot joint targets (6)  ──→  send to motors via MIT mode
```

### Workspace Scaling

The two arms have different reaches, so we scale Cartesian positions proportionally:

| | Franka Panda | reBot B601 DM |
|---|---|---|
| DOF | 7 | 6 |
| Reach | 855 mm | 650 mm |
| Scale factor | 1.0x | 0.76x |

Orientations pass through directly — both arms have wrist articulation.

## Setup

### Prerequisites

```bash
pip install pinocchio robot_descriptions openpi-client opencv-python numpy
pip install -e ./reBotArm_control_py   # reBot motor control + URDF
```

### One-time setup

```bash
./setup.sh
```

This clones openpi, reBotArm_control_py, and installs all dependencies.

## Running

### 1. Start the pi0.5 server (on GPU machine)

**Local GPU:**
```bash
./serve.sh
```

**Vast.ai (remote RTX 4090):**
```bash
./vast_deploy.sh
# Then tunnel:
ssh -N -L 8000:localhost:8000 -p <PORT> root@<HOST>
```

### 2. Run the control loop (on machine connected to arm)

```bash
python run_rebot.py \
    --prompt "pick up the red block" \
    --host localhost \
    --port 8000 \
    --camera-id 0
```

**Options:**
| Flag | Default | Description |
|------|---------|-------------|
| `--prompt` | "pick up the object" | Natural language task instruction |
| `--host` | localhost | pi0.5 server address |
| `--port` | 8000 | pi0.5 server port |
| `--camera-id` | 0 | Main camera device ID |
| `--wrist-camera-id` | None | Wrist camera device ID |
| `--urdf` | auto-detected | Path to reBot URDF file |

## Files

| File | Purpose |
|------|---------|
| `embodiment_translator.py` | Core translation layer (FK/IK between embodiments) |
| `run_rebot.py` | Main control loop with translation |
| `serve.sh` | Start pi0.5 server locally |
| `vast_deploy.sh` | Deploy pi0.5 on Vast.ai RTX 4090 |
| `setup.sh` | One-time dependency installation |
| `test_inference.py` | Quick test to verify server is responding |

## How It Works Under the Hood

### `EmbodimentTranslator` class

```python
from embodiment_translator import EmbodimentTranslator

translator = EmbodimentTranslator(rebot_urdf_path="path/to/reBot.urdf")

# Observation: convert reBot state to what pi0.5 expects
franka_joints = translator.rebot_to_franka(rebot_joint_positions)

# Action: convert pi0.5 output to reBot commands
rebot_targets = translator.translate_action_chunk(
    franka_actions=pi05_output,        # (15, 8) from model
    franka_state=current_franka_q,     # (7,) virtual Franka state
    current_rebot_q=current_rebot_q,   # (6,) actual arm state
)
```

### IK Solver

Both Franka and reBot use damped least-squares CLIK (Closed-Loop Inverse Kinematics) via Pinocchio:
- Seeded from previous solution for temporal continuity
- Clamped to joint limits each iteration
- Falls back to last known state if solve fails

### Action Chunks

pi0.5 outputs **15 future timesteps** per inference call (~360ms compute). The control loop:
1. Queries model → gets 15 Franka joint deltas
2. Translates all 15 steps to reBot joint targets via FK/IK
3. Executes one target per control cycle (50 Hz)
4. After 15 steps, queries the model again

## Limitations & Known Issues

- **Zero-shot accuracy**: The model has never seen the reBot. Actions will be approximate — the EE-pose bridge helps, but some motions won't transfer perfectly (workspace boundaries, singularities, orientation limits differ).
- **IK failures**: If the Franka model commands a pose outside reBot's reachable workspace, IK will fail and the arm holds its last position.
- **Latency**: ~360ms per inference + IK overhead. Real-time at 50 Hz works because we chunk 15 steps ahead.
- **No gripper calibration**: Gripper value (0-1) passes through directly. May need scaling for reBot's specific gripper range.

## Next Steps

- [ ] Collect teleoperation demonstrations on reBot
- [ ] Fine-tune pi0.5 on reBot data (eliminates need for translation layer)
- [ ] Add workspace boundary checking and graceful recovery
- [ ] Calibrate gripper mapping
- [ ] Benchmark translation accuracy (FK/IK error accumulation)
