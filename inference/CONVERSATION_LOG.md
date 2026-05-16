# Pi0.5 + reBot Arm Integration — Session Notes

**Date:** 2026-05-16

## Summary

Explored how to run Physical Intelligence's **π0.5** model on an **RTX 4090** and control the **reBot B601 DM** arm with the model's predicted actions.

## Key Decisions

- **Training framework:** Using OpenPI (github.com/Physical-Intelligence/openpi) instead of LeRobot for π0.5 specifically
- **Inference GPU:** RTX 4090 locally (~8 GB VRAM for inference). AWS only needed for fine-tuning (A100/H100).
- **Architecture:** Pi0.5 server (WebSocket on port 8000) → Client control loop (50 Hz) → Damiao motors via CAN bus (MIT mode)
- **Base config:** `pi05_droid` — closest match to a single 6-DOF arm + gripper + cameras

## Architecture

```
pi0.5 Server (RTX 4090)  ←WebSocket→  run_rebot.py (control loop)  ←CAN bus→  Damiao Motors
```

## Files Created

| File | Purpose |
|------|---------|
| `setup.sh` | One-time install: clones openpi + reBotArm_control_py, installs deps |
| `serve.sh` | Starts pi0.5 model server on RTX 4090 (port 8000) |
| `run_rebot.py` | Main control loop — reads cameras + joints, queries pi0.5, sends commands to arm |

## How It Works

1. `serve.sh` loads the `pi05_droid` checkpoint and serves it over WebSocket
2. `run_rebot.py` connects to the server and the arm hardware
3. Each cycle (50 Hz): read camera frame (224x224) + joint positions → send to pi0.5 → get action chunk (10 future steps)
4. Execute one joint command per cycle via MIT mode (kp/kd spring-damper control)
5. When chunk is exhausted, query model again

## Hardware Context

- **Arm:** reBot B601 DM — 6 DOF + gripper, Damiao DM4310/DM4340P motors
- **Motor control:** `motorbridge` Python SDK over USB-to-CAN serial bridge
- **Cameras:** External (Intel D435i/D405) + optional wrist camera
- **Compute:** RTX 4090 for inference, Jetson/Pi optional for arm-side control

## Known TODOs

- Custom OpenPI config for reBot's exact 6-DOF + gripper layout (DROID config expects 7 joints)
- Wire up real gripper state (currently hardcoded to 0.0)
- Calibrate action interpretation — delta vs absolute joint positions
- Collect demonstration data and fine-tune on reBot-specific tasks
- Compute normalization stats for reBot joint ranges

## External Resources

- OpenPI repo: github.com/Physical-Intelligence/openpi
- reBot arm control: github.com/vectorBH6/reBotArm_control_py
- Motor SDK: motorbridge.seeedstudio.com
- Checkpoint: gs://openpi-assets/checkpoints/pi05_droid
