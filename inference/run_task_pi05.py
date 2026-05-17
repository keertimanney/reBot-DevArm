"""
Run pi0.5 on the reBot arm to perform a task with in-context demonstration.

Uses the recorded demonstration from HuggingFace as reference for workspace understanding,
then runs pi0.5 DROID inference with the embodiment translator.

Usage:
    conda activate seed_env
    python run_task_pi05.py \
        --prompt "pick the purple box and place it on the black tape" \
        --demo-dataset andlyu/rebot_hackathon_v4 \
        --host wss://awesome-throws-olive-pirates.trycloudflare.com

Note: pi0.5 does not natively support in-context learning from video demonstrations
at inference time. The demonstration data is used here to:
1. Establish the workspace reference (joint ranges, camera perspectives)
2. Provide the initial joint state for the embodiment translator
The actual policy is zero-shot conditioned on the text prompt.
"""

import argparse
import time
import os
import sys

import cv2
import numpy as np
from openpi_client import websocket_client_policy, image_tools

# Add shared dir to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'shared'))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CONTROL_HZ = 15            # Control loop frequency (limited by inference latency)
NUM_REBOT_JOINTS = 6       # reBot arm DOF (excluding gripper)
NUM_FRANKA_JOINTS = 7      # Franka/DROID DOF (excluding gripper)
SERVER_URL = "wss://awesome-throws-olive-pirates.trycloudflare.com"


def load_demo_context(repo_id: str):
    """Load demonstration data from HuggingFace for workspace reference."""
    from huggingface_hub import hf_hub_download
    import pandas as pd

    print(f"Loading demonstration from {repo_id}...")

    # Download the joint data
    parquet_path = hf_hub_download(
        repo_id=repo_id,
        filename="data/chunk-000/file-000.parquet",
        repo_type="dataset",
    )
    df = pd.read_parquet(parquet_path)

    # Extract joint statistics for workspace understanding
    states = np.array(df['observation.state'].tolist())
    actions = np.array(df['action'].tolist())

    demo_info = {
        "num_frames": len(df),
        "joint_mean": states.mean(axis=0),
        "joint_std": states.std(axis=0),
        "joint_min": states.min(axis=0),
        "joint_max": states.max(axis=0),
        "action_mean": actions.mean(axis=0),
        "action_std": actions.std(axis=0),
        "initial_state": states[0],
        "final_state": states[-1],
    }

    print(f"  Loaded {demo_info['num_frames']} frames")
    print(f"  Joint ranges: {demo_info['joint_min'].round(1)} to {demo_info['joint_max'].round(1)}")
    print(f"  Initial state: {demo_info['initial_state'].round(2)}")

    return demo_info


def rebot_to_franka_simple(rebot_state: np.ndarray) -> np.ndarray:
    """
    Simple mapping from reBot 6-DOF state (degrees) to Franka 7-DOF state (radians).

    Maps via normalized joint positions:
    - reBot joint ranges → [0, 1] → Franka joint ranges
    """
    # reBot joint limits (degrees, from calibration)
    rebot_limits = np.array([
        [-145, 145],   # shoulder_pan
        [-170, 1],     # shoulder_lift
        [-200, 1],     # elbow_flex
        [-80, 90],     # wrist_flex
        [-90, 90],     # wrist_yaw
        [-90, 90],     # wrist_roll
    ])

    # Franka joint limits (radians)
    franka_limits = np.array([
        [-2.8973, 2.8973],   # joint1
        [-1.7628, 1.7628],   # joint2
        [-2.8973, 2.8973],   # joint3
        [-3.0718, -0.0698],  # joint4
        [-2.8973, 2.8973],   # joint5
        [-0.0175, 3.7525],   # joint6
        [-2.8973, 2.8973],   # joint7
    ])

    # Normalize reBot state to [0, 1]
    state_6 = rebot_state[:6]
    normalized = (state_6 - rebot_limits[:, 0]) / (rebot_limits[:, 1] - rebot_limits[:, 0])
    normalized = np.clip(normalized, 0, 1)

    # Map to Franka space (use first 6 joints, set 7th to neutral)
    franka_state = np.zeros(7)
    for i in range(6):
        franka_state[i] = franka_limits[i, 0] + normalized[i] * (franka_limits[i, 1] - franka_limits[i, 0])
    franka_state[6] = 0.785  # Franka joint 7 neutral

    return franka_state


def franka_to_rebot_simple(franka_action: np.ndarray, current_rebot_state: np.ndarray) -> np.ndarray:
    """
    Convert Franka action deltas back to reBot joint targets (degrees).

    Takes the delta from pi0.5, scales it proportionally, and applies to current state.
    """
    # Scale factor: Franka deltas (radians) → reBot deltas (degrees)
    # Franka range ~5.8 rad, reBot range ~290 deg → rough scale
    scale = np.array([50, 30, 35, 30, 30, 30])  # degrees per radian, per joint

    # Take first 6 joint deltas, convert to degrees
    delta_rad = franka_action[:6]
    delta_deg = delta_rad * scale

    # Apply to current state
    new_state = current_rebot_state[:6] + delta_deg

    # Clamp to reBot limits
    rebot_limits_min = np.array([-145, -170, -200, -80, -90, -90])
    rebot_limits_max = np.array([145, 1, 1, 90, 90, 90])
    new_state = np.clip(new_state, rebot_limits_min, rebot_limits_max)

    return new_state


def run(args):
    # -----------------------------------------------------------------------
    # Load demonstration context
    # -----------------------------------------------------------------------
    demo_info = load_demo_context(args.demo_dataset)

    # -----------------------------------------------------------------------
    # Connect to pi0.5 server
    # -----------------------------------------------------------------------
    print(f"\nConnecting to pi0.5 at {args.host}...")
    policy = websocket_client_policy.WebsocketClientPolicy(host=args.host)
    print("Connected.")

    # -----------------------------------------------------------------------
    # Initialize cameras
    # -----------------------------------------------------------------------
    print("Opening cameras...")
    front_cam = cv2.VideoCapture(1)  # front = index 1
    wrist_cam = cv2.VideoCapture(0)  # wrist = index 0

    if not front_cam.isOpened():
        raise RuntimeError("Cannot open front camera (index 1)")
    if not wrist_cam.isOpened():
        print("Warning: wrist camera (index 0) not available, zero-filling")
        wrist_cam = None

    # Warm up cameras
    for _ in range(10):
        front_cam.read()
        if wrist_cam:
            wrist_cam.read()
    print("Cameras ready.")

    # -----------------------------------------------------------------------
    # Initialize arm connection (via motorbridge)
    # -----------------------------------------------------------------------
    try:
        from motorbridge import Controller, Mode
        print("Connecting to reBot arm...")
        ctrl = Controller.from_dm_serial('/dev/tty.usbmodem00000000050C1', 921600)
        ids = [(1, 17, '4340P'), (2, 18, '4340P'), (3, 19, '4340P'),
               (4, 20, '4310'), (5, 21, '4310'), (6, 22, '4310'), (7, 23, '4310')]
        motors = []
        for mid, fid, model in ids:
            motors.append(ctrl.add_damiao_motor(mid, fid, model))
        ctrl.enable_all()
        time.sleep(1)
        print("Motors enabled.")

        # Set mode: POS_VEL for joints, FORCE_POS for gripper
        for i, m in enumerate(motors):
            mode = Mode.FORCE_POS if i == 6 else Mode.POS_VEL
            for attempt in range(5):
                try:
                    m.ensure_mode(mode, timeout_ms=10000)
                    print(f"  Motor {i+1} ready ({mode})")
                    break
                except Exception as e:
                    if attempt == 4:
                        print(f"  Motor {i+1} failed after 5 attempts, skipping")
                        break
                    print(f"  Motor {i+1} retry {attempt+1}...")
                    time.sleep(1)
        print("Arm connected and ready.")
        arm_connected = True
    except Exception as e:
        print(f"Warning: Arm not connected ({e}). Running inference only (no motor commands).")
        arm_connected = False
        ctrl = None

    # -----------------------------------------------------------------------
    # Get initial state
    # -----------------------------------------------------------------------
    if arm_connected:
        # Read current joint positions
        for m in motors:
            m.request_feedback()
        time.sleep(0.1)
        ctrl.poll_feedback_once()
        current_state = np.array([m.get_state().pos for m in motors[:6]])  # radians
        current_state_deg = np.degrees(current_state)
    else:
        # Use demo initial state
        current_state_deg = demo_info['initial_state'][:6]

    print(f"Initial state (deg): {current_state_deg.round(1)}")

    # -----------------------------------------------------------------------
    # Control loop
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  Task: {args.prompt}")
    print(f"  Control rate: {CONTROL_HZ} Hz")
    print(f"  Max steps: {args.max_steps}")
    print(f"{'='*60}")
    print("Running... Press Ctrl+C to stop.\n")

    # Convert initial reBot state to Franka state for the model
    franka_state = rebot_to_franka_simple(current_state_deg)

    dt = 1.0 / CONTROL_HZ
    action_chunk = None
    chunk_step = 0
    chunk_size = 15  # DROID horizon

    try:
        for step in range(args.max_steps):
            t0 = time.perf_counter()

            # Get new action chunk when needed
            if action_chunk is None or chunk_step >= chunk_size:
                # Read cameras
                ret_f, front_frame = front_cam.read()
                if not ret_f:
                    print("Front camera read failed!")
                    break

                front_img = cv2.resize(front_frame, (224, 224))
                front_img = cv2.cvtColor(front_img, cv2.COLOR_BGR2RGB)

                if wrist_cam:
                    ret_w, wrist_frame = wrist_cam.read()
                    wrist_img = cv2.resize(wrist_frame, (224, 224)) if ret_w else np.zeros((224, 224, 3), dtype=np.uint8)
                    wrist_img = cv2.cvtColor(wrist_img, cv2.COLOR_BGR2RGB) if ret_w else wrist_img
                else:
                    wrist_img = np.zeros((224, 224, 3), dtype=np.uint8)

                # Build DROID observation
                obs = {
                    "observation/exterior_image_1_left": image_tools.convert_to_uint8(front_img),
                    "observation/wrist_image_left": image_tools.convert_to_uint8(wrist_img),
                    "observation/joint_position": franka_state.astype(np.float64),
                    "observation/gripper_position": np.array([0.0]),
                    "prompt": args.prompt,
                }

                # Call pi0.5
                t_infer = time.perf_counter()
                result = policy.infer(obs)
                infer_time = time.perf_counter() - t_infer

                action_chunk = result["actions"]  # (15, 8)
                chunk_step = 0
                chunk_size = action_chunk.shape[0]

                if step == 0:
                    print(f"First inference: {infer_time:.2f}s (JIT warmup)")
                else:
                    print(f"Step {step}: inference {infer_time:.3f}s, chunk shape {action_chunk.shape}")

            # Get current action
            action = action_chunk[chunk_step]
            chunk_step += 1

            # Convert Franka delta → reBot target
            rebot_target = franka_to_rebot_simple(action, current_state_deg)
            gripper_target = action[7] if len(action) > 7 else 0.0

            # Update virtual Franka state (accumulate deltas)
            franka_state = franka_state + action[:7]

            # Send to arm
            if arm_connected:
                # Convert degrees to radians for POS_VEL command
                target_rad = np.radians(rebot_target)
                vlim = np.array([2.0, 2.0, 2.0, 3.0, 3.0, 3.0])  # rad/s velocity limits
                for i, m in enumerate(motors[:6]):
                    m.send_pos_vel(target_rad[i], vlim[i])

            current_state_deg = rebot_target

            if step % 15 == 0:
                print(f"  Step {step}: target={rebot_target.round(1)} gripper={gripper_target:.2f}")

            # Maintain loop rate
            elapsed = time.perf_counter() - t0
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        if arm_connected:
            ctrl.disable_all()
            ctrl.close()
        front_cam.release()
        if wrist_cam:
            wrist_cam.release()
        print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Run pi0.5 task on reBot arm")
    parser.add_argument("--prompt", type=str,
                        default="pick the purple box and place it on the black tape")
    parser.add_argument("--demo-dataset", type=str, default="andlyu/rebot_hackathon_v4",
                        help="HuggingFace dataset with demonstration recording")
    parser.add_argument("--host", type=str, default=SERVER_URL,
                        help="pi0.5 WebSocket server URL")
    parser.add_argument("--max-steps", type=int, default=300,
                        help="Maximum control steps (300 = ~20s at 15Hz)")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
