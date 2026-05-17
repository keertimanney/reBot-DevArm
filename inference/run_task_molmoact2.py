"""
Run MolmoAct2 on the reBot arm with in-context demonstration.

Loads demo frames from a HuggingFace dataset and passes them alongside
the current camera view to MolmoAct2. The model sees the demo frames as
visual context, enabling one-shot task learning.

Usage:
    conda activate seed_env
    python run_task_molmoact2.py \
        --prompt "pick the purple box and place it on the black tape" \
        --demo-dataset andlyu/rebot_hackathon_v4 \
        --host wss://higher-jurisdiction-warrant-players.trycloudflare.com
"""

import argparse
import asyncio
import time
import os

import cv2
import numpy as np
import msgpack
import websockets


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CONTROL_HZ = 5             # MolmoAct2 is slower, ~2s per inference
NUM_JOINTS = 6             # SO-101 / reBot DOF
SERVER_URL = "wss://higher-jurisdiction-warrant-players.trycloudflare.com"


def load_demo_frames(repo_id: str, num_frames: int = 5):
    """
    Load evenly-spaced demo frames from the HuggingFace dataset.
    Returns list of (image_array, joint_state) tuples.
    """
    from huggingface_hub import hf_hub_download
    import pandas as pd
    import av

    print(f"Loading demonstration from {repo_id}...")

    # Download video and joint data
    video_path = hf_hub_download(
        repo_id=repo_id,
        filename="videos/observation.images.front/chunk-000/file-000.mp4",
        repo_type="dataset",
    )
    parquet_path = hf_hub_download(
        repo_id=repo_id,
        filename="data/chunk-000/file-000.parquet",
        repo_type="dataset",
    )

    # Load video frames
    container = av.open(video_path)
    all_frames = [f.to_ndarray(format='rgb24') for f in container.decode(video=0)]

    # Load joint data
    df = pd.read_parquet(parquet_path)
    all_states = np.array(df['observation.state'].tolist())

    # Sample evenly-spaced frames
    n = len(all_frames)
    indices = np.linspace(0, n - 1, num_frames, dtype=int)

    demo_data = []
    for idx in indices:
        frame = cv2.resize(all_frames[idx], (224, 224))
        state = all_states[idx]
        demo_data.append((frame, state))

    print(f"  Loaded {len(demo_data)} demo frames from {n} total")
    print(f"  Frame indices: {indices.tolist()}")
    return demo_data


async def call_molmoact2(url: str, demo_frames: list, current_img: np.ndarray,
                          state: np.ndarray, prompt: str) -> np.ndarray:
    """
    Call MolmoAct2 with demo frames as in-context examples + current observation.

    The model receives:
    - demo/image_0 through demo/image_N: demonstration frames (visual context)
    - observation/image: current camera view
    - observation/joint_position: current state
    - prompt: task instruction
    """
    obs = {
        "prompt": prompt,
        "observation/joint_position": state.astype(np.float64).tobytes(),
        "observation/image": current_img.astype(np.uint8).tobytes(),
    }

    # Add demo frames as visual context
    for i, (frame, _) in enumerate(demo_frames):
        obs[f"demo/image_{i}"] = frame.astype(np.uint8).tobytes()

    packed = msgpack.packb(obs, use_bin_type=True)

    async with websockets.connect(url, open_timeout=30, max_size=100 * 1024 * 1024) as ws:
        await ws.send(packed)
        response = await asyncio.wait_for(ws.recv(), timeout=120)

    result = msgpack.unpackb(response, raw=False)
    actions = np.frombuffer(result["actions"], dtype=np.float32)
    actions = actions.reshape(-1, NUM_JOINTS)
    return actions


def run(args):
    # -----------------------------------------------------------------------
    # Load demonstration frames
    # -----------------------------------------------------------------------
    demo_frames = load_demo_frames(args.demo_dataset, num_frames=args.num_demo_frames)

    # -----------------------------------------------------------------------
    # Initialize cameras
    # -----------------------------------------------------------------------
    print("Opening cameras...")
    front_cam = cv2.VideoCapture(1)  # front = index 1
    if not front_cam.isOpened():
        raise RuntimeError("Cannot open front camera (index 1)")
    for _ in range(10):
        front_cam.read()
    print("Camera ready.")

    # -----------------------------------------------------------------------
    # Initialize arm
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

        for i, m in enumerate(motors):
            mode = Mode.FORCE_POS if i == 6 else Mode.POS_VEL
            for attempt in range(5):
                try:
                    m.ensure_mode(mode, timeout_ms=10000)
                    print(f"  Motor {i+1} ready")
                    break
                except:
                    if attempt == 4:
                        print(f"  Motor {i+1} failed, skipping")
                    time.sleep(1)
        arm_connected = True
        print("Arm connected.")

        # Read current state
        for m in motors:
            m.request_feedback()
        time.sleep(0.1)
        ctrl.poll_feedback_once()
        current_state = np.array([m.get_state().pos for m in motors[:6]])
        current_state_deg = np.degrees(current_state)
    except Exception as e:
        print(f"Arm not connected ({e}). Running inference only.")
        arm_connected = False
        ctrl = None
        current_state_deg = demo_frames[0][1][:6]

    # -----------------------------------------------------------------------
    # Control loop
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  Task: {args.prompt}")
    print(f"  Demo frames: {args.num_demo_frames}")
    print(f"  Max steps: {args.max_steps}")
    print(f"  Arm connected: {arm_connected}")
    print(f"{'='*60}")
    print("Running... Press Ctrl+C to stop.\n")

    action_chunk = None
    chunk_step = 0

    try:
        for step in range(args.max_steps):
            t0 = time.perf_counter()

            # Get new action chunk when needed
            if action_chunk is None or chunk_step >= len(action_chunk):
                # Capture current frame
                ret, frame = front_cam.read()
                if not ret:
                    print("Camera read failed!")
                    break
                current_img = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (224, 224))

                # Call MolmoAct2 with demo context
                t_infer = time.perf_counter()
                action_chunk = asyncio.run(call_molmoact2(
                    url=args.host,
                    demo_frames=demo_frames,
                    current_img=current_img,
                    state=current_state_deg[:NUM_JOINTS],
                    prompt=args.prompt,
                ))
                infer_time = time.perf_counter() - t_infer
                chunk_step = 0

                print(f"Step {step}: inference {infer_time:.2f}s, chunk {action_chunk.shape}")
                print(f"  Actions[0]: {action_chunk[0].round(2)}")

            # Get target from chunk (MolmoAct2 outputs absolute positions in degrees)
            target_deg = action_chunk[chunk_step]
            chunk_step += 1

            # Send to arm
            if arm_connected:
                target_rad = np.radians(target_deg)
                vlim = np.array([2.0, 2.0, 2.0, 3.0, 3.0, 3.0])
                for i, m in enumerate(motors[:6]):
                    m.send_pos_vel(target_rad[i], vlim[i])

            current_state_deg[:NUM_JOINTS] = target_deg

            # Maintain loop rate
            elapsed = time.perf_counter() - t0
            sleep_time = (1.0 / CONTROL_HZ) - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        if arm_connected:
            ctrl.disable_all()
            ctrl.close()
        front_cam.release()
        print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Run MolmoAct2 with in-context demo on reBot arm")
    parser.add_argument("--prompt", type=str,
                        default="pick the purple box and place it on the black tape")
    parser.add_argument("--demo-dataset", type=str, default="andlyu/rebot_hackathon_v4")
    parser.add_argument("--host", type=str, default=SERVER_URL)
    parser.add_argument("--num-demo-frames", type=int, default=5,
                        help="Number of evenly-spaced demo frames to use as context")
    parser.add_argument("--max-steps", type=int, default=100)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
