"""
Run pi0.5 inference on the reBot arm.

Usage:
    1. Start the pi0.5 server first:  ./serve.sh
    2. Run this script:               python run_rebot.py --prompt "pick up the red block"

The script connects to the pi0.5 model server over WebSocket, reads camera
frames + joint states from the reBot arm, and sends predicted joint commands
back at 50 Hz.
"""

import argparse
import time

import cv2
import numpy as np
from openpi_client import websocket_client_policy, image_tools

from reBotArm_control_py.actuator import RobotArm


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CONTROL_HZ = 50            # Control loop frequency
ACTION_HORIZON = 10         # Steps per action chunk from pi0.5
NUM_JOINTS = 6              # reBot arm DOF (excluding gripper)
SERVER_HOST = "localhost"
SERVER_PORT = 8000


def resize_cam(frame: np.ndarray) -> np.ndarray:
    """Resize camera frame to 224x224 uint8 HWC as pi0.5 expects."""
    img = image_tools.resize_with_pad(frame, 224, 224)
    return image_tools.convert_to_uint8(img)


def build_observation(
    arm: RobotArm,
    cam: cv2.VideoCapture,
    wrist_cam: cv2.VideoCapture | None,
    prompt: str,
) -> dict:
    """Read sensors and build the observation dict for pi0.5."""
    pos, vel, torq = arm.get_state()

    ret, frame = cam.read()
    if not ret:
        raise RuntimeError("Failed to read from main camera")

    obs = {
        "observation/exterior_image_1_left": resize_cam(frame),
        "observation/joint_position": pos[:NUM_JOINTS].astype(np.float64),
        "observation/gripper_position": np.array([0.0]),  # TODO: read real gripper
        "prompt": prompt,
    }

    # Wrist camera (optional — zero-fill if not available)
    if wrist_cam is not None:
        ret_w, wrist_frame = wrist_cam.read()
        if ret_w:
            obs["observation/wrist_image_left"] = resize_cam(wrist_frame)
        else:
            obs["observation/wrist_image_left"] = np.zeros((224, 224, 3), dtype=np.uint8)
    else:
        obs["observation/wrist_image_left"] = np.zeros((224, 224, 3), dtype=np.uint8)

    return obs


def run(args):
    # -----------------------------------------------------------------------
    # Connect to pi0.5 server
    # -----------------------------------------------------------------------
    print(f"Connecting to pi0.5 server at {args.host}:{args.port} ...")
    policy = websocket_client_policy.WebsocketClientPolicy(
        host=args.host, port=args.port
    )
    print("Connected.")

    # -----------------------------------------------------------------------
    # Initialize arm
    # -----------------------------------------------------------------------
    print("Initializing reBot arm ...")
    arm = RobotArm()
    arm.connect()
    arm.enable()
    arm.mode_mit()
    print("Arm ready.")

    # -----------------------------------------------------------------------
    # Initialize cameras
    # -----------------------------------------------------------------------
    cam = cv2.VideoCapture(args.camera_id)
    if not cam.isOpened():
        raise RuntimeError(f"Cannot open camera {args.camera_id}")

    wrist_cam = None
    if args.wrist_camera_id is not None:
        wrist_cam = cv2.VideoCapture(args.wrist_camera_id)
        if not wrist_cam.isOpened():
            print(f"Warning: wrist camera {args.wrist_camera_id} not available, zero-filling.")
            wrist_cam = None

    # -----------------------------------------------------------------------
    # Control loop
    # -----------------------------------------------------------------------
    print(f"\nRunning policy: \"{args.prompt}\"")
    print(f"Control rate: {CONTROL_HZ} Hz | Action horizon: {ACTION_HORIZON}")
    print("Press Ctrl+C to stop.\n")

    dt = 1.0 / CONTROL_HZ
    action_chunk = None
    chunk_step = ACTION_HORIZON  # start exhausted so we query immediately

    try:
        while True:
            t0 = time.perf_counter()

            # Get new action chunk when current one is exhausted
            if chunk_step >= ACTION_HORIZON:
                obs = build_observation(arm, cam, wrist_cam, args.prompt)
                result = policy.infer(obs)
                action_chunk = result["actions"]  # (action_horizon, action_dim)
                chunk_step = 0

            # Extract joint targets from current step
            action = action_chunk[chunk_step]
            joint_targets = action[:NUM_JOINTS]
            chunk_step += 1

            # Send to arm
            arm.mit(pos=joint_targets)

            # Maintain loop rate
            elapsed = time.perf_counter() - t0
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nStopping ...")
    finally:
        arm.estop()
        arm.disable()
        arm.disconnect()
        cam.release()
        if wrist_cam is not None:
            wrist_cam.release()
        print("Shutdown complete.")


def main():
    parser = argparse.ArgumentParser(description="Run pi0.5 on reBot arm")
    parser.add_argument("--prompt", type=str, default="pick up the object",
                        help="Task instruction for the model")
    parser.add_argument("--host", type=str, default=SERVER_HOST,
                        help="pi0.5 server host")
    parser.add_argument("--port", type=int, default=SERVER_PORT,
                        help="pi0.5 server port")
    parser.add_argument("--camera-id", type=int, default=0,
                        help="Main camera device ID")
    parser.add_argument("--wrist-camera-id", type=int, default=None,
                        help="Wrist camera device ID (optional)")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
