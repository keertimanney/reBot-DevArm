"""
Call pi0.5 DROID from any machine.

Usage:
    pip install openpi-client opencv-python numpy
    python call_pi05.py --prompt "pick up the red block"

This connects to the pi0.5 server over the public WebSocket endpoint
and returns joint deltas for a 7-DOF arm (Franka/DROID format, in radians).
"""

import argparse
import time

import cv2
import numpy as np
from openpi_client import websocket_client_policy, image_tools

# pi0.5 server endpoint
SERVER_URL = "wss://awesome-throws-olive-pirates.trycloudflare.com"

# DROID: 7 joints + 1 gripper
NUM_JOINTS = 7
ACTION_HORIZON = 15  # model returns 15 future steps


def main():
    parser = argparse.ArgumentParser(description="Call pi0.5 DROID")
    parser.add_argument("--prompt", type=str, default="pick up the object")
    parser.add_argument("--url", type=str, default=SERVER_URL)
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--state", type=float, nargs=7,
                        default=[0, -0.161, 0, -2.445, 0, 2.227, 0.785],
                        help="Current joint positions in radians (Franka home)")
    args = parser.parse_args()

    # Capture a frame from camera (or use dummy if no camera)
    cap = cv2.VideoCapture(args.camera_id)
    if cap.isOpened():
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print("Camera read failed, using dummy image")
            frame = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            frame = cv2.resize(frame, (224, 224))
    else:
        print(f"No camera {args.camera_id}, using dummy image")
        frame = np.zeros((224, 224, 3), dtype=np.uint8)

    state = np.array(args.state, dtype=np.float64)

    print(f"Server: {args.url}")
    print(f"Prompt: {args.prompt}")
    print(f"State:  {state} (radians)")
    print()

    # Connect to server
    print("Connecting...")
    policy = websocket_client_policy.WebsocketClientPolicy(host=args.url)

    # Build DROID observation
    obs = {
        "observation/exterior_image_1_left": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(frame, 224, 224)
        ),
        "observation/wrist_image_left": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/joint_position": state,
        "observation/gripper_position": np.array([0.0]),
        "prompt": args.prompt,
    }

    # Call the model
    print("Running inference...")
    t0 = time.time()
    result = policy.infer(obs)
    latency = time.time() - t0

    actions = result["actions"]  # (15, 8)

    print(f"\nInference latency: {latency:.2f}s")
    print(f"Action chunk: {actions.shape[0]} steps × {actions.shape[1]} dims")
    print(f"  dims 0-6: joint position deltas (radians)")
    print(f"  dim 7:    gripper")
    print(f"\nPredicted trajectory (rad deltas):")
    print(f"{'Step':>4s}  {'J1':>7s}  {'J2':>7s}  {'J3':>7s}  {'J4':>7s}  {'J5':>7s}  {'J6':>7s}  {'J7':>7s}  {'Grip':>7s}")
    print("-" * 76)
    for i in range(min(10, len(actions))):
        vals = "  ".join(f"{v:7.4f}" for v in actions[i])
        print(f"{i:4d}  {vals}")
    if len(actions) > 10:
        print(f" ... ({len(actions) - 10} more steps)")


if __name__ == "__main__":
    main()
