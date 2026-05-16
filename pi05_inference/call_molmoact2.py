"""
Call MolmoAct2 SO101 from any machine.

Usage:
    pip install websockets msgpack numpy opencv-python pillow
    python call_molmoact2.py --prompt "pick up the red block"

This connects to the MolmoAct2 server over the public WebSocket endpoint
and returns joint targets for a 6-DOF arm (SO-100/101 format, in degrees).
"""

import argparse
import asyncio
import time

import cv2
import numpy as np
import msgpack
import websockets

# MolmoAct2 server endpoint
SERVER_URL = "wss://carter-missing-jeremy-punk.trycloudflare.com"

# SO-101 has 6 DOF (all values in degrees)
NUM_JOINTS = 6
ACTION_HORIZON = 30  # model returns 30 future steps


async def call_molmoact2(images: list, state: np.ndarray, prompt: str, url: str) -> np.ndarray:
    """
    Send observation to MolmoAct2, get back joint targets.

    Args:
        images: list of numpy arrays (HWC uint8, 224x224 each)
        state: current joint positions (6,) in degrees
        prompt: task instruction string
        url: WebSocket server URL

    Returns:
        actions: (30, 6) array of joint targets in degrees
    """
    # Pack observation as msgpack
    obs = {
        "prompt": prompt,
        "observation/joint_position": state.astype(np.float64).tobytes(),
    }

    # Add images
    for i, img in enumerate(images):
        # Resize to 224x224 if needed
        if img.shape[:2] != (224, 224):
            img = cv2.resize(img, (224, 224))
        key = f"observation/image_{i}" if i > 0 else "observation/image"
        obs[key] = img.astype(np.uint8).tobytes()

    packed = msgpack.packb(obs, use_bin_type=True)

    async with websockets.connect(url, open_timeout=30) as ws:
        await ws.send(packed)
        response = await asyncio.wait_for(ws.recv(), timeout=120)

    result = msgpack.unpackb(response, raw=False)
    actions = np.frombuffer(result["actions"], dtype=np.float32)
    actions = actions.reshape(-1, NUM_JOINTS)  # (30, 6)
    return actions


def main():
    parser = argparse.ArgumentParser(description="Call MolmoAct2 SO101")
    parser.add_argument("--prompt", type=str, default="pick up the object")
    parser.add_argument("--url", type=str, default=SERVER_URL)
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--state", type=float, nargs=6, default=[0, 0, 0, 0, 0, 0],
                        help="Current joint positions in degrees")
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
        print(f"No camera {args.camera_id}, using dummy image")
        frame = np.zeros((224, 224, 3), dtype=np.uint8)

    state = np.array(args.state, dtype=np.float32)

    print(f"Server: {args.url}")
    print(f"Prompt: {args.prompt}")
    print(f"State:  {state} (degrees)")
    print()

    # Call the model
    t0 = time.time()
    actions = asyncio.run(call_molmoact2(
        images=[frame],
        state=state,
        prompt=args.prompt,
        url=args.url,
    ))
    latency = time.time() - t0

    print(f"Inference latency: {latency:.2f}s")
    print(f"Action chunk: {actions.shape[0]} steps × {actions.shape[1]} joints")
    print(f"\nPredicted trajectory (degrees):")
    print(f"{'Step':>4s}  {'J1':>8s}  {'J2':>8s}  {'J3':>8s}  {'J4':>8s}  {'J5':>8s}  {'J6':>8s}")
    print("-" * 60)
    for i in range(min(10, len(actions))):
        print(f"{i:4d}  {actions[i,0]:8.2f}  {actions[i,1]:8.2f}  {actions[i,2]:8.2f}  "
              f"{actions[i,3]:8.2f}  {actions[i,4]:8.2f}  {actions[i,5]:8.2f}")
    if len(actions) > 10:
        print(f" ... ({len(actions) - 10} more steps)")


if __name__ == "__main__":
    main()
