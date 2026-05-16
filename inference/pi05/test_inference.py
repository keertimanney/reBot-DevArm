"""
Quick test: send a dummy observation to the pi0.5 server and print the action output.

Usage:
    1. Open SSH tunnel:  ssh -N -L 8000:localhost:8000 -p 15792 root@ssh3.vast.ai
    2. Run this script:  python test_inference.py
"""

import time
import numpy as np

from openpi_client import websocket_client_policy

HOST = "localhost"
PORT = 8000

print(f"Connecting to pi0.5 server at {HOST}:{PORT} ...")
policy = websocket_client_policy.WebsocketClientPolicy(host=HOST, port=PORT)
print("Connected!\n")

# Build a dummy observation matching DROID format
obs = {
    "observation/exterior_image_1_left": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
    "observation/wrist_image_left": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
    "observation/joint_position": np.zeros(7, dtype=np.float64),
    "observation/gripper_position": np.array([0.0], dtype=np.float64),
    "prompt": "pick up the red block",
}

print("Sending dummy observation ...")
t0 = time.time()
result = policy.infer(obs)
latency = time.time() - t0

actions = result["actions"]
print(f"\nInference latency: {latency:.3f}s")
print(f"Action chunk shape: {actions.shape}")
print(f"Action dim: {actions.shape[-1]} (expected: 8 for DROID = 7 joints + 1 gripper)")
print(f"Action horizon: {actions.shape[0]} steps")
print(f"\nFirst action step: {actions[0]}")
print(f"Last action step:  {actions[-1]}")
print(f"\nAction ranges:")
print(f"  min: {actions.min(axis=0)}")
print(f"  max: {actions.max(axis=0)}")
print("\npi0.5 is working!")
