"""
Run pi0.5 inference on the reBot arm with embodiment translation.

Uses end-effector pose as the common space between Franka (DROID) and reBot:
  Observation: reBot joints → FK → EE pose → Franka IK → pi0.5
  Action:      pi0.5 output → Franka FK → EE pose → reBot IK → reBot joints

Usage:
    1. Start the pi0.5 server first:  ./serve.sh
    2. Run this script:               python run_rebot.py --prompt "pick up the red block"
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from openpi_client import websocket_client_policy, image_tools

from reBotArm_control_py.actuator import RobotArm
from embodiment_translator import EmbodimentTranslator, FrankaModel


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CONTROL_HZ = 50            # Control loop frequency
ACTION_HORIZON = 15         # Steps per action chunk from pi0.5 (DROID config)
NUM_JOINTS = 6              # reBot arm DOF (excluding gripper)
SERVER_HOST = "localhost"
SERVER_PORT = 8000

# Default path to reBot URDF (relative to reBotArm_control_py install)
DEFAULT_URDF = str(
    Path(__file__).parent / "reBotArm_control_py" / "urdf"
    / "reBot-DevArm_fixend_description" / "urdf" / "reBot-DevArm_fixend.urdf"
)


def resize_cam(frame: np.ndarray) -> np.ndarray:
    """Resize camera frame to 224x224 uint8 HWC as pi0.5 expects."""
    img = image_tools.resize_with_pad(frame, 224, 224)
    return image_tools.convert_to_uint8(img)


def build_observation(
    arm: RobotArm,
    cam: cv2.VideoCapture,
    wrist_cam: cv2.VideoCapture | None,
    prompt: str,
    translator: EmbodimentTranslator,
) -> dict:
    """
    Read sensors and build the observation dict for pi0.5 (DROID format).

    Converts reBot joint positions → Franka joint positions via FK/IK
    so the model sees data in its native embodiment space.
    """
    pos, vel, torq = arm.get_state()
    rebot_q = pos[:NUM_JOINTS]

    # Translate reBot joints → Franka joints (7-DOF)
    franka_q = translator.rebot_to_franka(rebot_q)

    ret, frame = cam.read()
    if not ret:
        raise RuntimeError("Failed to read from main camera")

    obs = {
        "observation/exterior_image_1_left": resize_cam(frame),
        "observation/joint_position": franka_q.astype(np.float64),
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

    return obs, franka_q


def run(args):
    # -----------------------------------------------------------------------
    # Initialize embodiment translator
    # -----------------------------------------------------------------------
    urdf_path = args.urdf or DEFAULT_URDF
    print(f"Loading embodiment translator (URDF: {urdf_path}) ...")
    translator = EmbodimentTranslator(rebot_urdf_path=urdf_path)
    print("Translator ready.")
    print(f"  Franka home EE: {translator._franka_home_pos}")
    print(f"  reBot home EE:  {translator._rebot_home_pos}")
    print(f"  Scale factor:   {translator._scale_franka_to_rebot:.3f}")

    # -----------------------------------------------------------------------
    # Connect to pi0.5 server
    # -----------------------------------------------------------------------
    print(f"\nConnecting to pi0.5 server at {args.host}:{args.port} ...")
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
    rebot_targets = None
    chunk_step = ACTION_HORIZON  # start exhausted so we query immediately

    try:
        while True:
            t0 = time.perf_counter()

            # Get current reBot state
            current_pos, _, _ = arm.get_state()
            current_rebot_q = current_pos[:NUM_JOINTS]

            # Get new action chunk when current one is exhausted
            if chunk_step >= ACTION_HORIZON:
                obs, franka_state = build_observation(
                    arm, cam, wrist_cam, args.prompt, translator
                )
                result = policy.infer(obs)
                franka_actions = result["actions"]  # (horizon, 8)

                # Translate entire action chunk: Franka deltas → reBot joints
                rebot_targets = translator.translate_action_chunk(
                    franka_actions=franka_actions,
                    franka_state=franka_state,
                    current_rebot_q=current_rebot_q,
                )
                ACTION_HORIZON_ACTUAL = rebot_targets.shape[0]
                chunk_step = 0

            # Execute current step
            joint_targets = rebot_targets[chunk_step, :6]
            gripper_target = rebot_targets[chunk_step, 6]
            chunk_step += 1

            # Send to arm
            arm.mit(pos=joint_targets)
            # TODO: send gripper_target to gripper actuator

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
    parser = argparse.ArgumentParser(description="Run pi0.5 on reBot arm with embodiment translation")
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
    parser.add_argument("--urdf", type=str, default=None,
                        help="Path to reBot URDF file")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
