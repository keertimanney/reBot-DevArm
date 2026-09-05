#!/usr/bin/env python3
"""Replay a recorded episode on the follower arm and save it as a new episode.

Usage:
    python record_replay.py                     # replay episode 0, append as new episode
    python record_replay.py --episode 1         # replay a different source episode
    python record_replay.py --root ./data --repo-id kanishk/rebot-devarm
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "reBotArm_control_py"))

FOLLOWER_PORT = "/dev/cu.usbmodem00000000050C1"
REPO_ID       = "kanishk/rebot-devarm"
DATA_ROOT     = "./data"
FPS           = 30


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a dataset episode while recording it as a new episode.")
    parser.add_argument("--episode", type=int, default=0, help="Source episode index to replay (default: 0)")
    parser.add_argument("--root", default=DATA_ROOT)
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--port", default=FOLLOWER_PORT)
    args = parser.parse_args()

    import lerobot_robot_seeed_b601  # noqa: F401 — registers robot type
    from lerobot_robot_seeed_b601 import SeeedB601DMFollower
    from lerobot_robot_seeed_b601.config_seeed_b601_dm_follower import SeeedB601DMFollowerConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    # ── load source dataset ───────────────────────────────────────────────────
    print(f"Loading dataset from {args.root} …")
    dataset = LeRobotDataset(args.repo_id, root=args.root)

    ep_data = dataset.hf_dataset.filter(lambda row: row["episode_index"] == args.episode)
    if len(ep_data) == 0:
        sys.exit(f"Episode {args.episode} not found in dataset.")

    actions = [dict(zip(dataset.features["action"]["names"], frame))
               for frame in ep_data["action"]]
    total_frames = len(actions)
    print(f"Episode {args.episode}: {total_frames} frames at {dataset.fps} fps "
          f"({total_frames / dataset.fps:.1f}s)")

    # ── connect robot ─────────────────────────────────────────────────────────
    config = SeeedB601DMFollowerConfig(id="follower1", port=args.port, can_adapter="damiao")
    robot = SeeedB601DMFollower(config)
    print(f"Connecting to follower on {args.port} …")
    robot.connect()
    print("Connected.")

    # ── replay + record ───────────────────────────────────────────────────────
    dt = 1.0 / dataset.fps
    print(f"\nReplaying episode {args.episode} and recording as episode {dataset.num_episodes} …")
    print("Press Ctrl+C to abort.\n")

    recorded_frames: list[dict] = []
    try:
        t_start = time.perf_counter()
        for i, action in enumerate(actions):
            t0 = time.perf_counter()

            obs = robot.get_observation()
            robot.send_action(action)

            recorded_frames.append({
                "action": [action[k] for k in dataset.features["action"]["names"]],
                "observation.state": [obs.get(k, 0.0) for k in dataset.features["action"]["names"]],
                "timestamp": i / dataset.fps,
            })

            elapsed = (i + 1) / dataset.fps
            print(f"\r  frame {i+1}/{total_frames}  t={elapsed:.1f}s", end="", flush=True)

            rem = dt - (time.perf_counter() - t0)
            if rem > 0:
                time.sleep(rem)

        print(f"\nReplay done ({time.perf_counter() - t_start:.1f}s).")

    except KeyboardInterrupt:
        print("\nAborted — discarding partial episode.")
        robot.disconnect()
        return
    finally:
        robot.disconnect()
        print("Robot disconnected.")

    if not recorded_frames:
        print("No frames recorded.")
        return

    # ── append new episode to dataset ─────────────────────────────────────────
    print("Saving new episode …")
    for frame in recorded_frames:
        dataset.add_frame(frame)

    dataset.save_episode(episode_data={"task": "pick and place"})
    print(f"Saved as episode {dataset.num_episodes - 1}. "
          f"Dataset now has {dataset.num_episodes} episode(s).")


if __name__ == "__main__":
    main()
