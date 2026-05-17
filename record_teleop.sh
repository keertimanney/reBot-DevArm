#!/usr/bin/env bash
set -euo pipefail

# ── ports ─────────────────────────────────────────────────────────────────────
LEADER_PORT="/dev/cu.usbserial-110"         # reBot Arm 102
FOLLOWER_PORT="/dev/cu.usbmodem00000000050C1"  # Damiao USB2CAN bridge

# ── dataset settings ──────────────────────────────────────────────────────────
REPO_ID="kanishk/rebot-devarm"
DATA_ROOT="./data"
TASK="pick and place"
NUM_EPISODES=1
FPS=30
EPISODE_TIME_S=30
RESET_TIME_S=5

# Clear stale data from a previous failed run (no episodes = safe to wipe)
if [ -d "${DATA_ROOT}" ] && [ ! -d "${DATA_ROOT}/episodes" ]; then
  echo "Removing leftover empty dataset at ${DATA_ROOT}..."
  rm -rf "${DATA_ROOT}"
fi

lerobot-record \
  --teleop.type=rebot_arm_102_leader \
  --teleop.id=leader1 \
  --teleop.port="${LEADER_PORT}" \
  --robot.type=seeed_b601_dm_follower \
  --robot.id=follower1 \
  --robot.port="${FOLLOWER_PORT}" \
  --robot.can_adapter=damiao \
  --dataset.repo_id="${REPO_ID}" \
  --dataset.root="${DATA_ROOT}" \
  --dataset.single_task="${TASK}" \
  --dataset.num_episodes="${NUM_EPISODES}" \
  --dataset.fps="${FPS}" \
  --dataset.episode_time_s="${EPISODE_TIME_S}" \
  --dataset.reset_time_s="${RESET_TIME_S}" \
  --dataset.push_to_hub=false
