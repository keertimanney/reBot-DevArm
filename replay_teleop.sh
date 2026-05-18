#!/usr/bin/env bash
set -euo pipefail

# ── ports ─────────────────────────────────────────────────────────────────────
FOLLOWER_PORT="/dev/cu.usbmodem00000000050C1"  # Damiao USB2CAN bridge

# ── dataset settings ──────────────────────────────────────────────────────────
REPO_ID="kanishk/rebot-devarm"
DATA_ROOT="./data"
EPISODE=0          # episode index to replay (0 = first recorded episode)

/Users/kanishk/miniforge3/bin/python3 "$(dirname "$0")/record_replay.py" \
  --episode "${EPISODE}" \
  --root "${DATA_ROOT}" \
  --repo-id "${REPO_ID}" \
  --port "${FOLLOWER_PORT}"
