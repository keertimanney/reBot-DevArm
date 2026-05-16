#!/usr/bin/env bash
# Serve pi0.5 model on RTX 4090
# This downloads the checkpoint from GCS on first run (~3-5 min).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PORT="${PORT:-8000}"

cd "$SCRIPT_DIR/openpi"

echo "=== Serving pi0.5 (DROID config) on port $PORT ==="
echo "GPU memory required: ~8 GB (inference)"
echo "Checkpoint: gs://openpi-assets/checkpoints/pi05_droid"
echo ""

uv run scripts/serve_policy.py \
    policy:checkpoint \
    --policy.config=pi05_droid \
    --policy.dir=gs://openpi-assets/checkpoints/pi05_droid \
    --port="$PORT"
