#!/usr/bin/env bash
# Deploy pi0.5 server on a Vast.ai RTX 4090 instance.
#
# Usage:
#   ./vast_deploy.sh          # find cheapest 4090, create instance, deploy
#   ./vast_deploy.sh <id>     # use a specific offer ID
#
# After deployment, run locally:
#   python run_rebot.py --host <VAST_IP> --port 8000

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
IMAGE="nvidia/cuda:12.4.0-devel-ubuntu22.04"
DISK_GB=64

# -----------------------------------------------------------------------
# 1. Pick an offer
# -----------------------------------------------------------------------
if [ -n "${1:-}" ]; then
    OFFER_ID="$1"
    echo "Using offer ID: $OFFER_ID"
else
    echo "=== Finding cheapest RTX 4090 ==="
    OFFER_ID=$(vastai search offers \
        'gpu_name=RTX_4090 num_gpus=1 inet_down>200 reliability>0.95 disk_space>=50 dph<0.60' \
        -o 'dph' --raw 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin)[0]['id'])")
    echo "Selected offer: $OFFER_ID"
fi

# -----------------------------------------------------------------------
# 2. Create the instance
# -----------------------------------------------------------------------
echo "=== Creating instance (image: $IMAGE, disk: ${DISK_GB}GB) ==="
RESULT=$(vastai create instance "$OFFER_ID" \
    --image "$IMAGE" \
    --disk "$DISK_GB" \
    --onstart-cmd "apt-get update && apt-get install -y git curl && curl -LsSf https://astral.sh/uv/install.sh | sh" \
    --raw 2>/dev/null)

INSTANCE_ID=$(echo "$RESULT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('new_contract', d.get('id', '')))")
echo "Instance ID: $INSTANCE_ID"

# -----------------------------------------------------------------------
# 3. Wait for instance to be running
# -----------------------------------------------------------------------
echo "=== Waiting for instance to start ==="
for i in $(seq 1 60); do
    STATUS=$(vastai show instance "$INSTANCE_ID" --raw 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin).get('actual_status',''))" 2>/dev/null || echo "")
    if [ "$STATUS" = "running" ]; then
        echo "Instance is running!"
        break
    fi
    echo "  Status: ${STATUS:-starting} (attempt $i/60)"
    sleep 10
done

# -----------------------------------------------------------------------
# 4. Get connection info
# -----------------------------------------------------------------------
echo "=== Getting connection details ==="
INSTANCE_INFO=$(vastai show instance "$INSTANCE_ID" --raw 2>/dev/null)
SSH_HOST=$(echo "$INSTANCE_INFO" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('ssh_host',''))")
SSH_PORT=$(echo "$INSTANCE_INFO" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('ssh_port',''))")

echo ""
echo "SSH: ssh -p $SSH_PORT root@$SSH_HOST"
echo ""

# -----------------------------------------------------------------------
# 5. Deploy pi0.5 on the instance
# -----------------------------------------------------------------------
echo "=== Deploying pi0.5 server ==="
ssh -o StrictHostKeyChecking=no -p "$SSH_PORT" "root@$SSH_HOST" bash -s <<'REMOTE_SCRIPT'
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"

# Wait for uv to be installed (from onstart-cmd)
for i in $(seq 1 30); do
    if command -v uv &>/dev/null; then break; fi
    echo "Waiting for uv install... ($i)"
    sleep 5
done

echo "=== Cloning openpi ==="
if [ ! -d /root/openpi ]; then
    GIT_LFS_SKIP_SMUDGE=1 git clone --recurse-submodules \
        https://github.com/Physical-Intelligence/openpi.git /root/openpi
fi

echo "=== Installing openpi ==="
cd /root/openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync

echo "=== Starting pi0.5 server ==="
# Run in background with nohup so it persists after SSH disconnects
nohup uv run scripts/serve_policy.py \
    policy:checkpoint \
    --policy.config=pi05_droid \
    --policy.dir=gs://openpi-assets/checkpoints/pi05_droid \
    --port=8000 \
    > /root/pi05_server.log 2>&1 &

echo "Server PID: $!"
echo "Log: /root/pi05_server.log"
echo "Waiting for server to load model..."
sleep 10
tail -20 /root/pi05_server.log
REMOTE_SCRIPT

echo ""
echo "============================================"
echo "  pi0.5 deployed!"
echo "============================================"
echo ""
echo "Instance ID:  $INSTANCE_ID"
echo "SSH:          ssh -p $SSH_PORT root@$SSH_HOST"
echo "Server log:   ssh -p $SSH_PORT root@$SSH_HOST tail -f /root/pi05_server.log"
echo ""
echo "To forward port 8000 locally:"
echo "  ssh -N -L 8000:localhost:8000 -p $SSH_PORT root@$SSH_HOST"
echo ""
echo "Then run:"
echo "  python run_rebot.py --host localhost --port 8000 --prompt 'pick up the block'"
echo ""
echo "To destroy when done:"
echo "  vastai destroy instance $INSTANCE_ID"
