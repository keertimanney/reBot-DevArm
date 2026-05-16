#!/usr/bin/env bash
# Setup script for running pi0.5 on RTX 4090 with reBot arm
# Run this once to install everything.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== 1. Clone openpi ==="
if [ ! -d "$SCRIPT_DIR/openpi" ]; then
    GIT_LFS_SKIP_SMUDGE=1 git clone --recurse-submodules \
        https://github.com/Physical-Intelligence/openpi.git "$SCRIPT_DIR/openpi"
else
    echo "openpi already cloned, skipping."
fi

echo "=== 2. Install openpi (uv) ==="
cd "$SCRIPT_DIR/openpi"
GIT_LFS_SKIP_SMUDGE=1 uv sync

echo "=== 3. Install openpi-client ==="
pip install openpi-client

echo "=== 4. Install reBot arm control ==="
if [ ! -d "$SCRIPT_DIR/reBotArm_control_py" ]; then
    git clone https://github.com/vectorBH6/reBotArm_control_py.git "$SCRIPT_DIR/reBotArm_control_py"
fi
pip install -e "$SCRIPT_DIR/reBotArm_control_py"

echo "=== 5. Install additional dependencies ==="
pip install opencv-python numpy

echo ""
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Start the pi0.5 server:  ./serve.sh"
echo "  2. Run the arm controller:  python run_rebot.py"
