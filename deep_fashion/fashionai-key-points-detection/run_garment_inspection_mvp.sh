#!/usr/bin/env bash
set -e

source /opt/ros/jazzy/setup.bash
source /home/jiyoung/.venvs/sam2/bin/activate

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "[START] garment inspection MVP"
echo "[START] Python: $(command -v python3)"
echo "[START] Working directory: $SCRIPT_DIR"

python3 -u garment_inspection_roi_mvp.py "$@"
