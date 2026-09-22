#!/usr/bin/env bash
set -eo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <label> <1280x720.png> <1920x1080.png>"
  exit 2
fi

LABEL="$1"
IMAGE_720="$2"
IMAGE_1080="$3"

source /opt/ros/jazzy/setup.bash
source /home/jiyoung/.venvs/sam2/bin/activate
set -u

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ROOT="outputs/resolution_comparison/$LABEL"

python3 sam2_garment_keypoint_once.py \
  --input "$IMAGE_720" --auto-mask --no-show \
  --output-dir "$ROOT/1280x720/fashionai"
python3 flingbot_value_once.py \
  --input "$IMAGE_720" --no-show \
  --output-dir "$ROOT/1280x720/flingbot"

python3 sam2_garment_keypoint_once.py \
  --input "$IMAGE_1080" --auto-mask --no-show \
  --output-dir "$ROOT/1920x1080/fashionai"
python3 flingbot_value_once.py \
  --input "$IMAGE_1080" --no-show \
  --output-dir "$ROOT/1920x1080/flingbot"

python3 summarize_resolution_pair.py "$ROOT"
