#!/usr/bin/env bash
set -eo pipefail

source /opt/ros/jazzy/setup.bash
source /home/jiyoung/.venvs/sam2/bin/activate
set -u

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

python3 sam2_garment_keypoint_once.py \
  --ros-topic /camera1/image_raw \
  --clothing-type blouse \
  "$@"
