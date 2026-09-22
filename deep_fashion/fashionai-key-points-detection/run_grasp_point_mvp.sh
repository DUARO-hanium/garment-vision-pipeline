#!/usr/bin/env bash
# 반품 의류 파지점 추출 MVP 실행 스크립트.
#
#   ./run_grasp_point_mvp.sh                     # ROS 카메라에서 한 장 (gscam이 떠 있어야 함)
#   ./run_grasp_point_mvp.sh real_test5.jpg      # 저장된 이미지 한 장
#   ./run_grasp_point_mvp.sh --demo              # real_test*.jpg 전부 + demo.mp4
#   ./run_grasp_point_mvp.sh --demo --fast       # FlingBot 생략 (빠른 확인)
#
# run_sam2_garment_once.sh / run_flingbot_value_once.sh 와 같은 규약을 따른다:
#   ROS 소싱 -> venv 활성화 순서, 소싱 중에는 set -u 를 켜지 않는다
#   (ROS setup 스크립트가 아직 정의되지 않은 변수를 참조한다).
set -eo pipefail

source /opt/ros/jazzy/setup.bash
source /home/jiyoung/.venvs/sam2/bin/activate
set -u

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

FLINGBOT_CKPT="${FLINGBOT_CKPT:-/mnt/d/문서/Users/23011/Documents/26-한이음/ros_projects/flingbot/flingbot.pth}"
SAM_MODEL="${SAM_MODEL:-/home/jiyoung/models/sam2_t.pt}"
TOPIC="${CAMERA_TOPIC:-/camera1/image_raw}"

COMMON=(--clothing-type "${CLOTHING_TYPE:-blouse}"
        --pair-mode "${PAIR_MODE:-top_bottom}"
        --sam-model "$SAM_MODEL"
        --flingbot-checkpoint "$FLINGBOT_CKPT")

DEMO=0
FAST=0
INPUT=""
for arg in "$@"; do
  case "$arg" in
    --demo) DEMO=1 ;;
    --fast) FAST=1 ;;
    -*) echo "알 수 없는 옵션: $arg" >&2; exit 2 ;;
    *) INPUT="$arg" ;;
  esac
done

[[ $FAST -eq 1 ]] && COMMON+=(--no-flingbot)

if [[ $DEMO -eq 1 ]]; then
  echo "[RUN] 배치 + 시연 영상 (ROS 불필요)"
  python3 grasp_point_mvp.py "${COMMON[@]}" \
    --input-dir . --pattern "real_test*.jpg" \
    --video demo.mp4 --video-hold-seconds 3 --no-show
elif [[ -n "$INPUT" ]]; then
  echo "[RUN] 단일 이미지: $INPUT (ROS 불필요)"
  python3 grasp_point_mvp.py "${COMMON[@]}" --input "$INPUT"
else
  # gscam이 실제로 퍼블리시 중인지 먼저 확인한다. 안 그러면 15초 타임아웃 뒤에야 실패한다.
  if ! ros2 topic list 2>/dev/null | grep -qx "$TOPIC"; then
    echo "[ERROR] $TOPIC 토픽이 없다. 다른 터미널에서 카메라를 먼저 띄울 것:"
    echo "        cd /mnt/d/문서/Users/23011/Documents/26-한이음/ros_projects"
    echo "        ./run_gscam_b0196.sh quality"
    exit 1
  fi
  echo "[RUN] ROS 토픽: $TOPIC"
  python3 grasp_point_mvp.py "${COMMON[@]}" --ros-topic "$TOPIC" --ros-timeout 15
fi
