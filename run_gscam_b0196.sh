#!/usr/bin/env bash
set -eo pipefail

MODE="${1:-quality}"
DEVICE="${ARDUCAM_DEVICE:-/dev/video0}"
ROTATE="${ARDUCAM_ROTATE:-clockwise}"

case "$MODE" in
  quality)
    WIDTH=1920
    HEIGHT=1080
    FPS=30
    ;;
  realtime)
    WIDTH=1280
    HEIGHT=720
    FPS=30
    ;;
  max8mp)
    WIDTH=3264
    HEIGHT=2448
    FPS=15
    echo "[WARN] max8mp produced corrupt JPEG frames and a USB disconnect in WSL."
    echo "       Use it only for a short diagnostic, not normal live inference."
    ;;
  *)
    echo "Usage: $0 [quality|realtime|max8mp]"
    echo "  quality:  1920x1080 MJPG at 30 FPS (recommended)"
    echo "  realtime: 1280x720 MJPG at 30 FPS (lower load)"
    echo "  max8mp:   3264x2448 MJPG at 15 FPS (unstable in the current WSL test)"
    exit 2
    ;;
esac

case "$ROTATE" in
  none)
    ROTATE_PIPELINE=""
    ;;
  clockwise|counterclockwise|rotate-180)
    ROTATE_PIPELINE="videoflip method=$ROTATE ! "
    ;;
  *)
    echo "[ERROR] ARDUCAM_ROTATE must be none, clockwise, counterclockwise, or rotate-180"
    exit 2
    ;;
esac

# ROS setup files may inspect variables that have not been defined yet, so do
# not enable Bash's nounset (-u) while sourcing them.
source /opt/ros/jazzy/setup.bash
set -u

if [[ ! -e "$DEVICE" ]]; then
  echo "[ERROR] $DEVICE is missing. Reattach the Arducam with usbipd from Windows PowerShell."
  exit 1
fi

if command -v fuser >/dev/null 2>&1 && fuser "$DEVICE" >/dev/null 2>&1; then
  echo "[ERROR] $DEVICE is already in use:"
  fuser -v "$DEVICE" || true
  echo "Stop the existing gscam, gst-launch, ffmpeg, or camera viewer first."
  exit 1
fi

echo "[INFO] B0196 mode=$MODE, ${WIDTH}x${HEIGHT} MJPG @ ${FPS} FPS, rotate=$ROTATE"

ros2 run gscam gscam_node --ros-args \
  -r /camera/image_raw:=/camera1/image_raw \
  -r /camera/camera_info:=/camera1/camera_info \
  -p camera_name:=camera1 \
  -p frame_id:=camera1_optical_frame \
  -p image_encoding:=rgb8 \
  -p sync_sink:=false \
  -p use_gst_timestamps:=false \
  -p "gscam_config:=v4l2src device=$DEVICE io-mode=2 ! image/jpeg,width=$WIDTH,height=$HEIGHT,framerate=$FPS/1 ! jpegparse ! jpegdec ! ${ROTATE_PIPELINE}videoconvert ! video/x-raw,format=RGB"
