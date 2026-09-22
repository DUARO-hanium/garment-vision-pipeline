#!/usr/bin/env bash
set -euo pipefail

DEVICE="${ARDUCAM_DEVICE:-/dev/video0}"
FOCUS_VALUE="${1:-208}"

if [[ ! -e "$DEVICE" ]]; then
  echo "[ERROR] $DEVICE is missing. Attach the Arducam to WSL with usbipd first."
  exit 1
fi

if ! command -v v4l2-ctl >/dev/null 2>&1; then
  echo "[ERROR] v4l2-ctl is missing. Install it with: sudo apt install v4l-utils"
  exit 1
fi

if command -v fuser >/dev/null 2>&1 && fuser "$DEVICE" >/dev/null 2>&1; then
  echo "[ERROR] $DEVICE is already in use. Stop gscam/rqt/ffmpeg first with Ctrl+C."
  fuser -v "$DEVICE" || true
  exit 1
fi

CONTROLS="$(v4l2-ctl -d "$DEVICE" --list-ctrls-menus 2>&1)"

has_control() {
  grep -Eq "^[[:space:]]*$1[[:space:]]" <<<"$CONTROLS"
}

set_control_if_supported() {
  local name="$1"
  local value="$2"
  if has_control "$name"; then
    echo "[SET] $name=$value"
    v4l2-ctl -d "$DEVICE" --set-ctrl="$name=$value"
  else
    echo "[SKIP] $name is not exposed by this camera/driver"
  fi
}

echo "[INFO] Applying B0196 image-quality starting values to $DEVICE"

# Datasheet Video Proc Amp starting values. A control is changed only when the
# Linux UVC driver actually exposes it.
set_control_if_supported brightness 0
set_control_if_supported contrast 32
set_control_if_supported saturation 64
set_control_if_supported sharpness 3
set_control_if_supported gamma 100
set_control_if_supported backlight_compensation 1
# Korea uses 60 Hz mains power. V4L2 enum: 1=50 Hz, 2=60 Hz.
set_control_if_supported power_line_frequency 2

# Prefer automatic exposure/white balance for the first comparison. Constant,
# bright lighting can later be used to tune and lock these values.
set_control_if_supported white_balance_automatic 1
set_control_if_supported white_balance_temperature_auto 1
set_control_if_supported auto_exposure 3
set_control_if_supported exposure_auto 3

# The datasheet's Windows example starts near 208. This is only a starting
# point: tune it at the real camera-to-garment distance.
set_control_if_supported focus_automatic_continuous 0
set_control_if_supported focus_auto 0
set_control_if_supported focus_absolute "$FOCUS_VALUE"

echo
echo "[INFO] Current camera controls:"
v4l2-ctl -d "$DEVICE" --list-ctrls-menus
echo
echo "[NEXT] Start the stable quality stream with: ./run_gscam_b0196.sh quality"
echo "[NOTE] If the image is still soft, stop gscam and retry this script with a nearby focus value."
echo "       Example: ./configure_arducam_b0196.sh 220"
