# 1280x720 vs 1920x1080 comparison

Use the same camera mount, focus, exposure, lighting, garment state, and
physical workspace for both captures. Do not touch the garment while changing
the gscam mode. PNG is used so the saved comparison file is not JPEG-compressed
again.

## 1. Common shell

```bash
cd "/mnt/d/문서/Users/23011/Documents/26-한이음/ros_projects/DUARO-hanium/Detection/deep_fashion/fashionai-key-points-detection"
source /opt/ros/jazzy/setup.bash
source /home/jiyoung/.venvs/sam2/bin/activate
```

## 2. Capture 1280x720

In the camera shell:

```bash
cd "/mnt/d/문서/Users/23011/Documents/26-한이음/ros_projects"
ARDUCAM_ROTATE=none bash ./run_gscam_b0196.sh realtime
```

In another shell:

```bash
python3 capture_ros_frame_once.py \
  --topic /camera1/image_raw \
  --expected-width 1280 \
  --expected-height 720 \
  --output "resolution_samples/state01/1280x720.png"
```

Stop only gscam with Ctrl+C. Do not move the camera or garment.

## 3. Capture 1920x1080

Restart the camera:

```bash
cd "/mnt/d/문서/Users/23011/Documents/26-한이음/ros_projects"
ARDUCAM_ROTATE=none bash ./run_gscam_b0196.sh quality
```

Capture in the inference shell:

```bash
python3 capture_ros_frame_once.py \
  --topic /camera1/image_raw \
  --expected-width 1920 \
  --expected-height 1080 \
  --output "resolution_samples/state01/1920x1080.png"
```

## 4. Run both model pipelines on the pair

```bash
bash ./run_resolution_pair.sh \
  state01 \
  "resolution_samples/state01/1280x720.png" \
  "resolution_samples/state01/1920x1080.png"
```

The summary is written to:

```text
outputs/resolution_comparison/state01/summary.json
```

Inspect these images for each resolution:

```text
fashionai_model_input_512.png
shoulder_heatmap_overlay.png
flingbot_normalized_input.png
flingbot_selected_input_64.png
best_value_map.png
result.png
```

## 5. Repeat

Repeat as `state02`, `state03`, and so on for at least ten distinct garment
states. Switch resolution without disturbing each state.

Compare normalized UV (`u/width`, `v/height`), mask selection, FashionAI peak
scores and usability, FlingBot action selection, preprocessing sharpness, and
latency. A single state is a smoke test; it is not enough to choose the final
resolution.

## Decision rule

- If normalized coordinates and success/failure decisions are similarly stable,
  prefer 1280x720 when ACT and calibration use 1280x720.
- Prefer 1920x1080 only when it produces a repeatable, material improvement in
  mask/keypoint/action quality that justifies recalibration and added load.
- The final calibration must be performed for the chosen Arducam mode, focus,
  mounting pose, and rectified image coordinate system.
