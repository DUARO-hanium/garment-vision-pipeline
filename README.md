# DUARO garment vision pipeline

This repository packages the local garment-vision work that had not been published with the existing `DUARO-hanium/Detection` repository. The actively used path is ROS camera input → fal.ai SAM3 garment mask → FashionAI keypoints → shoulder/hem grasp candidate or segmentation-based unfold candidate. The FlingBot value-network path is retained as an experiment, **not** as part of the active decision path.

## Code map

- `run_gscam_b0196.sh` / `configure_arducam_b0196.sh`: Arducam B0196 UVC camera setup and ROS 2 `gscam` publisher.
- `deep_fashion/fashionai-key-points-detection/run_garment_inspection_mvp.sh`: launcher for the current live MVP.
- `deep_fashion/fashionai-key-points-detection/garment_inspection_roi_mvp.py`: ROS subscriber, scene-stillness trigger, shoulder/hem phases, FashionAI confidence gate, image-space left/right arm ROI, mask-axis unfold fallback, visualization, JSON and video output. The fallback searches a horizontal mask span near the robot-facing side when the two requested keypoint scores fail the threshold. The current ready path deliberately skips ROI and angle gating after a FashionAI confidence pass.
- `deep_fashion/fashionai-key-points-detection/sam3_fal_segmenter.py`: uploads one frame to fal.ai SAM3; retries text prompts on empty/tiny masks; restores the mask to source-image coordinates.
- `deep_fashion/fashionai-key-points-detection/fashionai_keypoint_capture.py` and `fashionai_key_points_detection_utils.py`: modified FashionAI 512×512 preprocessing, ONNX Runtime inference, heatmap decoding, and image-space coordinates.
- `deep_fashion/fashionai-key-points-detection/sam2_garment_keypoint_once.py`: earlier local SAM2 + FashionAI single-frame trial. The active MVP imports only mask utility functions from it; it does not run the SAM2 model.
- `deep_fashion/fashionai-key-points-detection/flingbot_value_once.py`: independent FlingBot value-network single-frame trial, using an externally supplied checkpoint.
- `deep_fashion/fashionai-key-points-detection/garment_inspection_mvp.py`, `garment_grasp_pipeline.py`, `grasp_point_mvp.py`: earlier grasp/unfold experiments, kept for comparison.
- `deep_fashion/fashionai-key-points-detection/capture_ros_frame_once.py`, `run_resolution_pair.sh`, `summarize_resolution_pair.py`: camera capture and resolution comparison tools.
- `util/model_utils.py`: model-download helper required by the FashionAI code.

## Run the current MVP

This launcher assumes Ubuntu/WSL, ROS 2 Jazzy, a Python environment at `/home/jiyoung/.venvs/sam2`, the `gscam` ROS package, and an attached UVC camera at `/dev/video0`. Edit the launcher paths for another machine. Python packages include `numpy`, `opencv-python`, `onnxruntime`, `fal-client`, plus ROS Python packages `rclpy`, `sensor_msgs`, and `cv_bridge`. The older FlingBot/SAM2 experiments additionally require PyTorch, SciPy and their original model dependencies.

In terminal 1, from the repository root:

```bash
ARDUCAM_ROTATE=none bash ./run_gscam_b0196.sh realtime
```

In terminal 2, set a real fal.ai key **locally** (never commit it) and run:

```bash
cd deep_fashion/fashionai-key-points-detection
read -rsp 'FAL API Key: ' FAL_KEY; echo; export FAL_KEY
bash ./run_garment_inspection_mvp.sh --ros-topic /camera1/image_raw --clothing-type blouse --inference-interval 5.0 --sam3-prompt shirt --keypoint-threshold 150 --keypoint-inset-ratio 0.018 --max-attempts 3 --display-stage-seconds 0 --pipeline-panel-width 0
```

The script loads `blouse_100.onnx` (downloaded by the FashionAI model helper when missing). Other clothing-type weights must likewise be obtained separately. `blouse_100.onnx.prototxt` is included; `.onnx` and `.pth` weights are intentionally excluded. The FlingBot trial expects a separately obtained `flingbot.pth` checkpoint and a local SAM2 model; pass their paths using its command-line options.

SAM3 is a paid external API: camera frames are uploaded for segmentation. Inference can take seconds; this is not a guaranteed real-time control loop. All displayed grasp/unfold outputs are **image pixel `(u,v)` candidates**, not robot commands. Calibration, world-coordinate conversion, collision checks, and physical grasp validation are separate work.

## Provenance

FashionAI model integration is adapted from the `ailia-models` FashionAI example. The FlingBot experiment reproduces a value-network architecture/weight-loading path for local evaluation, without the simulator or robot action implementation. See the original upstream projects for model/checkpoint licensing and attribution. Do not assume this repository redistributes their weights.
