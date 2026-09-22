"""One-frame SAM2 garment segmentation and standalone FlingBot value inference.

This intentionally imports no simulator or PyFlex code. It loads only the RGB
FlingBot value-network weights and returns two image-space grasp candidates.
The returned pixels are visualization/evaluation candidates, not robot commands.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from scipy import ndimage as nd

from sam2_garment_keypoint_once import receive_ros_frame, run_sam2_auto


ROTATIONS = np.linspace(-90.0, 90.0, 12).tolist()
SCALES = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75]
OBS_DIM = 64
PRETRANSFORM_DIM = 256
PIX_GRASP_DIST = 8


class BasicBlock(nn.Module):
    def __init__(self, inplanes: int, planes: int, kernel_size: int, stride: int, non_linearity=nn.LeakyReLU):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(inplanes, planes, kernel_size, stride, padding=1, bias=False)
        ]
        if non_linearity is not None:
            layers.extend([nn.BatchNorm2d(planes), non_linearity()])
        self.net = nn.Sequential(*layers)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class ResidualBlock(nn.Module):
    def __init__(self, planes: int = 16):
        super().__init__()
        self.conv1 = nn.Conv2d(planes, planes, 3, 1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        identity = value
        value = self.relu(self.bn1(self.conv1(value)))
        value = self.bn2(self.conv2(value))
        return self.relu(value + identity)


class SpatialValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            BasicBlock(3, 16, 3, 1),
            *[ResidualBlock(16) for _ in range(8)],
            BasicBlock(16, 1, 3, 1, non_linearity=None),
        )
        self.steps = nn.Parameter(torch.tensor(0), requires_grad=False)
        self.register_buffer(
            "mean", torch.tensor([0.18, 0.18, 0.18]).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "std", torch.tensor([0.10, 0.10, 0.10]).view(1, 3, 1, 1), persistent=False
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net((value - self.mean) / self.std)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SAM2 auto-mask followed by standalone pretrained FlingBot value-map inference."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="Input image path.")
    source.add_argument("--ros-topic", help="ROS 2 sensor_msgs/Image topic.")
    parser.add_argument("--checkpoint", default="/mnt/d/문서/Users/23011/Documents/26-한이음/ros_projects/flingbot/flingbot.pth")
    parser.add_argument("--sam-model", default="/home/jiyoung/models/sam2_t.pt")
    parser.add_argument("--mask", help="Optional existing binary mask; skips SAM2 for repeatable testing.")
    parser.add_argument("--output-dir", default="outputs/flingbot_value_once")
    parser.add_argument("--ros-timeout", type=float, default=15.0)
    parser.add_argument("--cloth-fill-ratio", type=float, default=0.67)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=96, help="Value-network transform batch size.")
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


def load_value_net(checkpoint_path: str, device: torch.device) -> SpatialValueNet:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    source_state = checkpoint["net"]
    prefix = "value_nets.fling."
    state = {key[len(prefix):]: value for key, value in source_state.items() if key.startswith(prefix)}
    model = SpatialValueNet()
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    return model


def square_cloth_crop(
    frame_bgr: np.ndarray,
    mask: np.ndarray,
    fill_ratio: float,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise RuntimeError("Garment mask is empty")
    width = int(xs.max() - xs.min() + 1)
    height = int(ys.max() - ys.min() + 1)
    ratio = float(np.clip(fill_ratio, 0.35, 0.90))
    side = max(64, int(np.ceil(max(width, height) / ratio)))
    center_x = (float(xs.min()) + float(xs.max())) / 2.0
    center_y = (float(ys.min()) + float(ys.max())) / 2.0
    origin_x = int(round(center_x - side / 2.0))
    origin_y = int(round(center_y - side / 2.0))

    crop = np.zeros((side, side, 3), dtype=np.uint8)
    crop_mask = np.zeros((side, side), dtype=np.uint8)
    src_x1, src_y1 = max(0, origin_x), max(0, origin_y)
    src_x2 = min(frame_bgr.shape[1], origin_x + side)
    src_y2 = min(frame_bgr.shape[0], origin_y + side)
    dst_x1, dst_y1 = src_x1 - origin_x, src_y1 - origin_y
    dst_x2, dst_y2 = dst_x1 + (src_x2 - src_x1), dst_y1 + (src_y2 - src_y1)
    crop[dst_y1:dst_y2, dst_x1:dst_x2] = frame_bgr[src_y1:src_y2, src_x1:src_x2]
    crop_mask[dst_y1:dst_y2, dst_x1:dst_x2] = mask[src_y1:src_y2, src_x1:src_x2]
    crop[crop_mask == 0] = 0
    return crop, crop_mask, (origin_x, origin_y)


def transform_rgb(image_rgb: np.ndarray, rotation: float, scale: float) -> np.ndarray:
    transformed = nd.rotate(image_rgb, angle=rotation, reshape=False, mode="nearest")
    new_dim = int(scale * transformed.shape[0])
    if scale < 1.0:
        start = transformed.shape[0] // 2 - new_dim // 2
        transformed = transformed[start:start + new_dim, start:start + new_dim]
    elif scale > 1.0:
        padding = (new_dim - transformed.shape[0]) // 2
        transformed = cv2.copyMakeBorder(
            transformed, padding, padding, padding, padding, cv2.BORDER_REPLICATE
        )
    transformed = cv2.resize(transformed, (OBS_DIM, OBS_DIM), interpolation=cv2.INTER_NEAREST)
    return np.transpose(transformed, (2, 0, 1)).astype(np.float32) / 255.0


def build_transformed_batch(crop_bgr: np.ndarray) -> tuple[np.ndarray, list[tuple[float, float]]]:
    # The official real-world pipeline first resizes its square workspace crop
    # to 256x256, then creates the rotation/scale batch at 64x64.
    normalized_crop = cv2.resize(
        crop_bgr, (PRETRANSFORM_DIM, PRETRANSFORM_DIM), interpolation=cv2.INTER_AREA
    )
    image_rgb = cv2.cvtColor(normalized_crop, cv2.COLOR_BGR2RGB)
    transformations = [(rotation, scale) for rotation in ROTATIONS for scale in SCALES]
    inputs = np.stack([transform_rgb(image_rgb, rotation, scale) for rotation, scale in transformations])
    return inputs, transformations


def rot2d(angle_degrees: float) -> np.ndarray:
    angle = np.pi * angle_degrees / 180.0
    return np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])


def translate2d(translation: tuple[float, float] | np.ndarray) -> np.ndarray:
    return np.array(
        [[1, 0, translation[0]], [0, 1, translation[1]], [0, 0, 1]],
        dtype=np.float64,
    ).T


def scale2d(scale: float) -> np.ndarray:
    return np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], dtype=np.float64)


def inverse_transform_matrix(original_dim: int, rotation: float, scale: float) -> np.ndarray:
    # Equivalent to FlingBot environment.utils.get_transform_matrix with row vectors.
    center = np.ones(2) * (OBS_DIM // 2)
    resize_mat = scale2d(original_dim / OBS_DIM)
    scale_mat = translate2d(-center) @ scale2d(scale) @ translate2d(center)
    rotation_mat = translate2d(-center) @ rot2d(-rotation) @ translate2d(center)
    return scale_mat @ rotation_mat @ resize_mat


def local_mask_fraction(mask: np.ndarray, u: int, v: int, radius: int) -> float:
    if not (0 <= u < mask.shape[1] and 0 <= v < mask.shape[0]):
        return 0.0
    y1, y2 = max(0, v - radius), min(mask.shape[0], v + radius + 1)
    x1, x2 = max(0, u - radius), min(mask.shape[1], u + radius + 1)
    patch = mask[y1:y2, x1:x2]
    return float(np.count_nonzero(patch) / max(1, patch.size))


def select_grasp_pair(
    value_maps: np.ndarray,
    transformations: list[tuple[float, float]],
    crop_side: int,
    crop_origin: tuple[int, int],
    full_mask: np.ndarray,
) -> dict:
    filtered = value_maps.copy()
    filtered[:, :PIX_GRASP_DIST, :] = -np.inf
    filtered[:, -PIX_GRASP_DIST:, :] = -np.inf
    filtered[:, :, :PIX_GRASP_DIST] = -np.inf
    filtered[:, :, -PIX_GRASP_DIST:] = -np.inf
    order = np.argsort(filtered.reshape(-1))[::-1]
    radius = max(3, int(round(crop_side / OBS_DIM * 0.65)))

    for flat_index in order[:100000]:
        transform_index, row, col = np.unravel_index(flat_index, filtered.shape)
        value = float(filtered[transform_index, row, col])
        if not np.isfinite(value):
            break
        rotation, scale = transformations[transform_index]
        transformed_points = np.array(
            [[row + PIX_GRASP_DIST, col, 1.0], [row - PIX_GRASP_DIST, col, 1.0]],
            dtype=np.float64,
        )
        normalized_points = (
            transformed_points
            @ inverse_transform_matrix(PRETRANSFORM_DIM, rotation, scale)
        )[:, :2]
        crop_points = normalized_points * (crop_side / PRETRANSFORM_DIM)
        points = []
        valid = True
        for crop_row, crop_col in crop_points:
            u = int(round(crop_col + crop_origin[0]))
            v = int(round(crop_row + crop_origin[1]))
            fraction = local_mask_fraction(full_mask, u, v, radius)
            if fraction < 0.55:
                valid = False
                break
            points.append({"u": u, "v": v, "mask_fraction": round(fraction, 4)})
        if not valid:
            continue
        points.sort(key=lambda point: point["u"])
        distance = float(np.hypot(points[1]["u"] - points[0]["u"], points[1]["v"] - points[0]["v"]))
        if distance < max(20.0, crop_side * 0.08):
            continue
        return {
            "value": value,
            "transform_index": int(transform_index),
            "rotation_deg": float(rotation),
            "scale": float(scale),
            "value_map_center": {"row": int(row), "col": int(col)},
            "left_grasp": points[0],
            "right_grasp": points[1],
            "distance_pixels": round(distance, 3),
        }
    raise RuntimeError("No FlingBot grasp pair had both points safely inside the garment mask")


def main() -> None:
    args = parse_args()
    if args.input:
        frame = cv2.imread(args.input, cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"Could not read input image: {args.input}")
        source_name = args.input
    else:
        print(f"[INFO] waiting for one frame from {args.ros_topic}")
        frame = receive_ros_frame(args.ros_topic, args.ros_timeout)
        source_name = args.ros_topic

    if args.mask:
        mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Could not read mask: {args.mask}")
        if mask.shape != frame.shape[:2]:
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.uint8) * 255
        sam_ms = 0.0
        sam_details = {"method": "provided_mask"}
        sam_confidence = None
    else:
        print("[INFO] running automatic SAM2 garment segmentation")
        mask, sam_ms, sam_details, _ = run_sam2_auto(frame, args.sam_model)
        sam_confidence = float(sam_details["selected_confidence"])

    crop, crop_mask, crop_origin = square_cloth_crop(frame, mask, args.cloth_fill_ratio)
    inputs, transformations = build_transformed_batch(crop)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but this PyTorch installation has no CUDA support")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    print(f"[INFO] loading FlingBot RGB value network on {device}")
    model = load_value_net(args.checkpoint, device)
    started = time.perf_counter()
    outputs = []
    batch_size = max(1, args.batch_size)
    with torch.inference_mode():
        for start in range(0, len(inputs), batch_size):
            tensor = torch.from_numpy(inputs[start:start + batch_size]).to(device)
            outputs.append(model(tensor).squeeze(1).cpu().numpy())
    value_maps = np.concatenate(outputs, axis=0)
    value_ms = (time.perf_counter() - started) * 1000.0
    best = select_grasp_pair(value_maps, transformations, crop.shape[0], crop_origin, mask)

    result = frame.copy()
    overlay = np.zeros_like(result)
    overlay[mask > 0] = (30, 180, 30)
    result = cv2.addWeighted(result, 0.82, overlay, 0.18, 0)
    left = (best["left_grasp"]["u"], best["left_grasp"]["v"])
    right = (best["right_grasp"]["u"], best["right_grasp"]["v"])
    cv2.line(result, left, right, (0, 255, 255), 5, cv2.LINE_AA)
    cv2.circle(result, left, 14, (255, 80, 80), -1)
    cv2.circle(result, right, 14, (80, 80, 255), -1)
    cv2.putText(result, "L", (left[0] + 12, left[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 80, 80), 3)
    cv2.putText(result, "R", (right[0] + 12, right[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 80, 255), 3)
    cv2.putText(result, f"FlingBot value={best['value']:.4f} rot={best['rotation_deg']:.1f} scale={best['scale']:.2f}", (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(result, f"SAM={sam_ms:.0f}ms ValueNet={value_ms:.0f}ms device={device}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    selected_map = value_maps[best["transform_index"]]
    normalized = cv2.normalize(selected_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    value_heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    center = best["value_map_center"]
    cv2.circle(value_heatmap, (center["col"], center["row"]), 3, (255, 255, 255), 1)

    run_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(run_dir / "input.png"), frame)
    cv2.imwrite(str(run_dir / "garment_mask.png"), mask)
    normalized_preview = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(run_dir / "flingbot_normalized_input.png"), normalized_preview)
    selected_input_rgb = np.transpose(inputs[best["transform_index"]], (1, 2, 0))
    selected_input_bgr = cv2.cvtColor(
        np.clip(selected_input_rgb * 255.0, 0, 255).astype(np.uint8),
        cv2.COLOR_RGB2BGR,
    )
    cv2.imwrite(str(run_dir / "flingbot_selected_input_64.png"), selected_input_bgr)
    cv2.imwrite(str(run_dir / "best_value_map.png"), cv2.resize(value_heatmap, (512, 512), interpolation=cv2.INTER_NEAREST))
    cv2.imwrite(str(run_dir / "result.png"), result)
    metadata = {
        "warning": "Image-space candidates only; do not send directly to a robot without calibration and reachability checks.",
        "source": source_name,
        "image_size": {"width": frame.shape[1], "height": frame.shape[0]},
        "device": str(device),
        "sam2_ms": round(sam_ms, 3),
        "sam2_mask_confidence": round(sam_confidence, 6) if sam_confidence is not None else None,
        "flingbot_value_ms": round(value_ms, 3),
        "crop_origin_xy": list(crop_origin),
        "crop_side_pixels": crop.shape[0],
        "best_action": best,
        "sam_auto_selection": sam_details,
    }
    with open(run_dir / "result.json", "w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, ensure_ascii=False)

    print(f"[DONE] results saved under: {run_dir}")
    print(
        f"[RESULT] value={best['value']:.4f} rotation={best['rotation_deg']:.1f} "
        f"scale={best['scale']:.2f} L=({left[0]},{left[1]}) R=({right[0]},{right[1]})"
    )
    print("[SAFETY] These are image candidates only, not calibrated robot coordinates.")
    if not args.no_show:
        heatmap_large = cv2.resize(value_heatmap, (result.shape[0], result.shape[0]), interpolation=cv2.INTER_NEAREST)
        if heatmap_large.shape[0] != result.shape[0]:
            heatmap_large = cv2.resize(heatmap_large, (result.shape[0], result.shape[0]))
        comparison = np.hstack((result, heatmap_large))
        scale = min(1.0, 1600.0 / comparison.shape[1], 850.0 / comparison.shape[0])
        preview = cv2.resize(comparison, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        cv2.imshow("FlingBot grasp candidates | selected value map (press any key)", preview)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
