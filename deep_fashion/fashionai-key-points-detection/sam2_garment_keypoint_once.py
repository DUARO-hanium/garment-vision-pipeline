"""One-shot garment segmentation and shoulder-keypoint inference.

The garment can be selected with an interactive box, --box x1,y1,x2,y2,
or automatically with --auto-mask. SAM2 produces the garment mask, then
the existing FashionAI model runs on the masked garment crop.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from fashionai_keypoint_capture import (
    HM_STRIDE,
    KEYPOINT_NAMES,
    load_net,
    predict_keypoints_with_heatmaps,
)


def parse_box(value: str) -> tuple[int, int, int, int]:
    try:
        values = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("box must contain integers: x1,y1,x2,y2") from error
    if len(values) != 4:
        raise argparse.ArgumentTypeError("box must contain four values: x1,y1,x2,y2")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use a prompted SAM2 mask before FashionAI shoulder inference."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="Input image path.")
    source.add_argument("--ros-topic", help="ROS 2 sensor_msgs/Image topic, e.g. /camera1/image_raw.")
    parser.add_argument(
        "--box",
        type=parse_box,
        help="Optional garment box in original-image pixels: x1,y1,x2,y2. Without it, draw a box with the mouse.",
    )
    parser.add_argument(
        "--auto-mask",
        action="store_true",
        help="Do not ask for a box. Use automatic point prompts and select the largest plausible central object.",
    )
    parser.add_argument("--clothing-type", choices=sorted(KEYPOINT_NAMES), default="blouse")
    parser.add_argument("--model", default="/home/jiyoung/models/sam2_t.pt")
    parser.add_argument("--output-dir", default="outputs/sam2_once")
    parser.add_argument("--padding", type=float, default=0.08, help="Padding around the mask-derived crop.")
    parser.add_argument("--ros-timeout", type=float, default=15.0)
    parser.add_argument(
        "--shoulder-score-threshold",
        type=float,
        default=10.0,
        help="Experimental minimum heatmap peak score for a usable shoulder candidate.",
    )
    parser.add_argument("--no-show", action="store_true", help="Save results without opening a result window.")
    args = parser.parse_args()
    if args.auto_mask and args.box is not None:
        parser.error("--auto-mask and --box cannot be used together")
    return args


def receive_ros_frame(topic: str, timeout_seconds: float) -> np.ndarray:
    import rclpy
    from cv_bridge import CvBridge
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import Image

    rclpy.init()
    node = rclpy.create_node("sam2_garment_frame_receiver")
    bridge = CvBridge()
    holder: dict[str, np.ndarray] = {}
    qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )

    def callback(message: Image) -> None:
        if "frame" not in holder:
            holder["frame"] = bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")

    subscription = node.create_subscription(Image, topic, callback, qos)
    deadline = time.monotonic() + timeout_seconds
    try:
        while "frame" not in holder and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.2)
    finally:
        node.destroy_subscription(subscription)
        node.destroy_node()
        rclpy.shutdown()

    if "frame" not in holder:
        raise RuntimeError(f"No image received from {topic} within {timeout_seconds:.1f} seconds")
    return holder["frame"]


def select_box(frame: np.ndarray) -> tuple[int, int, int, int]:
    height, width = frame.shape[:2]
    scale = min(1.0, 1400.0 / width, 850.0 / height)
    preview = frame
    if scale < 1.0:
        preview = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    window = "Draw a box around ONE garment, then press ENTER or SPACE"
    x, y, w, h = cv2.selectROI(window, preview, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(window)
    if w <= 0 or h <= 0:
        raise RuntimeError("Garment box selection was cancelled")

    x1 = int(round(x / scale))
    y1 = int(round(y / scale))
    x2 = int(round((x + w) / scale))
    y2 = int(round((y + h) / scale))
    return clamp_box((x1, y1, x2, y2), width, height)


def clamp_box(box: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))
    return x1, y1, x2, y2


def run_sam2(
    frame: np.ndarray,
    box: tuple[int, int, int, int],
    model_path: str,
) -> tuple[np.ndarray, float, float]:
    from ultralytics import SAM

    started = time.perf_counter()
    model = SAM(model_path)
    results = model.predict(
        source=frame,
        bboxes=list(box),
        device="cpu",
        verbose=False,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if not results or results[0].masks is None or len(results[0].masks.data) == 0:
        raise RuntimeError("SAM2 did not return a mask for the selected box")

    masks = results[0].masks.data.cpu().numpy()
    mask = max(masks, key=lambda item: float(item.sum()))
    mask = (mask > 0.5).astype(np.uint8) * 255
    if mask.shape != frame.shape[:2]:
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

    confidence = 1.0
    if results[0].boxes is not None and results[0].boxes.conf is not None:
        values = results[0].boxes.conf.cpu().numpy()
        if len(values):
            confidence = float(np.max(values))

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask, elapsed_ms, confidence


def run_sam2_auto(
    frame: np.ndarray,
    model_path: str,
    sam_model=None,
) -> tuple[np.ndarray, float, dict, np.ndarray]:
    """Prompt SAM2 on a central grid and select a plausible single garment mask."""
    from ultralytics import SAM

    height, width = frame.shape[:2]
    # Positive point prompts covering the central 80% of the table view. Each
    # point produces an independent object candidate without mouse input.
    xs = np.linspace(0.10 * width, 0.90 * width, 5)
    ys = np.linspace(0.10 * height, 0.90 * height, 5)
    points = [[float(x), float(y)] for y in ys for x in xs]
    labels = [1] * len(points)

    started = time.perf_counter()
    # One-shot callers pass only model_path.  Streaming callers can pass an
    # already loaded model so the checkpoint is not reloaded every cycle.
    model = sam_model if sam_model is not None else SAM(model_path)
    results = model.predict(
        source=frame,
        points=points,
        labels=labels,
        device="cpu",
        verbose=False,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if not results or results[0].masks is None or len(results[0].masks.data) == 0:
        raise RuntimeError("SAM2 did not return any automatic mask candidates")

    masks = results[0].masks.data.cpu().numpy()
    confidences = np.ones(len(masks), dtype=np.float32)
    if results[0].boxes is not None and results[0].boxes.conf is not None:
        box_confidences = results[0].boxes.conf.cpu().numpy()
        if len(box_confidences) == len(masks):
            confidences = box_confidences

    candidates = []
    candidate_masks = []
    image_area = float(width * height)
    image_center = np.array([width / 2.0, height / 2.0])
    half_diagonal = max(1.0, float(np.linalg.norm(image_center)))

    for index, (raw_mask, confidence) in enumerate(zip(masks, confidences)):
        mask = (raw_mask > 0.5).astype(np.uint8) * 255
        if mask.shape != frame.shape[:2]:
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        area = int(np.count_nonzero(mask))
        area_ratio = area / image_area
        ys_mask, xs_mask = np.where(mask > 0)
        if area == 0:
            center_distance = 1.0
        else:
            mask_center = np.array([float(xs_mask.mean()), float(ys_mask.mean())])
            center_distance = float(np.linalg.norm(mask_center - image_center) / half_diagonal)

        touches = {
            "top": bool(np.any(mask[0, :])),
            "bottom": bool(np.any(mask[-1, :])),
            "left": bool(np.any(mask[:, 0])),
            "right": bool(np.any(mask[:, -1])),
        }
        touched_sides = sum(touches.values())
        eligible = bool(
            0.02 <= area_ratio <= 0.85
            # The table/background normally reaches all four borders. Allow a
            # partially cropped garment to touch as many as three borders.
            and touched_sides < 4
            and center_distance <= 0.85
        )
        selection_score = area_ratio * (1.5 - 0.5 * center_distance) + 0.03 * float(confidence)
        candidates.append(
            {
                "index": index,
                "area_pixels": area,
                "area_ratio": round(area_ratio, 6),
                "confidence": round(float(confidence), 6),
                "center_distance": round(center_distance, 6),
                "touched_sides": touched_sides,
                "eligible": eligible,
                "selection_score": round(selection_score, 6),
            }
        )
        candidate_masks.append(mask)

    eligible_indices = [item["index"] for item in candidates if item["eligible"]]
    if not eligible_indices:
        raise RuntimeError(
            "No plausible garment mask was found automatically. Keep the whole garment visible near the image center."
        )
    selected_index = max(eligible_indices, key=lambda idx: candidates[idx]["selection_score"])
    selected_mask = candidate_masks[selected_index]
    kernel = np.ones((5, 5), np.uint8)
    selected_mask = cv2.morphologyEx(selected_mask, cv2.MORPH_CLOSE, kernel)

    candidate_view = frame.copy()
    for item, mask in zip(candidates, candidate_masks):
        if not item["eligible"]:
            continue
        color = (0, 255, 255) if item["index"] == selected_index else (160, 160, 160)
        thickness = 4 if item["index"] == selected_index else 1
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(candidate_view, contours, -1, color, thickness)
    cv2.putText(
        candidate_view,
        f"auto selected mask #{selected_index}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    details = {
        "method": "5x5_positive_point_grid",
        "point_count": len(points),
        "selected_index": selected_index,
        "selected_confidence": candidates[selected_index]["confidence"],
        "selected_area_ratio": candidates[selected_index]["area_ratio"],
        "candidate_count": len(candidates),
        "eligible_candidate_count": len(eligible_indices),
        "candidates": candidates,
    }
    return selected_mask, elapsed_ms, details, candidate_view


def mask_crop_box(mask: np.ndarray, padding_ratio: float) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise RuntimeError("SAM2 returned an empty mask")
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    padding = int(round(max(x2 - x1, y2 - y1) * max(0.0, padding_ratio)))
    return clamp_box((x1 - padding, y1 - padding, x2 + padding, y2 + padding), mask.shape[1], mask.shape[0])


def draw_shoulders(
    image: np.ndarray,
    clothing_type: str,
    keypoints: np.ndarray,
) -> np.ndarray:
    output = image.copy()
    names = KEYPOINT_NAMES[clothing_type]
    shoulder_names = ("shoulder_left", "shoulder_right")
    colors = ((255, 80, 80), (80, 80, 255))
    for name, color in zip(shoulder_names, colors):
        index = names.index(name)
        u, v, visible = keypoints[index].tolist()
        if visible <= 0:
            continue
        point = (int(u), int(v))
        cv2.drawMarker(output, point, color, cv2.MARKER_CROSS, 24, 4)
        cv2.circle(output, point, 8, color, 3)
        cv2.putText(output, name, (point[0] + 10, max(25, point[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return output


def make_overlay(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    color = np.zeros_like(frame)
    color[:, :] = (30, 210, 30)
    blended = cv2.addWeighted(frame, 0.72, color, 0.28, 0)
    output = frame.copy()
    output[mask > 0] = blended[mask > 0]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(output, contours, -1, (0, 255, 0), 3)
    return output


def shoulder_heatmap_image(
    heatmap: np.ndarray,
    crop_size: tuple[int, int],
    valid_heatmap_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Map the unpadded model heatmap back onto the garment crop."""
    crop_width, crop_height = crop_size
    valid_width, valid_height = valid_heatmap_size
    valid = heatmap[:valid_height, :valid_width]
    resized = cv2.resize(valid, (crop_width, crop_height), interpolation=cv2.INTER_CUBIC)
    normalized = cv2.normalize(resized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    return resized, colored


def point_near_mask(mask: np.ndarray, point: tuple[int, int], tolerance: int) -> bool:
    kernel_size = max(3, tolerance * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    expanded = cv2.dilate(mask, kernel)
    u, v = point
    return 0 <= v < mask.shape[0] and 0 <= u < mask.shape[1] and bool(expanded[v, u])


def main() -> None:
    args = parse_args()
    if args.input:
        frame = cv2.imread(args.input, cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"Could not read image: {args.input}")
        source_name = args.input
    else:
        print(f"[INFO] waiting for one frame from {args.ros_topic}")
        frame = receive_ros_frame(args.ros_topic, args.ros_timeout)
        source_name = args.ros_topic

    height, width = frame.shape[:2]
    auto_details = None
    auto_candidate_view = None
    if args.auto_mask:
        box = None
        print("[INFO] auto-mask enabled: no mouse box is required")
        print("[INFO] running SAM2 tiny automatic point prompts on CPU")
        mask, sam_ms, auto_details, auto_candidate_view = run_sam2_auto(frame, args.model)
        sam_mask_confidence = float(auto_details["selected_confidence"])
        print(
            f"[INFO] auto-selected mask #{auto_details['selected_index']} "
            f"from {auto_details['candidate_count']} candidates | "
            f"SAM confidence={sam_mask_confidence:.4f}"
        )
    else:
        box = clamp_box(args.box, width, height) if args.box else select_box(frame)
        print(f"[INFO] selected box: {box}")
        print("[INFO] running SAM2 tiny on CPU; the first run may take a while")
        mask, sam_ms, sam_mask_confidence = run_sam2(frame, box, args.model)
    crop_box = mask_crop_box(mask, args.padding)
    x1, y1, x2, y2 = crop_box

    garment_only = np.full_like(frame, 255)
    garment_only[mask > 0] = frame[mask > 0]
    crop = garment_only[y1:y2, x1:x2]

    print(f"[INFO] running FashionAI '{args.clothing_type}' shoulders on the garment crop")
    net = load_net(args.clothing_type, env_id=None)
    started = time.perf_counter()
    keypoints, combined_heatmap, scores = predict_keypoints_with_heatmaps(
        args.clothing_type, crop, net
    )
    fashionai_ms = (time.perf_counter() - started) * 1000.0
    keypoints[:, 0] += x1
    keypoints[:, 1] += y1

    result = make_overlay(frame, mask)
    if box is not None:
        cv2.rectangle(result, (box[0], box[1]), (box[2] - 1, box[3] - 1), (0, 200, 255), 2)
    cv2.rectangle(result, (x1, y1), (x2 - 1, y2 - 1), (255, 0, 255), 2)
    result = draw_shoulders(result, args.clothing_type, keypoints)
    cv2.putText(result, f"SAM2: {sam_ms:.0f} ms | FashionAI: {fashionai_ms:.0f} ms", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    run_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(run_dir / "input.png"), frame)
    cv2.imwrite(str(run_dir / "garment_mask.png"), mask)
    cv2.imwrite(str(run_dir / "garment_only.png"), garment_only)
    cv2.imwrite(str(run_dir / "garment_crop.png"), crop)
    model_scale = 512.0 / max(crop.shape[1], crop.shape[0])
    model_width = max(1, int(crop.shape[1] * model_scale))
    model_height = max(1, int(crop.shape[0] * model_scale))
    fashionai_input_preview = np.zeros((512, 512, 3), dtype=np.uint8)
    fashionai_input_preview[:model_height, :model_width] = cv2.resize(
        crop,
        (model_width, model_height),
        interpolation=cv2.INTER_CUBIC,
    )
    cv2.imwrite(str(run_dir / "fashionai_model_input_512.png"), fashionai_input_preview)
    if auto_candidate_view is not None:
        cv2.imwrite(str(run_dir / "auto_mask_candidates.png"), auto_candidate_view)

    names = KEYPOINT_NAMES[args.clothing_type]
    shoulders = {}
    heatmap_colors = []
    heatmap_raw = []
    model_scale = 512.0 / max(crop.shape[1], crop.shape[0])
    valid_hm_width = max(1, int(crop.shape[1] * model_scale) // HM_STRIDE)
    valid_hm_height = max(1, int(crop.shape[0] * model_scale) // HM_STRIDE)
    tolerance = max(5, int(round(max(crop.shape[:2]) * 0.03)))
    for name in ("shoulder_left", "shoulder_right"):
        index = names.index(name)
        u, v, visible = keypoints[index].tolist()
        mapped_heatmap, colored_heatmap = shoulder_heatmap_image(
            combined_heatmap[index],
            (crop.shape[1], crop.shape[0]),
            (valid_hm_width, valid_hm_height),
        )
        heatmap_raw.append(mapped_heatmap)
        heatmap_colors.append(colored_heatmap)
        point = (int(u), int(v))
        near_mask = point_near_mask(mask, point, tolerance)
        score = float(scores[index])
        detected = bool(visible > 0 and score >= args.shoulder_score_threshold and near_mask)
        shoulders[name] = {
            "u": point[0],
            "v": point[1],
            "visible": bool(visible > 0),
            "heatmap_peak_score": round(score, 6),
            "near_garment_mask": near_mask,
            "detected": detected,
        }

    heatmap_left = heatmap_colors[0]
    heatmap_right = heatmap_colors[1]
    heatmap_both_values = np.maximum(heatmap_raw[0], heatmap_raw[1])
    heatmap_both_u8 = cv2.normalize(heatmap_both_values, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap_both = cv2.applyColorMap(heatmap_both_u8, cv2.COLORMAP_JET)
    heatmap_overlay_crop = cv2.addWeighted(crop, 0.55, heatmap_both, 0.45, 0)
    heatmap_overlay = frame.copy()
    heatmap_overlay[y1:y2, x1:x2] = heatmap_overlay_crop
    heatmap_overlay = draw_shoulders(heatmap_overlay, args.clothing_type, keypoints)
    cv2.imwrite(str(run_dir / "shoulder_left_heatmap.png"), heatmap_left)
    cv2.imwrite(str(run_dir / "shoulder_right_heatmap.png"), heatmap_right)
    cv2.imwrite(str(run_dir / "shoulder_both_heatmap.png"), heatmap_both)
    cv2.imwrite(str(run_dir / "shoulder_heatmap_overlay.png"), heatmap_overlay)

    left_point = np.array([shoulders["shoulder_left"]["u"], shoulders["shoulder_left"]["v"]])
    right_point = np.array([shoulders["shoulder_right"]["u"], shoulders["shoulder_right"]["v"]])
    shoulder_distance = float(np.linalg.norm(left_point - right_point))
    minimum_distance = 0.10 * crop.shape[1]
    shoulders_usable = bool(
        shoulders["shoulder_left"]["detected"]
        and shoulders["shoulder_right"]["detected"]
        and shoulder_distance >= minimum_distance
    )
    cv2.putText(
        result,
        f"SAM mask confidence: {sam_mask_confidence:.4f}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        result,
        "Shoulder heatmap peak: "
        f"L={shoulders['shoulder_left']['heatmap_peak_score']:.4f} "
        f"R={shoulders['shoulder_right']['heatmap_peak_score']:.4f}",
        (20, 103),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        result,
        f"Shoulder candidates usable: {'YES' if shoulders_usable else 'NO'}",
        (20, 136),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (0, 255, 0) if shoulders_usable else (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(run_dir / "result.png"), result)
    metadata = {
        "source": source_name,
        "image_size": {"width": width, "height": height},
        "prompt_box_xyxy": list(box) if box is not None else None,
        "auto_mask": auto_details,
        "mask_crop_box_xyxy": list(crop_box),
        "mask_area_pixels": int(np.count_nonzero(mask)),
        "sam2_ms": round(sam_ms, 3),
        "sam2_mask_confidence": round(sam_mask_confidence, 6),
        "fashionai_ms": round(fashionai_ms, 3),
        "clothing_type": args.clothing_type,
        "shoulder_score_threshold": args.shoulder_score_threshold,
        "shoulder_distance_pixels": round(shoulder_distance, 3),
        "shoulder_candidates_usable": shoulders_usable,
        "shoulders": shoulders,
    }
    with open(run_dir / "result.json", "w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, ensure_ascii=False)

    print(f"[DONE] results saved under: {run_dir}")
    print(
        "[RESULT] shoulder candidates usable: "
        f"{'YES' if shoulders_usable else 'NO'} | "
        f"SAM mask confidence={sam_mask_confidence:.4f} | "
        f"left={shoulders['shoulder_left']['heatmap_peak_score']:.4f}, "
        f"right={shoulders['shoulder_right']['heatmap_peak_score']:.4f}"
    )
    if not args.no_show:
        comparison = np.hstack((result, heatmap_overlay))
        scale = min(1.0, 1600.0 / comparison.shape[1], 850.0 / comparison.shape[0])
        preview = cv2.resize(comparison, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        cv2.imshow("Mask + shoulders | shoulder heatmap (press any key)", preview)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
