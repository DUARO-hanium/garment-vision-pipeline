"""Live ROS garment inspection MVP.

The camera preview remains live while a background worker periodically runs:
SAM2 garment segmentation -> FashionAI semantic keypoints -> (if needed)
FlingBot value-network grasp candidates.  Results are image-space candidates;
they are not robot commands until calibration and reachability checks are added.
"""

from __future__ import annotations

import argparse
import json
import queue
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from fashionai_keypoint_capture import KEYPOINT_NAMES, load_net, predict_keypoints_with_heatmaps
from flingbot_value_once import (
    build_transformed_batch,
    load_value_net,
    select_grasp_pair,
    square_cloth_crop,
)
from sam2_garment_keypoint_once import mask_crop_box, point_near_mask, run_sam2_auto


TARGETS = {
    "blouse": {
        "shoulders": ("shoulder_left", "shoulder_right"),
        "lower": ("top_hem_left", "top_hem_right"),
    },
    "outwear": {
        "shoulders": ("shoulder_left", "shoulder_right"),
        "lower": ("waistline_left", "waistline_right"),
    },
    "dress": {
        "shoulders": ("shoulder_left", "shoulder_right"),
        "lower": ("waistline_left", "waistline_right"),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live SAM2 + FashionAI + FlingBot inspection MVP")
    parser.add_argument("--ros-topic", default="/camera1/image_raw")
    parser.add_argument("--clothing-type", choices=sorted(TARGETS), default="blouse")
    parser.add_argument("--sam-model", default="/home/jiyoung/models/sam2_t.pt")
    parser.add_argument(
        "--fling-checkpoint",
        default="/mnt/d/문서/Users/23011/Documents/26-한이음/ros_projects/flingbot/flingbot.pth",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--inference-interval", type=float, default=2.0)
    parser.add_argument("--shoulder-threshold", type=float, default=10.0)
    parser.add_argument("--lower-threshold", type=float, default=10.0)
    parser.add_argument("--fling-threshold", type=float, default=0.0)
    parser.add_argument("--padding", type=float, default=0.08)
    parser.add_argument("--cloth-fill-ratio", type=float, default=0.67)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--display-width", type=int, default=1280)
    parser.add_argument("--record-fps", type=float, default=20.0)
    parser.add_argument("--output-dir", default="outputs/garment_inspection_mvp")
    parser.add_argument("--no-fling", action="store_true")
    parser.add_argument("--no-record", action="store_true")
    return parser.parse_args()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def choose_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is not available")
    use_cuda = requested == "cuda" or (requested == "auto" and torch.cuda.is_available())
    return torch.device("cuda" if use_cuda else "cpu")


def evaluate_target(
    name: str,
    names: list[str],
    keypoints: np.ndarray,
    scores: np.ndarray,
    mask: np.ndarray,
    threshold: float,
    tolerance: int,
) -> dict[str, Any]:
    index = names.index(name)
    u_float, v_float, visible = keypoints[index].tolist()
    point = (int(round(u_float)), int(round(v_float)))
    score = float(scores[index])
    near_mask = point_near_mask(mask, point, tolerance)
    detected = bool(visible > 0 and score >= threshold and near_mask)
    return {
        "name": name,
        "u": point[0],
        "v": point[1],
        "score": round(score, 6),
        "visible": bool(visible > 0),
        "near_garment_mask": near_mask,
        "detected": detected,
    }


def pair_usable(pair: list[dict[str, Any]], minimum_distance: float) -> bool:
    if not all(item["detected"] for item in pair):
        return False
    p0 = np.array([pair[0]["u"], pair[0]["v"]], dtype=np.float32)
    p1 = np.array([pair[1]["u"], pair[1]["v"]], dtype=np.float32)
    return bool(float(np.linalg.norm(p0 - p1)) >= minimum_distance)


def run_fling_inference(
    frame: np.ndarray,
    mask: np.ndarray,
    model,
    device: torch.device,
    fill_ratio: float,
    batch_size: int,
) -> tuple[dict[str, Any], float]:
    crop, _, crop_origin = square_cloth_crop(frame, mask, fill_ratio)
    inputs, transformations = build_transformed_batch(crop)
    outputs = []
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(inputs), max(1, batch_size)):
            tensor = torch.from_numpy(inputs[start : start + batch_size]).to(device)
            outputs.append(model(tensor).squeeze(1).cpu().numpy())
    value_maps = np.concatenate(outputs, axis=0)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    best = select_grasp_pair(value_maps, transformations, crop.shape[0], crop_origin, mask)
    return best, elapsed_ms


def analyze_frame(
    frame: np.ndarray,
    attempt: int,
    args: argparse.Namespace,
    sam_model,
    fashion_net,
    fling_model,
    device: torch.device,
) -> dict[str, Any]:
    total_started = time.perf_counter()
    mask, sam_ms, sam_details, _ = run_sam2_auto(frame, args.sam_model, sam_model=sam_model)
    x1, y1, x2, y2 = mask_crop_box(mask, args.padding)

    garment_only = np.full_like(frame, 255)
    garment_only[mask > 0] = frame[mask > 0]
    crop = garment_only[y1:y2, x1:x2]

    fashion_started = time.perf_counter()
    keypoints, _, scores = predict_keypoints_with_heatmaps(args.clothing_type, crop, fashion_net)
    fashion_ms = (time.perf_counter() - fashion_started) * 1000.0
    keypoints[:, 0] += x1
    keypoints[:, 1] += y1

    names = KEYPOINT_NAMES[args.clothing_type]
    tolerance = max(5, int(round(max(crop.shape[:2]) * 0.03)))
    config = TARGETS[args.clothing_type]
    shoulders = [
        evaluate_target(name, names, keypoints, scores, mask, args.shoulder_threshold, tolerance)
        for name in config["shoulders"]
    ]
    lower = [
        evaluate_target(name, names, keypoints, scores, mask, args.lower_threshold, tolerance)
        for name in config["lower"]
    ]
    minimum_distance = max(20.0, crop.shape[1] * 0.08)
    shoulders_ok = pair_usable(shoulders, minimum_distance)
    lower_ok = pair_usable(lower, minimum_distance)
    ready = shoulders_ok and lower_ok

    fling_action = None
    fling_ms = None
    error_message = None
    if not ready and not args.no_fling and fling_model is not None and attempt < args.max_attempts:
        try:
            fling_action, fling_ms = run_fling_inference(
                frame,
                mask,
                fling_model,
                device,
                args.cloth_fill_ratio,
                args.batch_size,
            )
        except Exception as error:  # Keep the camera stream alive for an MVP demo.
            error_message = f"FlingBot: {error}"

    if ready:
        state = "GOAL_READY"
    elif attempt >= args.max_attempts:
        state = "MAX_ATTEMPTS_REACHED"
    elif fling_action is not None and fling_action["value"] >= args.fling_threshold:
        state = "UNFOLD_REQUIRED"
    elif fling_action is not None:
        state = "LOW_FLING_VALUE"
    else:
        state = "NOT_READY"

    public = {
        "timestamp": datetime.now().isoformat(timespec="milliseconds"),
        "state": state,
        "attempt": attempt,
        "image_size": {"width": frame.shape[1], "height": frame.shape[0]},
        "mask_crop_box_xyxy": [x1, y1, x2, y2],
        "mask_area_pixels": int(np.count_nonzero(mask)),
        "sam_confidence": float(sam_details["selected_confidence"]),
        "shoulders_usable": shoulders_ok,
        "lower_pair_usable": lower_ok,
        "targets": {"shoulders": shoulders, "lower": lower},
        "fling_action": fling_action,
        "timing_ms": {
            "sam2": round(sam_ms, 3),
            "fashionai": round(fashion_ms, 3),
            "flingbot": round(fling_ms, 3) if fling_ms is not None else None,
            "total": round((time.perf_counter() - total_started) * 1000.0, 3),
        },
        "error": error_message,
        "warning": "Image-space candidates only; calibration and robot safety checks are not applied.",
    }
    return {
        "public": public,
        "mask": mask,
        "source_frame": frame,
        "completed_monotonic": time.monotonic(),
    }


def put_label(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    scale: float = 0.7,
) -> None:
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def render_overlay(
    frame: np.ndarray,
    result: dict[str, Any] | None,
    busy: bool,
    auto_enabled: bool,
    attempt: int,
    max_attempts: int,
) -> np.ndarray:
    output = frame.copy()
    height, width = output.shape[:2]
    panel_height = min(170, max(120, height // 4))
    panel = output[:panel_height].copy()
    panel[:] = (15, 15, 15)
    output[:panel_height] = cv2.addWeighted(output[:panel_height], 0.25, panel, 0.75, 0)

    state = "WAITING FOR FIRST INFERENCE"
    state_color = (0, 220, 255)
    if result is not None:
        public = result["public"]
        state = public["state"]
        state_color = (0, 220, 0) if state == "GOAL_READY" else (0, 165, 255)
        if state in ("MAX_ATTEMPTS_REACHED", "NOT_READY"):
            state_color = (0, 0, 255)

        mask = result["mask"]
        if mask.shape == frame.shape[:2]:
            green = np.zeros_like(output)
            green[mask > 0] = (30, 190, 30)
            output = cv2.addWeighted(output, 0.82, green, 0.18, 0)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(output, contours, -1, (0, 255, 0), 2)

            target_colors = {
                "shoulders": ((255, 80, 80), (80, 80, 255)),
                "lower": ((255, 180, 40), (40, 180, 255)),
            }
            for group_name, targets in public["targets"].items():
                for target, color in zip(targets, target_colors[group_name]):
                    point = (target["u"], target["v"])
                    draw_color = color if target["detected"] else (90, 90, 90)
                    cv2.circle(output, point, 11, draw_color, 3)
                    cv2.drawMarker(output, point, draw_color, cv2.MARKER_CROSS, 22, 2)
                    put_label(
                        output,
                        f"{target['name']} {target['score']:.2f}",
                        (point[0] + 12, max(panel_height + 25, point[1] - 10)),
                        draw_color,
                        0.55,
                    )

            action = public.get("fling_action")
            if action is not None:
                left = (action["left_grasp"]["u"], action["left_grasp"]["v"])
                right = (action["right_grasp"]["u"], action["right_grasp"]["v"])
                cv2.line(output, left, right, (0, 255, 255), 4, cv2.LINE_AA)
                cv2.circle(output, left, 13, (255, 60, 60), -1)
                cv2.circle(output, right, 13, (60, 60, 255), -1)
                put_label(output, "FLING L", (left[0] + 10, left[1] - 10), (255, 60, 60))
                put_label(output, "FLING R", (right[0] + 10, right[1] - 10), (60, 60, 255))

        age = max(0.0, time.monotonic() - result["completed_monotonic"])
        scores = [item["score"] for values in public["targets"].values() for item in values]
        put_label(output, f"scores: " + " / ".join(f"{score:.2f}" for score in scores), (20, 76), (230, 230, 230), 0.6)
        put_label(output, f"result age={age:.1f}s  inference={public['timing_ms']['total']:.0f}ms", (20, 108), (230, 230, 230), 0.6)
        action = public.get("fling_action")
        if action is not None:
            put_label(output, f"FlingBot value={action['value']:.4f}", (20, 140), (0, 255, 255), 0.65)

    put_label(output, f"STATE: {state}", (20, 40), state_color, 0.9)
    right_text = f"attempt {attempt}/{max_attempts} | AI {'BUSY' if busy else 'IDLE'} | auto {'ON' if auto_enabled else 'OFF'}"
    text_size = cv2.getTextSize(right_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    put_label(output, right_text, (max(20, width - text_size[0] - 20), 40), (255, 255, 255), 0.6)
    put_label(output, "SPACE re-check | A auto | S snapshot | R reset | Q quit", (20, height - 20), (255, 255, 255), 0.6)
    return output


def main() -> None:
    args = parse_args()
    args.inference_interval = max(0.2, args.inference_interval)
    args.max_attempts = max(1, args.max_attempts)
    args.record_fps = max(1.0, args.record_fps)
    device = choose_device(args.device)

    print("[INFO] loading SAM2 once for streaming")
    from ultralytics import SAM

    sam_model = SAM(args.sam_model)
    print(f"[INFO] loading FashionAI '{args.clothing_type}'")
    fashion_net = load_net(args.clothing_type, env_id=None)
    fling_model = None
    if not args.no_fling:
        print(f"[INFO] loading FlingBot value network on {device}")
        fling_model = load_value_net(args.fling_checkpoint, device)

    import rclpy
    from cv_bridge import CvBridge
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import Image

    rclpy.init()
    node = rclpy.create_node("garment_inspection_mvp")
    bridge = CvBridge()
    frame_lock = threading.Lock()
    latest_frame: dict[str, Any] = {"image": None, "sequence": 0}

    def image_callback(message: Image) -> None:
        image = bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        with frame_lock:
            latest_frame["image"] = image.copy()
            latest_frame["sequence"] += 1

    qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )
    node.create_subscription(Image, args.ros_topic, image_callback, qos)

    run_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    task_queue: queue.Queue = queue.Queue(maxsize=1)
    result_lock = threading.Lock()
    latest_result: dict[str, Any] = {"value": None}
    events: list[dict[str, Any]] = []
    busy = threading.Event()
    stop = threading.Event()

    def worker() -> None:
        while not stop.is_set():
            try:
                task = task_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if task is None:
                break
            frame, attempt, sequence = task
            busy.set()
            try:
                result = analyze_frame(
                    frame, attempt, args, sam_model, fashion_net, fling_model, device
                )
                result["public"]["frame_sequence"] = sequence
                with result_lock:
                    latest_result["value"] = result
                    events.append(json_safe(result["public"]))
                cv2.imwrite(str(run_dir / "latest_input.png"), frame)
                cv2.imwrite(str(run_dir / "latest_mask.png"), result["mask"])
                with open(run_dir / "latest_result.json", "w", encoding="utf-8") as stream:
                    json.dump(json_safe(result["public"]), stream, indent=2, ensure_ascii=False)
                print(
                    f"[RESULT] {result['public']['state']} attempt={attempt} "
                    f"total={result['public']['timing_ms']['total']:.0f}ms"
                )
            except Exception as error:
                print(f"[ERROR] inference failed: {error}")
                with result_lock:
                    events.append(
                        {
                            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
                            "state": "INFERENCE_ERROR",
                            "attempt": attempt,
                            "error": str(error),
                        }
                    )
            finally:
                busy.clear()
                task_queue.task_done()

    thread = threading.Thread(target=worker, name="garment-inference", daemon=True)
    thread.start()

    print(f"[INFO] waiting for ROS image: {args.ros_topic}")
    print("[KEYS] SPACE re-check | A auto | S snapshot | R reset | Q quit")
    auto_enabled = True
    attempt = 1
    last_submit = -1e9
    force_submit = True
    writer = None
    last_video_write = -1e9
    snapshot_index = 0

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.001)
            with frame_lock:
                frame = None if latest_frame["image"] is None else latest_frame["image"].copy()
                sequence = latest_frame["sequence"]
            if frame is None:
                time.sleep(0.01)
                continue

            now = time.monotonic()
            due = auto_enabled and now - last_submit >= args.inference_interval
            if (force_submit or due) and not busy.is_set() and task_queue.empty():
                task_queue.put((frame.copy(), attempt, sequence))
                last_submit = now
                force_submit = False

            with result_lock:
                result = latest_result["value"]
            overlay = render_overlay(frame, result, busy.is_set(), auto_enabled, attempt, args.max_attempts)

            if writer is None and not args.no_record:
                video_path = run_dir / "demo.mp4"
                writer = cv2.VideoWriter(
                    str(video_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    args.record_fps,
                    (overlay.shape[1], overlay.shape[0]),
                )
                if not writer.isOpened():
                    print("[WARN] video writer could not be opened; recording disabled")
                    writer = None
                    args.no_record = True
            # The GUI loop can run much faster than the camera.  Throttle file
            # writes so the saved video's playback duration remains realistic.
            if writer is not None and now - last_video_write >= 1.0 / args.record_fps:
                writer.write(overlay)
                last_video_write = now

            preview = overlay
            if args.display_width > 0 and preview.shape[1] > args.display_width:
                scale = args.display_width / preview.shape[1]
                preview = cv2.resize(preview, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            cv2.imshow("DUARO garment inspection MVP", preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("a"):
                auto_enabled = not auto_enabled
                print(f"[INFO] automatic inference {'enabled' if auto_enabled else 'disabled'}")
            elif key == ord(" "):
                attempt = min(args.max_attempts, attempt + 1)
                with result_lock:
                    latest_result["value"] = None
                force_submit = True
                print(f"[INFO] manual re-check requested, attempt={attempt}")
            elif key == ord("r"):
                attempt = 1
                with result_lock:
                    latest_result["value"] = None
                force_submit = True
                print("[INFO] inspection cycle reset")
            elif key == ord("s"):
                snapshot_index += 1
                path = run_dir / f"snapshot_{snapshot_index:03d}.png"
                cv2.imwrite(str(path), overlay)
                print(f"[SAVED] {path}")
    finally:
        stop.set()
        try:
            task_queue.put_nowait(None)
        except queue.Full:
            pass
        thread.join(timeout=5.0)
        if writer is not None:
            writer.release()
        with open(run_dir / "session.json", "w", encoding="utf-8") as stream:
            json.dump(
                {
                    "ros_topic": args.ros_topic,
                    "clothing_type": args.clothing_type,
                    "thresholds": {
                        "shoulder": args.shoulder_threshold,
                        "lower": args.lower_threshold,
                        "fling": args.fling_threshold,
                    },
                    "events": events,
                },
                stream,
                indent=2,
                ensure_ascii=False,
            )
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        print(f"[DONE] session saved under: {run_dir}")


if __name__ == "__main__":
    main()
