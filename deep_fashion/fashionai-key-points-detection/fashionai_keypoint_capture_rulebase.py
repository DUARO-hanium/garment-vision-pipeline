"""Capture/test FashionAI clothing keypoints and save images plus JSON.

This is a thin test runner around ailia-models/deep_fashion/
fashionai-key-points-detection. It accepts a UVC/OpenCV camera, a video file,
one image, or a folder of images and writes per-frame annotated images plus
pixel keypoints in (u, v) image coordinates.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


MODEL_DIR = Path(__file__).resolve().parent
AILIA_MODELS_DIR = MODEL_DIR.parent.parent
UTIL_DIR = AILIA_MODELS_DIR / "util"

sys.path.insert(0, str(MODEL_DIR))
sys.path.insert(0, str(UTIL_DIR))

from fashionai_key_points_detection_utils import decode_np, draw_keypoints  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402


REMOTE_PATH = "https://storage.googleapis.com/ailia-models/fashionai-key-points-detection/"
IMAGE_SIZE = 512
MU = 0.65
SIGMA = 0.25
HM_STRIDE = 4

MODEL_FILES = {
    "blouse": ("blouse_100.onnx", "blouse_100.onnx.prototxt"),
    "dress": ("dress_100.onnx", "dress_100.onnx.prototxt"),
    "outwear": ("outwear_100.onnx", "outwear_100.onnx.prototxt"),
    "skirt": ("skirt_100.onnx", "skirt_100.onnx.prototxt"),
    "trousers": ("trousers_100.onnx", "trousers_100.onnx.prototxt"),
}

KEYPOINT_NAMES = {
    "blouse": [
        "neckline_left",
        "neckline_right",
        "center_front",
        "shoulder_left",
        "shoulder_right",
        "armpit_left",
        "armpit_right",
        "cuff_left_in",
        "cuff_left_out",
        "cuff_right_in",
        "cuff_right_out",
        "top_hem_left",
        "top_hem_right",
    ],
    "outwear": [
        "neckline_left",
        "neckline_right",
        "shoulder_left",
        "shoulder_right",
        "armpit_left",
        "armpit_right",
        "waistline_left",
        "waistline_right",
        "cuff_left_in",
        "cuff_left_out",
        "cuff_right_in",
        "cuff_right_out",
        "top_hem_left",
        "top_hem_right",
    ],
    "trousers": [
        "waistband_left",
        "waistband_right",
        "crotch",
        "bottom_left_in",
        "bottom_left_out",
        "bottom_right_in",
        "bottom_right_out",
    ],
    "skirt": [
        "waistband_left",
        "waistband_right",
        "hemline_left",
        "hemline_right",
    ],
    "dress": [
        "neckline_left",
        "neckline_right",
        "center_front",
        "shoulder_left",
        "shoulder_right",
        "armpit_left",
        "armpit_right",
        "waistline_left",
        "waistline_right",
        "cuff_left_in",
        "cuff_left_out",
        "cuff_right_in",
        "cuff_right_out",
        "hemline_left",
        "hemline_right",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Save FashionAI clothing keypoint candidates from camera/video/images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "image_path",
        nargs="?",
        help="Optional single image path. Same as --input, convenient for quick tests.",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--video", help="OpenCV camera id, video path, or stream URL. Use 0 for default UVC camera.")
    source.add_argument("--input", "--image-path", dest="input", help="Single image path.")
    source.add_argument("--input-dir", help="Folder containing images.")
    parser.add_argument(
        "-c",
        "--clothing-type",
        choices=sorted(MODEL_FILES),
        default="blouse",
        help="FashionAI clothing model to use.",
    )
    parser.add_argument("--output-dir", default="outputs/fashionai_keypoints", help="Base output folder.")
    parser.add_argument(
        "--env_id",
        type=int,
        default=None,
        help="Deprecated compatibility option; ONNX Runtime selects the CPU automatically.",
    )
    parser.add_argument("--fps", type=float, default=30.0, help="Requested camera FPS and saved metadata FPS.")
    parser.add_argument("--width", type=int, default=None, help="Requested camera width.")
    parser.add_argument("--height", type=int, default=None, help="Requested camera height.")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N processed video frames. 0 means unlimited.")
    parser.add_argument("--save-every", type=int, default=1, help="Save every Nth processed frame.")
    parser.add_argument("--show", action="store_true", help="Show annotated preview window. Press q to quit.")
    parser.add_argument("--save-raw", action="store_true", help="Also save raw input frames/images.")
    parser.add_argument("--no-labels", action="store_true", help="Do not draw keypoint index legend on output images.")
    parser.add_argument(
        "--viz-max-side",
        type=int,
        default=1200,
        help="Resize saved/preview visualization so its longest image side is at most this size. Use 0 to keep original size.",
    )
    args = parser.parse_args()
    selected = sum(
        value is not None
        for value in (args.image_path, args.video, args.input, args.input_dir)
    )
    if selected != 1:
        parser.error("provide exactly one source: image_path, --input/--image-path, --input-dir, or --video")
    if args.image_path:
        args.input = args.image_path
    return args


def preprocess(img: np.ndarray, img_size: tuple[int, int]) -> np.ndarray:
    img_w, img_h = img_size
    img = cv2.resize(img, (img_w, img_h), interpolation=cv2.INTER_CUBIC)
    img = np.transpose(img, (2, 0, 1)).astype(np.float32)
    img[[0, 2]] = img[[2, 0]]
    img = img / 255.0
    img = (img - MU) / SIGMA
    pad_imgs = np.zeros([1, 3, IMAGE_SIZE, IMAGE_SIZE], dtype=np.float32)
    pad_imgs[0, :, :img_h, :img_w] = img
    return pad_imgs


def post_process(
    clothing_type: str,
    hm_pred: np.ndarray,
    hm_pred2: np.ndarray,
    info: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    keypoint_names = KEYPOINT_NAMES[clothing_type]
    conjug = []
    for i, key in enumerate(keypoint_names):
        if "left" in key:
            j = keypoint_names.index(key.replace("left", "right"))
            conjug.append([i, j])

    flipped = np.zeros_like(hm_pred2)
    img_w2 = int(info["img_w2"])
    flipped[:, :, : img_w2 // HM_STRIDE] = np.flip(hm_pred2[:, :, : img_w2 // HM_STRIDE], 2)
    for conj in conjug:
        flipped[conj] = flipped[conj[::-1]]

    combined_heatmap = hm_pred + flipped
    scores = np.max(combined_heatmap.reshape(combined_heatmap.shape[0], -1), axis=1)

    x, y = decode_np(
        combined_heatmap,
        info["scale"],
        HM_STRIDE,
        (info["img_w"] / 2, info["img_h"] / 2),
        method="maxoffset",
    )
    keypoints = np.stack([x, y, np.ones(x.shape)], axis=1).astype(np.int16)
    return keypoints, scores.astype(np.float32)


def predict_keypoints(clothing_type: str, img_bgr: np.ndarray, net) -> tuple[np.ndarray, np.ndarray]:
    img_flip = cv2.flip(img_bgr, 1)
    img_h, img_w, _ = img_bgr.shape
    scale = IMAGE_SIZE / max(img_w, img_h)
    img_h2 = int(img_h * scale)
    img_w2 = int(img_w * scale)

    model_input = preprocess(img_bgr, (img_w2, img_h2))
    model_input_flip = preprocess(img_flip, (img_w2, img_h2))

    _, hm_pred = net.predict({"img": model_input})
    _, hm_pred2 = net.predict({"img": model_input_flip})
    hm_pred = np.maximum(hm_pred[0], 0)
    hm_pred2 = np.maximum(hm_pred2[0], 0)

    info = {
        "img_h": img_h,
        "img_w": img_w,
        "img_h2": img_h2,
        "img_w2": img_w2,
        "scale": scale,
    }
    return post_process(clothing_type, hm_pred, hm_pred2, info)


def make_output_dirs(base_dir: str, save_raw: bool) -> dict[str, Path]:
    run_dir = Path(base_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    dirs = {
        "run": run_dir,
        "images": run_dir / "images",
        "json": run_dir / "json",
    }
    if save_raw:
        dirs["raw"] = run_dir / "raw"
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def keypoints_to_json(
    clothing_type: str,
    keypoints: np.ndarray,
    scores: np.ndarray,
    frame_name: str,
    image_shape: tuple[int, int, int],
    elapsed_ms: float,
) -> dict:
    h, w = image_shape[:2]
    points = []
    max_score = float(np.max(scores)) if len(scores) else 0.0
    for name, (u, v, visible), score in zip(KEYPOINT_NAMES[clothing_type], keypoints.tolist(), scores.tolist()):
        heatmap_score = float(score)
        normalized_score = heatmap_score / max_score if max_score > 0 else 0.0
        points.append(
            {
                "name": name,
                "u": int(u),
                "v": int(v),
                "visible": bool(visible > 0),
                "score": round(normalized_score, 6),
                "heatmap_score": round(heatmap_score, 6),
            }
        )

    grasp_candidates = build_grasp_candidates(clothing_type, points)

    return {
        "frame": frame_name,
        "clothing_type": clothing_type,
        "image": {
            "width": int(w),
            "height": int(h),
            "coordinate_format": "(u, v)",
            "u_axis": "image x coordinate, left to right",
            "v_axis": "image y coordinate, top to bottom",
        },
        "elapsed_ms": round(elapsed_ms, 3),
        "keypoints": points,
        "grasp_candidates": grasp_candidates,
    }


def build_grasp_candidates(clothing_type: str, points: list[dict]) -> list[dict]:
    point_by_name = {point["name"]: point for point in points}
    shoulder_names = ("shoulder_left", "shoulder_right")
    if not all(name in point_by_name for name in shoulder_names):
        return []

    left = point_by_name["shoulder_left"]
    right = point_by_name["shoulder_right"]
    if not left["visible"] or not right["visible"]:
        return []

    pair_score = min(left["score"], right["score"])
    return [
        {
            "pair_name": "both_shoulders",
            "strategy": "rule_base_shoulder_pair",
            "description": "Use left and right shoulder keypoints as a dual-arm grasp candidate.",
            "left": left,
            "right": right,
            "pair_score": round(float(pair_score), 6),
            "coordinate_format": "(u, v)",
        }
    ]


def draw_keypoint_legend(
    image: np.ndarray,
    clothing_type: str,
    keypoints: np.ndarray,
) -> np.ndarray:
    names = KEYPOINT_NAMES[clothing_type]
    h, w = image.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    badge_font_scale = 0.4
    legend_font_scale = 0.44
    thickness = 1
    badge_color = (48, 103, 194)
    badge_text_color = (255, 255, 255)
    legend_width = max(245, int(w * 0.45))
    margin = 18

    canvas = np.full((h, w + legend_width, 3), 255, dtype=np.uint8)
    canvas[:, :w] = image
    cv2.line(canvas, (w, 0), (w, h - 1), (220, 220, 220), 1)

    for idx, kpt in enumerate(keypoints.tolist(), start=1):
        u, v, visible = kpt
        if visible <= 0:
            continue

        label = str(idx)
        (text_w, text_h), baseline = cv2.getTextSize(label, font, badge_font_scale, thickness)
        radius = max(8, int(max(text_w, text_h) * 0.75))
        cx = int(np.clip(u, radius, max(w - radius - 1, radius)))
        cy = int(np.clip(v, radius, max(h - radius - 1, radius)))
        cv2.circle(canvas, (cx, cy), radius, badge_color, -1)
        cv2.circle(canvas, (cx, cy), radius, (255, 255, 255), 1)
        text_x = cx - text_w // 2
        text_y = cy + text_h // 2
        cv2.putText(canvas, label, (text_x, text_y), font, badge_font_scale, badge_text_color, thickness, cv2.LINE_AA)

    line_height = 20
    x0 = w + margin
    y0 = margin + 8
    for idx, name in enumerate(names, start=1):
        label = f"{idx}.  {name}"
        y = y0 + (idx - 1) * line_height
        if y > h - margin:
            break
        cv2.putText(canvas, label, (x0, y), font, legend_font_scale, (20, 20, 20), thickness, cv2.LINE_AA)

    return canvas


def make_visualization(
    frame_bgr: np.ndarray,
    clothing_type: str,
    keypoints: np.ndarray,
    draw_labels: bool,
    max_side: int,
) -> np.ndarray:
    viz = frame_bgr.copy()
    viz_keypoints = keypoints.copy()

    if max_side and max(frame_bgr.shape[:2]) > max_side:
        h, w = frame_bgr.shape[:2]
        scale = max_side / max(h, w)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        viz = cv2.resize(viz, (new_w, new_h), interpolation=cv2.INTER_AREA)
        viz_keypoints[:, 0] = np.rint(viz_keypoints[:, 0] * scale)
        viz_keypoints[:, 1] = np.rint(viz_keypoints[:, 1] * scale)

    annotated = draw_keypoints(viz, viz_keypoints)
    if draw_labels:
        annotated = draw_keypoint_legend(annotated, clothing_type, viz_keypoints)
    return annotated


def save_result(
    dirs: dict[str, Path],
    clothing_type: str,
    frame_bgr: np.ndarray,
    keypoints: np.ndarray,
    scores: np.ndarray,
    index: int,
    elapsed_ms: float,
    save_raw: bool,
    draw_labels: bool,
    viz_max_side: int,
) -> None:
    stem = f"frame_{index:06d}"
    annotated = make_visualization(frame_bgr, clothing_type, keypoints, draw_labels, viz_max_side)
    cv2.imwrite(str(dirs["images"] / f"{stem}.png"), annotated)
    if save_raw:
        cv2.imwrite(str(dirs["raw"] / f"{stem}.png"), frame_bgr)

    payload = keypoints_to_json(clothing_type, keypoints, scores, f"{stem}.png", frame_bgr.shape, elapsed_ms)
    with open(dirs["json"] / f"{stem}.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


class OnnxRuntimeNet:
    """Small adapter that provides the ailia-style predict() API used below."""

    def __init__(self, weight_path: Path):
        try:
            import onnxruntime as ort
        except ImportError as error:
            raise RuntimeError(
                "onnxruntime is required. Install it with: pip install onnxruntime"
            ) from error

        self.session = ort.InferenceSession(
            str(weight_path),
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, inputs: dict[str, np.ndarray]):
        input_value = inputs.get(self.input_name)
        if input_value is None and len(inputs) == 1:
            input_value = next(iter(inputs.values()))
        if input_value is None:
            raise KeyError(f"Model input '{self.input_name}' was not provided")
        return self.session.run(None, {self.input_name: input_value})


def load_net(clothing_type: str, env_id: int | None):
    weight_name, model_name = MODEL_FILES[clothing_type]
    weight_path = MODEL_DIR / weight_name
    model_path = MODEL_DIR / model_name
    check_and_download_models(str(weight_path), str(model_path), REMOTE_PATH)

    if env_id is not None:
        print("[WARN] --env_id is ignored when using ONNX Runtime")
    print("[INFO] inference backend: ONNX Runtime (CPU)")
    return OnnxRuntimeNet(weight_path)


def iter_image_paths(path: str) -> list[Path]:
    exts = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    return sorted(p for p in Path(path).iterdir() if p.suffix.lower() in exts)


def process_image_paths(args: argparse.Namespace, net: ailia.Net, dirs: dict[str, Path]) -> int:
    paths = [Path(args.input)] if args.input else iter_image_paths(args.input_dir)
    saved = 0
    for i, path in enumerate(paths, start=1):
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            print(f"[WARN] skipped unreadable image: {path}")
            continue
        start = time.perf_counter()
        keypoints, scores = predict_keypoints(args.clothing_type, frame, net)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        save_result(
            dirs,
            args.clothing_type,
            frame,
            keypoints,
            scores,
            i,
            elapsed_ms,
            args.save_raw,
            not args.no_labels,
            args.viz_max_side,
        )
        saved += 1
        print(f"[SAVE] {path.name} -> frame_{i:06d}.png/json ({elapsed_ms:.1f} ms)")
    return saved


def open_capture(video: str, args: argparse.Namespace) -> cv2.VideoCapture:
    try:
        source = int(video)
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    except ValueError:
        cap = cv2.VideoCapture(video)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video/camera source: {video}")

    cap.set(cv2.CAP_PROP_FPS, args.fps)
    if args.width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    return cap


def process_video(args: argparse.Namespace, net: ailia.Net, dirs: dict[str, Path]) -> int:
    cap = open_capture(args.video, args)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] capture opened: {actual_w}x{actual_h}, fps={actual_fps:.2f}")

    processed = 0
    saved = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] no frame received; stopping")
                break

            processed += 1
            start = time.perf_counter()
            keypoints, scores = predict_keypoints(args.clothing_type, frame, net)
            elapsed_ms = (time.perf_counter() - start) * 1000.0

            if processed % args.save_every == 0:
                saved += 1
                save_result(
                    dirs,
                    args.clothing_type,
                    frame,
                    keypoints,
                    scores,
                    saved,
                    elapsed_ms,
                    args.save_raw,
                    not args.no_labels,
                    args.viz_max_side,
                )
                print(f"[SAVE] frame {processed} -> frame_{saved:06d}.png/json ({elapsed_ms:.1f} ms)")

            if args.show:
                preview = make_visualization(
                    frame,
                    args.clothing_type,
                    keypoints,
                    not args.no_labels,
                    args.viz_max_side,
                )
                cv2.imshow("fashionai keypoints", preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if args.max_frames and processed >= args.max_frames:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return saved


def write_manifest(args: argparse.Namespace, dirs: dict[str, Path], saved_count: int) -> None:
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "clothing_type": args.clothing_type,
        "source": {
            "video": args.video,
            "input": args.input,
            "input_dir": args.input_dir,
        },
        "requested_fps": args.fps,
        "save_every": args.save_every,
        "saved_count": saved_count,
        "json_coordinate_format": "(u, v)",
    }
    with open(dirs["run"] / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    if args.save_every < 1:
        raise ValueError("--save-every must be >= 1")

    dirs = make_output_dirs(args.output_dir, args.save_raw)
    net = load_net(args.clothing_type, args.env_id)

    if args.video is not None:
        saved_count = process_video(args, net, dirs)
    else:
        saved_count = process_image_paths(args, net, dirs)

    write_manifest(args, dirs, saved_count)
    print(f"[DONE] saved {saved_count} result(s) under: {dirs['run']}")


if __name__ == "__main__":
    main()
