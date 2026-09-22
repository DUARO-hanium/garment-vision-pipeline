"""Live garment-axis inspection MVP for two fixed robot arms.

State flow:
  SHOULDER : infer the full garment axis, then select the longest safe horizontal
             grasp span near the FashionAI shoulder pair.
  HEM      : after a physical 180-degree yaw, infer the same full garment axis
             and select the longest safe horizontal span near the hem pair.

All returned coordinates are image-space candidates.  Calibration, collision
checking, force control, and robot commands are intentionally outside this MVP.
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

from fashionai_keypoint_capture import KEYPOINT_NAMES, load_net, predict_keypoints_with_heatmaps
from sam2_garment_keypoint_once import mask_crop_box, point_near_mask
from sam3_fal_segmenter import FalSam3Segmenter


PHASE_SHOULDER = "SHOULDER"
PHASE_HEM = "HEM"
PHASE_COMPLETE = "COMPLETE"

TARGETS = {
    "blouse": {
        "shoulder": ("shoulder_left", "shoulder_right"),
        "hem": ("top_hem_left", "top_hem_right"),
    },
    "outwear": {
        "shoulder": ("shoulder_left", "shoulder_right"),
        "hem": ("top_hem_left", "top_hem_right"),
    },
    "dress": {
        "shoulder": ("shoulder_left", "shoulder_right"),
        "hem": ("hemline_left", "hemline_right"),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Two-arm ROI garment inspection MVP")
    parser.add_argument("--ros-topic", default="/camera1/image_raw")
    parser.add_argument("--clothing-type", choices=sorted(TARGETS), default="blouse")
    parser.add_argument(
        "--inference-interval",
        type=float,
        default=5.0,
        help="Minimum seconds between cloud inference calls (not a periodic timer)",
    )
    parser.add_argument(
        "--scene-change-ratio",
        type=float,
        default=0.05,
        help="Fraction of sampled pixels that must change before inference is requested",
    )
    parser.add_argument(
        "--scene-change-pixel-threshold",
        type=float,
        default=25.0,
        help="Per-pixel grayscale difference used by the scene-change detector",
    )
    parser.add_argument(
        "--scene-motion-ratio",
        type=float,
        default=0.01,
        help="Adjacent-frame changed-pixel ratio considered active motion",
    )
    parser.add_argument(
        "--scene-settle-seconds",
        type=float,
        default=1.5,
        help="How long the scene must remain still before models run",
    )
    parser.add_argument("--scene-sample-width", type=int, default=320)
    parser.add_argument(
        "--sam3-prompt",
        default="the single garment lying on the table",
        help="Text prompt sent to fal-ai/sam-3/image",
    )
    parser.add_argument("--sam3-input-size", type=int, default=1024)
    parser.add_argument("--keypoint-threshold", type=float, default=150.0)
    parser.add_argument("--padding", type=float, default=0.08)
    parser.add_argument("--roi-top", type=float, default=0.04)
    parser.add_argument("--roi-bottom", type=float, default=0.68)
    parser.add_argument("--roi-margin-x", type=float, default=0.04)
    parser.add_argument("--roi-center-overlap", type=float, default=0.04)
    parser.add_argument("--max-pair-angle", type=float, default=20.0)
    parser.add_argument("--min-pair-distance", type=float, default=0.12)
    parser.add_argument("--max-pair-distance", type=float, default=0.90)
    parser.add_argument("--grasp-search-radius", type=float, default=0.07)
    parser.add_argument(
        "--keypoint-inset-ratio",
        type=float,
        default=0.018,
        help="Target segmentation-mask clearance for confident FashionAI points, as image diagonal ratio",
    )
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--display-width", type=int, default=1280)
    parser.add_argument(
        "--display-stage-seconds",
        type=float,
        default=2.0,
        help="Seconds to show SAM3-only and FashionAI-only stages before the final decision",
    )
    parser.add_argument(
        "--pipeline-panel-width",
        type=int,
        default=360,
        help="Width of the left pipeline highlight panel; use 0 to hide it",
    )
    parser.add_argument("--record-fps", type=float, default=20.0)
    parser.add_argument("--output-dir", default="outputs/garment_inspection_roi_mvp")
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


def scene_signature(frame: np.ndarray, sample_width: int) -> np.ndarray:
    """Create a cheap, illumination-tolerant signature without running a model."""
    height, width = frame.shape[:2]
    target_width = min(max(80, int(sample_width)), width)
    target_height = max(1, int(round(height * target_width / width)))
    small = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # Removing the mean makes automatic exposure shifts less likely to trigger
    # a paid cloud request while preserving garment edge/shape changes.
    centered = gray.astype(np.float32) - float(np.mean(gray))
    return centered


def scene_change_ratio(
    first: np.ndarray | None,
    second: np.ndarray,
    pixel_threshold: float,
) -> float:
    if first is None or first.shape != second.shape:
        return 1.0
    difference = np.abs(first - second)
    return float(np.mean(difference >= pixel_threshold))


def roi_rectangles(
    width: int,
    height: int,
    top_ratio: float,
    bottom_ratio: float,
    margin_x_ratio: float,
    center_overlap_ratio: float,
) -> dict[str, tuple[int, int, int, int]]:
    top = int(round(np.clip(top_ratio, 0.0, 0.95) * height))
    bottom = int(round(np.clip(bottom_ratio, top_ratio + 0.05, 1.0) * height))
    margin = int(round(np.clip(margin_x_ratio, 0.0, 0.45) * width))
    overlap = int(round(np.clip(center_overlap_ratio, 0.0, 0.25) * width))
    center = width // 2
    return {
        "left": (margin, top, min(width, center + overlap), bottom),
        "right": (max(0, center - overlap), top, max(center + 1, width - margin), bottom),
    }


def point_in_rect(point: tuple[int, int], rect: tuple[int, int, int, int]) -> bool:
    u, v = point
    x1, y1, x2, y2 = rect
    return x1 <= u < x2 and y1 <= v < y2


def mask_orientation(mask: np.ndarray) -> float | None:
    ys, xs = np.where(mask > 0)
    if len(xs) < 10:
        return None
    points = np.column_stack((xs, ys)).astype(np.float32)
    mean = points.mean(axis=0, keepdims=True)
    centered = points - mean
    covariance = centered.T @ centered / max(1, len(points) - 1)
    values, vectors = np.linalg.eigh(covariance)
    axis = vectors[:, int(np.argmax(values))]
    angle = float(np.degrees(np.arctan2(axis[1], axis[0])))
    if angle > 90.0:
        angle -= 180.0
    if angle < -90.0:
        angle += 180.0
    return angle


def pair_geometry(
    points: list[tuple[int, int]],
    image_width: int,
    max_angle: float,
    min_distance_ratio: float,
    max_distance_ratio: float,
) -> dict[str, Any]:
    if len(points) != 2:
        return {"valid": False, "reason": "pair_missing"}
    left, right = sorted(points, key=lambda item: item[0])
    du = float(right[0] - left[0])
    dv = float(right[1] - left[1])
    distance = float(np.hypot(du, dv))
    angle = float(np.degrees(np.arctan2(dv, max(1e-6, du))))
    distance_ratio = distance / max(1.0, float(image_width))
    angle_ok = abs(angle) <= max_angle
    distance_ok = min_distance_ratio <= distance_ratio <= max_distance_ratio
    return {
        "valid": bool(angle_ok and distance_ok),
        "angle_deg": round(angle, 3),
        "horizontal_angle_ok": bool(angle_ok),
        "distance_pixels": round(distance, 3),
        "distance_ratio_of_width": round(distance_ratio, 6),
        "distance_ok": bool(distance_ok),
        "left_point": {"u": left[0], "v": left[1]},
        "right_point": {"u": right[0], "v": right[1]},
        "reason": None if angle_ok and distance_ok else (
            "pair_not_horizontal" if not angle_ok else "pair_distance_out_of_range"
        ),
    }


def best_mask_point_in_roi(
    mask: np.ndarray,
    distance: np.ndarray,
    rect: tuple[int, int, int, int],
) -> tuple[int, int] | None:
    x1, y1, x2, y2 = rect
    roi_mask = mask[y1:y2, x1:x2] > 0
    ys, xs = np.where(roi_mask)
    if len(xs) == 0:
        return None

    # Only consider the robot-near part of the garment in this arm's ROI.
    global_ys = ys + y1
    near_limit = float(np.quantile(global_ys, 0.35))
    near = global_ys <= near_limit
    candidate_ys = global_ys[near]
    candidate_xs = (xs + x1)[near]
    if len(candidate_xs) == 0:
        return None

    clearance = distance[candidate_ys, candidate_xs]
    vertical_penalty = (candidate_ys - y1).astype(np.float32) * 0.04
    scores = clearance - vertical_penalty
    index = int(np.argmax(scores))
    return int(candidate_xs[index]), int(candidate_ys[index])


def segmentation_grasp_pair(
    mask: np.ndarray,
    rois: dict[str, tuple[int, int, int, int]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    binary = (mask > 0).astype(np.uint8)
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    points = []
    for arm in ("left", "right"):
        point = best_mask_point_in_roi(mask, distance, rois[arm])
        if point is None:
            continue
        points.append(
            {
                "arm": arm,
                "u": point[0],
                "v": point[1],
                "mask_clearance_pixels": round(float(distance[point[1], point[0]]), 3),
            }
        )
    geometry = pair_geometry(
        [(item["u"], item["v"]) for item in points],
        mask.shape[1],
        args.max_pair_angle,
        args.min_pair_distance,
        args.max_pair_distance,
    )
    return points, geometry


def refine_semantic_point(
    raw_point: tuple[int, int],
    mask: np.ndarray,
    distance: np.ndarray,
    rect: tuple[int, int, int, int],
    radius: int,
) -> tuple[int, int] | None:
    raw_u, raw_v = raw_point
    x1 = max(rect[0], raw_u - radius)
    y1 = max(rect[1], raw_v - radius)
    x2 = min(rect[2], raw_u + radius + 1)
    y2 = min(rect[3], raw_v + radius + 1)
    if x1 >= x2 or y1 >= y2:
        return None
    patch_mask = mask[y1:y2, x1:x2] > 0
    ys, xs = np.where(patch_mask)
    if len(xs) == 0:
        return None
    us = xs + x1
    vs = ys + y1
    clearance = distance[vs, us]
    semantic_distance = np.hypot(us - raw_u, vs - raw_v)
    scores = clearance - 0.18 * semantic_distance
    index = int(np.argmax(scores))
    return int(us[index]), int(vs[index])


def semantic_grasp_pair(
    phase: str,
    clothing_type: str,
    keypoints: np.ndarray,
    scores: np.ndarray,
    mask: np.ndarray,
    rois: dict[str, tuple[int, int, int, int]],
    crop_shape: tuple[int, int],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    group = "shoulder" if phase == PHASE_SHOULDER else "hem"
    names = KEYPOINT_NAMES[clothing_type]
    wanted = TARGETS[clothing_type][group]
    tolerance = max(5, int(round(max(crop_shape) * 0.03)))
    raw_targets = []
    for name in wanted:
        index = names.index(name)
        raw_u, raw_v, visible = keypoints[index].tolist()
        point = (int(round(raw_u)), int(round(raw_v)))
        score = float(scores[index])
        raw_targets.append(
            {
                "name": name,
                "raw_u": point[0],
                "raw_v": point[1],
                "score": round(score, 6),
                "visible": bool(visible > 0),
                "near_garment_mask": point_near_mask(mask, point, tolerance),
                "threshold_passed": bool(visible > 0 and score >= args.keypoint_threshold),
            }
        )

    # Physical arm assignment is based on camera x, not semantic left/right.
    raw_targets.sort(key=lambda item: item["raw_u"])
    distance = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    radius = max(8, int(round(np.hypot(mask.shape[1], mask.shape[0]) * args.grasp_search_radius)))
    refined = []
    for arm, target in zip(("left", "right"), raw_targets):
        item = dict(target)
        item["arm"] = arm
        item["inside_arm_roi_raw"] = point_in_rect((item["raw_u"], item["raw_v"]), rois[arm])
        grasp = None
        if item["threshold_passed"] and item["near_garment_mask"]:
            grasp = refine_semantic_point(
                (item["raw_u"], item["raw_v"]), mask, distance, rois[arm], radius
            )
        item["grasp_u"] = grasp[0] if grasp is not None else None
        item["grasp_v"] = grasp[1] if grasp is not None else None
        item["grasp_found"] = grasp is not None
        item["mask_clearance_pixels"] = (
            round(float(distance[grasp[1], grasp[0]]), 3) if grasp is not None else None
        )
        refined.append(item)

    geometry = pair_geometry(
        [(item["grasp_u"], item["grasp_v"]) for item in refined if item["grasp_found"]],
        mask.shape[1],
        args.max_pair_angle,
        args.min_pair_distance,
        args.max_pair_distance,
    )
    return refined, geometry


def normalize_horizontal_angle(angle: float) -> float:
    """Return the absolute deviation from an image-horizontal line."""
    angle = ((angle + 90.0) % 180.0) - 90.0
    return abs(float(angle))


def semantic_record(
    name: str,
    names: list[str],
    keypoints: np.ndarray,
    scores: np.ndarray,
    mask: np.ndarray,
    threshold: float,
    tolerance: int,
) -> dict[str, Any]:
    index = names.index(name)
    u, v, visible = keypoints[index].tolist()
    point = (int(round(u)), int(round(v)))
    near_mask = point_near_mask(mask, point, tolerance)
    score = float(scores[index])
    confidence_passed = bool(visible > 0 and score >= threshold)
    return {
        "name": name,
        "u": point[0],
        "v": point[1],
        "score": round(score, 6),
        "visible": bool(visible > 0),
        "confidence_passed": confidence_passed,
        "near_garment_mask": near_mask,
        "usable": bool(confidence_passed and near_mask),
    }


def compute_garment_axes(
    shoulders: list[dict[str, Any]],
    hems: list[dict[str, Any]],
    max_angle: float,
) -> dict[str, Any]:
    shoulder_points = sorted([(p["u"], p["v"]) for p in shoulders], key=lambda p: p[0])
    hem_points = sorted([(p["u"], p["v"]) for p in hems], key=lambda p: p[0])
    sl, sr = np.asarray(shoulder_points[0], np.float32), np.asarray(shoulder_points[1], np.float32)
    hl, hr = np.asarray(hem_points[0], np.float32), np.asarray(hem_points[1], np.float32)
    shoulder_center = (sl + sr) / 2.0
    hem_center = (hl + hr) / 2.0
    longitudinal = hem_center - shoulder_center
    shoulder_angle = float(np.degrees(np.arctan2((sr - sl)[1], (sr - sl)[0])))
    hem_angle = float(np.degrees(np.arctan2((hr - hl)[1], (hr - hl)[0])))
    longitudinal_angle = float(np.degrees(np.arctan2(longitudinal[1], longitudinal[0])))
    vertical_deviation = abs(90.0 - abs(longitudinal_angle))
    shoulder_deviation = normalize_horizontal_angle(shoulder_angle)
    hem_deviation = normalize_horizontal_angle(hem_angle)
    orientation_ok = bool(
        shoulder_deviation <= max_angle
        and hem_deviation <= max_angle
        and vertical_deviation <= max_angle
    )
    return {
        "shoulder_left": shoulder_points[0],
        "shoulder_right": shoulder_points[1],
        "hem_left": hem_points[0],
        "hem_right": hem_points[1],
        "shoulder_center": [round(float(value), 3) for value in shoulder_center],
        "hem_center": [round(float(value), 3) for value in hem_center],
        "shoulder_line_angle_deg": round(shoulder_angle, 3),
        "hem_line_angle_deg": round(hem_angle, 3),
        "longitudinal_axis_angle_deg": round(longitudinal_angle, 3),
        "vertical_deviation_deg": round(vertical_deviation, 3),
        "shoulder_horizontal_deviation_deg": round(shoulder_deviation, 3),
        "hem_horizontal_deviation_deg": round(hem_deviation, 3),
        "orientation_ok": orientation_ok,
    }


def candidate_pixels_near_anchor(
    anchor: tuple[int, int],
    mask: np.ndarray,
    distance: np.ndarray,
    radius: int,
    target_clearance: float,
    limit: int = 100,
) -> list[dict[str, float]]:
    raw_u, raw_v = anchor
    x1, x2 = max(0, raw_u - radius), min(mask.shape[1], raw_u + radius + 1)
    y1, y2 = max(0, raw_v - radius), min(mask.shape[0], raw_v + radius + 1)
    patch = mask[y1:y2, x1:x2] > 0
    ys, xs = np.where(patch)
    if len(xs) == 0:
        return []
    us = xs + x1
    vs = ys + y1
    semantic_distance = np.hypot(us - raw_u, vs - raw_v)
    inside_radius = semantic_distance <= radius
    us, vs, semantic_distance = us[inside_radius], vs[inside_radius], semantic_distance[inside_radius]
    if len(us) == 0:
        return []
    clearance = distance[vs, us]
    # Prefer a controlled inset rather than the deepest point in the torso.
    clearance_error = np.abs(clearance - target_clearance) / max(1.0, target_clearance)
    proximity_error = semantic_distance / max(1.0, float(radius))
    single_score = -0.70 * clearance_error - 0.30 * proximity_error
    order = np.argsort(single_score)[::-1]

    selected: list[dict[str, float]] = []
    # Light non-maximum suppression keeps spatially diverse candidates.
    min_separation = max(2.0, radius * 0.04)
    for index in order:
        u, v = int(us[index]), int(vs[index])
        if any(np.hypot(u - item["u"], v - item["v"]) < min_separation for item in selected):
            continue
        selected.append(
            {
                "u": u,
                "v": v,
                "clearance": float(clearance[index]),
                "semantic_distance": float(semantic_distance[index]),
                "single_score": float(single_score[index]),
            }
        )
        if len(selected) >= limit:
            break
    return selected


def refine_keypoint_slightly_inward(
    point: tuple[int, int],
    mask: np.ndarray,
    distance: np.ndarray,
    inset_ratio: float,
) -> tuple[tuple[int, int], dict[str, Any]]:
    """Move a confident semantic keypoint the minimum distance to a safe inset."""
    raw_u, raw_v = point
    diagonal = float(np.hypot(mask.shape[1], mask.shape[0]))
    target_clearance = max(6.0, diagonal * max(0.0, inset_ratio))
    search_radius = max(12, int(round(target_clearance * 3.0)))

    inside = (
        0 <= raw_u < mask.shape[1]
        and 0 <= raw_v < mask.shape[0]
        and mask[raw_v, raw_u] > 0
    )
    raw_clearance = float(distance[raw_v, raw_u]) if inside else 0.0
    if inside and raw_clearance >= target_clearance:
        return point, {
            "applied": False,
            "reason": "raw_keypoint_already_has_safe_clearance",
            "target_clearance_pixels": round(target_clearance, 3),
            "raw_clearance_pixels": round(raw_clearance, 3),
            "movement_pixels": 0.0,
        }

    x1, x2 = max(0, raw_u - search_radius), min(mask.shape[1], raw_u + search_radius + 1)
    y1, y2 = max(0, raw_v - search_radius), min(mask.shape[0], raw_v + search_radius + 1)
    ys, xs = np.where(mask[y1:y2, x1:x2] > 0)
    if len(xs) == 0:
        return point, {
            "applied": False,
            "reason": "no_mask_pixel_near_keypoint_keep_raw",
            "target_clearance_pixels": round(target_clearance, 3),
            "raw_clearance_pixels": round(raw_clearance, 3),
            "movement_pixels": 0.0,
        }

    us, vs = xs + x1, ys + y1
    clearance = distance[vs, us]
    movement = np.hypot(us - raw_u, vs - raw_v)
    safe = clearance >= target_clearance
    if np.any(safe):
        safe_indices = np.where(safe)[0]
        index = int(safe_indices[np.argmin(movement[safe_indices])])
        reason = "moved_to_nearest_safe_interior_pixel"
    else:
        # A very thin or imperfect mask may not contain the requested inset.
        # Move toward its deepest nearby point without rejecting confidence.
        score = clearance - 0.15 * movement
        index = int(np.argmax(score))
        reason = "target_clearance_unavailable_used_deepest_nearby_pixel"

    refined = (int(us[index]), int(vs[index]))
    return refined, {
        "applied": refined != point,
        "reason": reason,
        "target_clearance_pixels": round(target_clearance, 3),
        "raw_clearance_pixels": round(raw_clearance, 3),
        "refined_clearance_pixels": round(float(clearance[index]), 3),
        "movement_pixels": round(float(movement[index]), 3),
        "search_radius_pixels": search_radius,
    }


def line_mask_fraction(mask: np.ndarray, left: tuple[int, int], right: tuple[int, int]) -> float:
    count = max(20, int(round(np.hypot(right[0] - left[0], right[1] - left[1]) / 4.0)))
    us = np.rint(np.linspace(left[0], right[0], count)).astype(np.int32)
    vs = np.rint(np.linspace(left[1], right[1], count)).astype(np.int32)
    valid = (us >= 0) & (us < mask.shape[1]) & (vs >= 0) & (vs < mask.shape[0])
    if not np.any(valid):
        return 0.0
    return float(np.mean(mask[vs[valid], us[valid]] > 0))


def select_joint_longest_pair(
    anchors: list[dict[str, Any]],
    mask: np.ndarray,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    anchors = sorted(anchors, key=lambda item: item["u"])
    if len(anchors) != 2:
        return [], {"valid": False, "reason": "semantic_pair_missing"}
    diagonal = float(np.hypot(mask.shape[1], mask.shape[0]))
    radius = max(12, int(round(diagonal * args.grasp_search_radius)))
    # Temporary pixel inset for the MVP. Calibration should later convert a
    # gripper-specific 5-15 mm inset into pixels/world coordinates.
    target_clearance = max(6.0, diagonal * 0.012)
    distance = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    left_candidates = candidate_pixels_near_anchor(
        (anchors[0]["u"], anchors[0]["v"]), mask, distance, radius, target_clearance
    )
    right_candidates = candidate_pixels_near_anchor(
        (anchors[1]["u"], anchors[1]["v"]), mask, distance, radius, target_clearance
    )
    best = None
    best_span = -np.inf
    best_quality = -np.inf
    # Spans within this small tolerance are effectively the same physical
    # width; among them choose the safer and more stable inset pair.
    span_tolerance = max(3.0, mask.shape[1] * 0.005)
    for left in left_candidates:
        for right in right_candidates:
            if right["u"] <= left["u"]:
                continue
            du = right["u"] - left["u"]
            dv = right["v"] - left["v"]
            span = float(np.hypot(du, dv))
            span_ratio = span / max(1.0, float(mask.shape[1]))
            if not (args.min_pair_distance <= span_ratio <= args.max_pair_distance):
                continue
            angle = float(np.degrees(np.arctan2(dv, du)))
            if normalize_horizontal_angle(angle) > args.max_pair_angle:
                continue
            occupancy = line_mask_fraction(
                mask, (int(left["u"]), int(left["v"])), (int(right["u"]), int(right["v"]))
            )
            if occupancy < 0.65:
                continue
            # Longest valid span is the primary objective. Inset quality,
            # semantic proximity and mask continuity only break near-ties.
            quality = (
                left["single_score"]
                + right["single_score"]
                + 0.8 * occupancy
                - 0.5 * normalize_horizontal_angle(angle) / max(1.0, args.max_pair_angle)
            )
            longer = span > best_span + span_tolerance
            near_tie_but_safer = abs(span - best_span) <= span_tolerance and quality > best_quality
            if longer or near_tie_but_safer:
                best_span = span
                best_quality = quality
                best = (left, right, angle, span, span_ratio, occupancy)
    if best is None:
        return [], {
            "valid": False,
            "reason": "no_joint_pair_satisfied_angle_span_and_mask_continuity",
            "left_candidate_count": len(left_candidates),
            "right_candidate_count": len(right_candidates),
            "search_radius_pixels": radius,
            "target_clearance_pixels": round(target_clearance, 3),
        }
    left, right, angle, span, span_ratio, occupancy = best
    targets = []
    for arm, anchor, point in zip(("left", "right"), anchors, (left, right)):
        targets.append(
            {
                "name": anchor["name"],
                "arm": arm,
                "raw_u": anchor["u"],
                "raw_v": anchor["v"],
                "score": anchor["score"],
                "threshold_passed": anchor["usable"],
                "near_garment_mask": anchor["near_garment_mask"],
                "grasp_u": int(point["u"]),
                "grasp_v": int(point["v"]),
                "grasp_found": True,
                "mask_clearance_pixels": round(point["clearance"], 3),
                "distance_from_keypoint_pixels": round(point["semantic_distance"], 3),
            }
        )
    geometry = {
        "valid": True,
        "reason": None,
        "angle_deg": round(float(angle), 3),
        "horizontal_angle_ok": True,
        "distance_pixels": round(float(span), 3),
        "distance_ratio_of_width": round(float(span_ratio), 6),
        "distance_ok": True,
        "line_mask_fraction": round(float(occupancy), 6),
        "selection": "longest_valid_span_then_safe_inset_quality",
        "inset_quality_score": round(float(best_quality), 6),
        "search_radius_pixels": radius,
        "target_clearance_pixels": round(target_clearance, 3),
        "left_candidate_count": len(left_candidates),
        "right_candidate_count": len(right_candidates),
    }
    return targets, geometry


def segmentation_axis_fallback_pair(
    mask: np.ndarray,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Choose a safe horizontal unfold pair near the robot-side mask end."""
    ys, xs = np.where(mask > 0)
    if len(xs) < 20:
        return [], {"valid": False, "reason": "mask_too_small"}, {}

    points = np.column_stack((xs, ys)).astype(np.float32)
    center = points.mean(axis=0)
    centered = points - center
    covariance = centered.T @ centered / max(1, len(points) - 1)
    values, vectors = np.linalg.eigh(covariance)
    axis = vectors[:, int(np.argmax(values))]
    # Orient the longitudinal axis from image top (robot-near) to bottom.
    if axis[1] < 0:
        axis = -axis
    projections = centered @ axis
    axis_angle = float(np.degrees(np.arctan2(axis[1], axis[0])))
    vertical_deviation = abs(90.0 - abs(axis_angle))
    cutoff = float(np.quantile(projections, 0.38))
    near_pixels = projections <= cutoff

    axis_info = {
        "source": "sam3_mask_pca",
        "center": [round(float(center[0]), 3), round(float(center[1]), 3)],
        "longitudinal_unit": [round(float(axis[0]), 6), round(float(axis[1]), 6)],
        "longitudinal_axis_angle_deg": round(axis_angle, 3),
        "vertical_deviation_deg": round(vertical_deviation, 3),
        "orientation_ok": bool(vertical_deviation <= args.max_pair_angle),
        "orientation_check_enforced": False,
        "robot_near_projection_cutoff": round(cutoff, 3),
    }

    near_mask = np.zeros_like(mask, dtype=np.uint8)
    near_mask[ys[near_pixels], xs[near_pixels]] = 1
    target_inset = max(6, int(round(np.hypot(mask.shape[1], mask.shape[0]) * 0.012)))
    best: tuple[int, int, int] | None = None
    for v in np.where(np.any(near_mask > 0, axis=1))[0]:
        row_xs = np.where(near_mask[v] > 0)[0]
        if len(row_xs) < 2:
            continue
        # Keep separate connected mask runs instead of crossing background.
        split_indices = np.where(np.diff(row_xs) > 1)[0] + 1
        for run in np.split(row_xs, split_indices):
            if len(run) < 2:
                continue
            left = int(run[0] + target_inset)
            right = int(run[-1] - target_inset)
            if right <= left:
                continue
            span_ratio = (right - left) / max(1.0, float(mask.shape[1]))
            if not (args.min_pair_distance <= span_ratio <= args.max_pair_distance):
                continue
            if best is None or right - left > best[2] - best[1]:
                best = (int(v), left, right)

    if best is None:
        return [], {
            "valid": False,
            "reason": "no_horizontal_span_in_robot_near_mask_end",
            "target_clearance_pixels": target_inset,
        }, axis_info

    v, left_u, right_u = best
    span = float(right_u - left_u)
    occupancy = line_mask_fraction(mask, (left_u, v), (right_u, v))
    targets = [
        {
            "name": "segmentation_unfold_left",
            "arm": "left",
            "u": left_u,
            "v": v,
            "mask_clearance_pixels": target_inset,
            "grasp_found": True,
        },
        {
            "name": "segmentation_unfold_right",
            "arm": "right",
            "u": right_u,
            "v": v,
            "mask_clearance_pixels": target_inset,
            "grasp_found": True,
        },
    ]
    geometry = {
        "valid": True,
        "reason": None,
        "selection": "sam3_principal_axis_robot_near_longest_horizontal_span",
        "angle_deg": 0.0,
        "horizontal_angle_ok": True,
        "distance_pixels": round(span, 3),
        "distance_ratio_of_width": round(span / mask.shape[1], 6),
        "distance_ok": True,
        "line_mask_fraction": round(occupancy, 6),
        "target_clearance_pixels": target_inset,
    }
    return targets, geometry, axis_info


def analyze_frame(
    frame: np.ndarray,
    phase: str,
    attempt: int,
    args: argparse.Namespace,
    segmenter: FalSam3Segmenter,
    fashion_net,
) -> dict[str, Any]:
    started = time.perf_counter()
    mask, segmentation_ms, segmentation_details = segmenter.segment(frame)
    rois = roi_rectangles(
        frame.shape[1],
        frame.shape[0],
        args.roi_top,
        args.roi_bottom,
        args.roi_margin_x,
        args.roi_center_overlap,
    )
    orientation = mask_orientation(mask)
    fashion_ms = None
    targets: list[dict[str, Any]] = []
    geometry: dict[str, Any] = {"valid": False, "reason": "not_analyzed"}
    axes: dict[str, Any] | None = None
    semantic_keypoints: list[dict[str, Any]] = []
    reachability: dict[str, Any] = {"checked": False, "valid": False}
    semantic_gate: dict[str, Any] = {"passed": False, "reason": "not_checked"}
    decision_source = "none"
    model_input_rotated_180 = False

    if phase in (PHASE_SHOULDER, PHASE_HEM):
        x1, y1, x2, y2 = mask_crop_box(mask, args.padding)
        garment_only = np.full_like(frame, 255)
        garment_only[mask > 0] = frame[mask > 0]
        crop = garment_only[y1:y2, x1:x2]
        model_crop = crop
        if phase == PHASE_HEM:
            # The physical garment is now upside down in the camera.  Present
            # an upright crop to FashionAI, then invert its pixel coordinates.
            model_crop = cv2.rotate(crop, cv2.ROTATE_180)
            model_input_rotated_180 = True
        fashion_started = time.perf_counter()
        keypoints, _, scores = predict_keypoints_with_heatmaps(
            args.clothing_type, model_crop, fashion_net
        )
        fashion_ms = (time.perf_counter() - fashion_started) * 1000.0
        if model_input_rotated_180:
            keypoints[:, 0] = (crop.shape[1] - 1) - keypoints[:, 0]
            keypoints[:, 1] = (crop.shape[0] - 1) - keypoints[:, 1]
        keypoints[:, 0] += x1
        keypoints[:, 1] += y1

        names = KEYPOINT_NAMES[args.clothing_type]
        tolerance = max(5, int(round(max(crop.shape[:2]) * 0.03)))
        target_group = "shoulder" if phase == PHASE_SHOULDER else "hem"
        target_names = TARGETS[args.clothing_type][target_group]
        semantic_keypoints = [
            semantic_record(
                name, names, keypoints, scores, mask,
                args.keypoint_threshold, tolerance,
            )
            for name in target_names
        ]

        # SAM3 supplies the replacement for the former FlingBot fallback.
        # The FashionAI path is intentionally confidence-only: once both
        # current-phase keypoints pass, use them without angle/ROI gating.
        fallback_targets, fallback_geometry, axes = segmentation_axis_fallback_pair(mask, args)
        ordered_semantic = sorted(semantic_keypoints, key=lambda item: item["u"])
        confidence_ok = all(item["confidence_passed"] for item in semantic_keypoints)
        semantic_gate = {
            "passed": bool(confidence_ok),
            "reason": None if confidence_ok else "confidence_below_threshold",
            "target_group": target_group,
            "confidence_ok": bool(confidence_ok),
            "angle_check_applied": False,
            "roi_check_applied": False,
            "mask_check_applied": "coordinate_refinement_only_not_a_gate",
        }

        if semantic_gate["passed"]:
            targets = []
            mask_distance = cv2.distanceTransform(
                (mask > 0).astype(np.uint8), cv2.DIST_L2, 5
            )
            for arm, item in zip(("left", "right"), ordered_semantic):
                refined, correction = refine_keypoint_slightly_inward(
                    (item["u"], item["v"]),
                    mask,
                    mask_distance,
                    args.keypoint_inset_ratio,
                )
                targets.append(
                    {
                        "name": item["name"],
                        "arm": arm,
                        "raw_u": item["u"],
                        "raw_v": item["v"],
                        "score": item["score"],
                        "threshold_passed": True,
                        "near_garment_mask": item["near_garment_mask"],
                        "grasp_u": refined[0],
                        "grasp_v": refined[1],
                        "grasp_found": True,
                        "mask_clearance_pixels": correction.get(
                            "refined_clearance_pixels",
                            correction.get("raw_clearance_pixels"),
                        ),
                        "inward_correction": correction,
                    }
                )
            geometry = {
                "valid": True,
                "reason": None,
                "selection": "fashionai_confidence_pass_then_minimal_mask_inset",
                "angle_deg": None,
                "horizontal_angle_ok": None,
                "distance_ok": None,
                "checks_skipped": ["angle", "roi"],
            }
            decision_source = "fashionai_keypoints_direct"

        if not semantic_gate["passed"]:
            targets, geometry = fallback_targets, fallback_geometry
            decision_source = "sam3_axis_unfold_fallback"

        if len(targets) != 2 or not geometry.get("valid", False):
            state = "GRASP_PAIR_NOT_FOUND"
        elif decision_source == "fashionai_keypoints_direct":
            state = (
                "SHOULDER_GRASP_READY"
                if phase == PHASE_SHOULDER
                else "HEM_GRASP_READY"
            )
            reachability = {
                "checked": False,
                "valid": None,
                "note": "Skipped by design after FashionAI confidence pass.",
            }
        else:
            # Angle was already enforced by the mask fallback; ROI is its final gate.
            def target_point(item: dict[str, Any]) -> tuple[int, int]:
                if "grasp_u" in item:
                    return int(item["grasp_u"]), int(item["grasp_v"])
                return int(item["u"]), int(item["v"])

            left_point, right_point = target_point(targets[0]), target_point(targets[1])
            left_ok = point_in_rect(left_point, rois["left"])
            right_ok = point_in_rect(right_point, rois["right"])
            targets[0]["inside_arm_roi"] = left_ok
            targets[1]["inside_arm_roi"] = right_ok
            reachability = {
                "checked": True,
                "valid": bool(left_ok and right_ok),
                "left_arm_roi_ok": bool(left_ok),
                "right_arm_roi_ok": bool(right_ok),
                "note": "Image ROI only; world-coordinate reachability is not implemented.",
            }
            if not reachability["valid"]:
                state = "GRASP_PAIR_OUT_OF_REACH"
            elif decision_source == "sam3_axis_unfold_fallback":
                state = "UNFOLD_REQUIRED"
    else:
        state = "INSPECTION_COMPLETE"
        geometry = {"valid": True, "reason": None}

    public = {
        "timestamp": datetime.now().isoformat(timespec="milliseconds"),
        "phase": phase,
        "state": state,
        "attempt": attempt,
        "image_size": {"width": frame.shape[1], "height": frame.shape[0]},
        "arm_rois_xyxy": {key: list(value) for key, value in rois.items()},
        "mask_area_pixels": int(np.count_nonzero(mask)),
        "mask_principal_axis_angle_deg": round(orientation, 3) if orientation is not None else None,
        "segmentation_backend": segmentation_details["backend"],
        "segmentation_confidence": float(segmentation_details["selected_confidence"]),
        "segmentation_details": segmentation_details,
        "semantic_keypoints": semantic_keypoints,
        "semantic_gate": semantic_gate,
        "decision_source": decision_source,
        "garment_axes": axes,
        "targets": targets,
        "pair_geometry": geometry,
        "reachability": reachability,
        "fashionai_input_rotated_180": model_input_rotated_180,
        "timing_ms": {
            "segmentation": round(segmentation_ms, 3),
            "fashionai": round(fashion_ms, 3) if fashion_ms is not None else None,
            "total": round((time.perf_counter() - started) * 1000.0, 3),
        },
        "warning": "Image-space candidates only; no calibration, collision, force, or robot command is applied.",
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
    scale: float = 0.65,
) -> None:
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def state_instruction(state: str) -> str:
    instructions = {
        "SHOULDER_GRASP_READY": "Grasp shoulders, yaw garment 180 deg, place, then SPACE",
        "HEM_GRASP_READY": "Grasp/lift hem, then SPACE -> complete",
        "UNFOLD_REQUIRED": "Use SAM3 unfold points, lift/drop garment, then SPACE to re-check",
        "WRONG_SIDE_FOR_PHASE": "Put the requested garment end on the robot-near (top) side",
        "GRASP_PAIR_NOT_FOUND": "No long, horizontal and mask-safe grasp pair was found",
        "GRASP_PAIR_OUT_OF_REACH": "Valid pair exists but is outside one or both arm ROIs",
        "INSPECTION_COMPLETE": "Sequence complete; press R to restart",
    }
    return instructions.get(state, "Waiting for analysis")


def result_display_stage(
    result: dict[str, Any] | None,
    display_stage_seconds: float,
) -> str:
    if result is None:
        return "waiting"
    age = max(0.0, time.monotonic() - result["completed_monotonic"])
    if age < display_stage_seconds:
        return "sam3"
    if age < display_stage_seconds * 2.0:
        return "fashionai"
    return "decision"


def draw_pipeline_panel(
    image: np.ndarray,
    panel_width: int,
    stage: str,
    final_state: str | None,
) -> np.ndarray:
    """Attach a video-friendly pipeline panel and highlight the current stage."""
    if panel_width <= 0:
        return image

    height = image.shape[0]
    panel = np.full((height, panel_width, 3), (20, 24, 31), dtype=np.uint8)
    active_green = (90, 225, 90)
    active_magenta = (255, 105, 235)
    active_yellow = (40, 220, 255)
    inactive = (105, 112, 124)
    text = (238, 241, 245)

    put_label(panel, "GARMENT INSPECTION", (22, 42), text, 0.65)
    put_label(panel, "PIPELINE", (22, 70), (180, 187, 198), 0.55)

    nodes = [
        ("STILL", "Garment motion stopped"),
        ("SAM3", "Garment segmentation"),
        ("FASHIONAI", "Shoulder score"),
        ("DECISION", "Select robot action"),
    ]
    active_indices = {
        "waiting": {0},
        "sam3": {0, 1},
        "fashionai": {2},
        "decision": {3},
    }.get(stage, set())
    active_color = {
        "waiting": active_green,
        "sam3": active_green,
        "fashionai": active_magenta,
        "decision": active_yellow,
    }.get(stage, active_green)

    top = 112
    box_height = 76
    gap = 28
    x1, x2 = 20, panel_width - 20
    for index, (title, subtitle) in enumerate(nodes):
        y1 = top + index * (box_height + gap)
        y2 = y1 + box_height
        is_active = index in active_indices
        color = active_color if is_active else inactive
        if is_active:
            overlay = panel.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            panel = cv2.addWeighted(overlay, 0.20, panel, 0.80, 0)
        cv2.rectangle(panel, (x1, y1), (x2, y2), color, 2)
        cv2.circle(panel, (42, (y1 + y2) // 2), 14, color, -1 if is_active else 2)
        put_label(panel, str(index + 1), (37, (y1 + y2) // 2 + 5), (15, 20, 25) if is_active else color, 0.44)
        put_label(panel, title, (68, y1 + 30), text if is_active else color, 0.56)
        put_label(panel, subtitle, (68, y1 + 55), (205, 210, 218) if is_active else inactive, 0.42)
        if index < len(nodes) - 1:
            arrow_x = panel_width // 2
            cv2.line(panel, (arrow_x, y2 + 3), (arrow_x, y2 + gap - 7), inactive, 2)
            cv2.line(panel, (arrow_x, y2 + gap - 7), (arrow_x - 5, y2 + gap - 13), inactive, 2)
            cv2.line(panel, (arrow_x, y2 + gap - 7), (arrow_x + 5, y2 + gap - 13), inactive, 2)

    branch_y = top + 4 * (box_height + gap) + 2
    pass_active = stage == "decision" and final_state in {
        "SHOULDER_GRASP_READY", "HEM_GRASP_READY", "INSPECTION_COMPLETE"
    }
    fail_active = stage == "decision" and not pass_active
    branch_gap = 8
    branch_width = (panel_width - 40 - branch_gap) // 2
    branches = [
        (x1, "PASS", "GRASP", pass_active, active_green),
        (x1 + branch_width + branch_gap, "LOW", "UNFOLD", fail_active, active_yellow),
    ]
    for bx, title, action, is_active, color in branches:
        by2 = min(height - 20, branch_y + 74)
        draw_color = color if is_active else inactive
        if is_active:
            overlay = panel.copy()
            cv2.rectangle(overlay, (bx, branch_y), (bx + branch_width, by2), draw_color, -1)
            panel = cv2.addWeighted(overlay, 0.20, panel, 0.80, 0)
        cv2.rectangle(panel, (bx, branch_y), (bx + branch_width, by2), draw_color, 2)
        put_label(panel, title, (bx + 10, branch_y + 28), text if is_active else draw_color, 0.48)
        put_label(panel, action, (bx + 10, branch_y + 54), text if is_active else draw_color, 0.44)

    return np.concatenate((panel, image), axis=1)


def render_overlay(
    frame: np.ndarray,
    result: dict[str, Any] | None,
    busy: bool,
    auto_enabled: bool,
    scene_status: str,
    detected_change_ratio: float,
    phase: str,
    attempt: int,
    max_attempts: int,
    rois: dict[str, tuple[int, int, int, int]],
    display_stage_seconds: float,
    pipeline_panel_width: int,
) -> np.ndarray:
    output = frame.copy()
    height, width = output.shape[:2]

    roi_colors = {"left": (255, 210, 0), "right": (0, 210, 255)}
    for arm, rect in rois.items():
        x1, y1, x2, y2 = rect
        cv2.rectangle(output, (x1, y1), (x2 - 1, y2 - 1), roi_colors[arm], 2)
        put_label(output, f"{arm.upper()} ARM ROI", (x1 + 8, y1 + 25), roi_colors[arm], 0.55)

    # Keep the status background compact so it does not hide the garment.
    panel_height = min(135, max(125, height // 5))
    dark = np.zeros_like(output[:panel_height])
    output[:panel_height] = cv2.addWeighted(output[:panel_height], 0.30, dark, 0.70, 0)
    state = "WAITING_FOR_FIRST_INFERENCE"
    state_color = (0, 220, 255)

    if result is not None:
        public = result["public"]
        final_state = public["state"]
        age = max(0.0, time.monotonic() - result["completed_monotonic"])
        show_fashionai = age >= display_stage_seconds
        show_decision = age >= display_stage_seconds * 2.0

        if not show_fashionai:
            state = "SAM3_SEGMENTATION"
            state_color = (0, 230, 0)
        elif not show_decision:
            state = "FASHIONAI_KEYPOINTS"
            state_color = (255, 100, 255)
        else:
            state = final_state

        if show_decision and state in ("SHOULDER_GRASP_READY", "HEM_GRASP_READY", "INSPECTION_COMPLETE"):
            state_color = (0, 230, 0)
        elif show_decision and state in ("UNFOLD_REQUIRED", "WRONG_SIDE_FOR_PHASE", "GRASP_PAIR_OUT_OF_REACH"):
            state_color = (0, 165, 255)
        elif show_decision:
            state_color = (0, 0, 255)

        mask = result["mask"]
        if mask.shape == frame.shape[:2]:
            green = np.zeros_like(output)
            green[mask > 0] = (30, 190, 30)
            output = cv2.addWeighted(output, 0.82, green, 0.18, 0)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(output, contours, -1, (0, 255, 0), 2)

        axes = public.get("garment_axes") if show_decision else None
        if axes is not None and "shoulder_left" in axes:
            shoulder_left = tuple(map(int, axes["shoulder_left"]))
            shoulder_right = tuple(map(int, axes["shoulder_right"]))
            hem_left = tuple(map(int, axes["hem_left"]))
            hem_right = tuple(map(int, axes["hem_right"]))
            shoulder_center = tuple(int(round(value)) for value in axes["shoulder_center"])
            hem_center = tuple(int(round(value)) for value in axes["hem_center"])
            cv2.line(output, shoulder_left, shoulder_right, (255, 80, 255), 3, cv2.LINE_AA)
            cv2.line(output, hem_left, hem_right, (255, 180, 40), 3, cv2.LINE_AA)
            cv2.line(output, shoulder_center, hem_center, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.circle(output, shoulder_center, 7, (255, 80, 255), -1)
            cv2.circle(output, hem_center, 7, (255, 180, 40), -1)
        elif axes is not None and "longitudinal_unit" in axes:
            center = np.asarray(axes["center"], dtype=np.float32)
            direction = np.asarray(axes["longitudinal_unit"], dtype=np.float32)
            half_length = min(height, width) * 0.24
            start = tuple(map(int, np.rint(center - direction * half_length)))
            end = tuple(map(int, np.rint(center + direction * half_length)))
            cv2.line(output, start, end, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.circle(output, tuple(map(int, np.rint(center))), 7, (255, 255, 255), -1)

        # FashionAI observations appear only after the SAM3-only stage.
        if show_fashionai:
            for item in public.get("semantic_keypoints", []):
                point = (item["u"], item["v"])
                color = (180, 255, 180) if item["confidence_passed"] else (80, 80, 255)
                cv2.drawMarker(output, point, color, cv2.MARKER_CROSS, 16, 2)
                if not show_decision or public.get("decision_source") != "fashionai_keypoints_direct":
                    put_label(
                        output,
                        f"{item['name']} score={item['score']:.2f}",
                        (point[0] + 8, max(panel_height + 20, point[1] - 8)),
                        color,
                        0.46,
                    )

        grasp_points = []
        for item in public["targets"] if show_decision else []:
            if "raw_u" in item:
                raw = (item["raw_u"], item["raw_v"])
                raw_color = (255, 100, 255) if item["threshold_passed"] else (100, 100, 100)
                cv2.drawMarker(output, raw, raw_color, cv2.MARKER_CROSS, 22, 2)
                put_label(
                    output,
                    f"{item['name']} {item['score']:.1f}",
                    (raw[0] + 10, max(panel_height + 22, raw[1] - 8)),
                    raw_color,
                    0.50,
                )
                if item["grasp_found"]:
                    point = (item["grasp_u"], item["grasp_v"])
                    grasp_points.append(point)
                    color = roi_colors[item["arm"]]
                    if point != raw:
                        cv2.line(output, raw, point, color, 2, cv2.LINE_AA)
                    cv2.circle(output, point, 13, color, -1)
                    movement = item.get("inward_correction", {}).get("movement_pixels")
                    suffix = "" if movement is None else f" inset={movement:.1f}px"
                    put_label(
                        output,
                        f"{item['arm'].upper()} GRASP{suffix}",
                        (point[0] + 12, point[1] - 10),
                        color,
                        0.55,
                    )
            else:
                point = (item["u"], item["v"])
                grasp_points.append(point)
                color = roi_colors[item["arm"]]
                cv2.circle(output, point, 13, color, -1)
                put_label(output, f"UNFOLD {item['arm'].upper()}", (point[0] + 12, point[1] - 10), color, 0.55)
        if len(grasp_points) == 2:
            grasp_points.sort(key=lambda point: point[0])
            cv2.line(output, grasp_points[0], grasp_points[1], (0, 255, 255), 4, cv2.LINE_AA)

        if show_decision:
            geometry = public["pair_geometry"]
            angle = geometry.get("angle_deg")
            angle_text = "n/a" if angle is None else f"{angle:.1f} deg"
            stage_instruction = state_instruction(state)
            put_label(
                output,
                f"STEP 3/3  Decision | pair angle={angle_text}",
                (20, 72),
                (240, 240, 240),
                0.60,
            )
        elif show_fashionai:
            stage_instruction = "FashionAI: shoulder keypoints and confidence scores"
            put_label(output, "STEP 2/3  FashionAI keypoint inference", (20, 72), state_color, 0.60)
        else:
            stage_instruction = "SAM3: garment region segmentation"
            put_label(output, "STEP 1/3  SAM3 garment mask", (20, 72), state_color, 0.60)
        put_label(
            output,
            f"result age={age:.1f}s  inference={public['timing_ms']['total']:.0f}ms",
            (20, 96),
            (240, 240, 240),
            0.52,
        )
        put_label(output, stage_instruction, (20, panel_height - 10), state_color, 0.50)

    put_label(output, f"PHASE: {phase} | STATE: {state}", (20, 42), state_color, 0.82)
    status = (
        f"attempt {attempt}/{max_attempts} | AI {'BUSY' if busy else 'IDLE'} | "
        f"motion {'ON' if auto_enabled else 'OFF'} | scene {scene_status} "
        f"({detected_change_ratio * 100.0:.1f}%)"
    )
    text_width = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)[0][0]
    put_label(output, status, (max(20, width - text_width - 20), 40), (255, 255, 255), 0.58)
    put_label(output, "SPACE next/re-check | A motion trigger | 1 shoulder | 2 hem | S save | R reset | Q quit", (20, height - 20), (255, 255, 255), 0.54)
    stage = result_display_stage(result, display_stage_seconds)
    final_state = None if result is None else result["public"]["state"]
    return draw_pipeline_panel(output, pipeline_panel_width, stage, final_state)


def next_phase(current_phase: str, state: str) -> str:
    if current_phase == PHASE_SHOULDER and state == "SHOULDER_GRASP_READY":
        return PHASE_HEM
    if current_phase == PHASE_HEM and state == "HEM_GRASP_READY":
        return PHASE_COMPLETE
    return current_phase


def main() -> None:
    args = parse_args()
    args.inference_interval = max(1.0, args.inference_interval)
    args.max_attempts = max(1, args.max_attempts)
    args.record_fps = max(1.0, args.record_fps)
    args.display_stage_seconds = max(0.0, args.display_stage_seconds)
    args.pipeline_panel_width = max(0, args.pipeline_panel_width)
    args.scene_change_ratio = float(np.clip(args.scene_change_ratio, 0.001, 1.0))
    args.scene_change_pixel_threshold = max(1.0, args.scene_change_pixel_threshold)
    args.scene_motion_ratio = float(np.clip(args.scene_motion_ratio, 0.0001, 1.0))
    args.scene_settle_seconds = max(0.2, args.scene_settle_seconds)
    args.scene_sample_width = max(80, args.scene_sample_width)

    if args.inference_interval < 5.0:
        print(
            "[WARN] SAM3 is a paid cloud API; a minimum call gap below 5 seconds "
            "can increase cost when the scene changes frequently."
        )
    print(f"[INFO] configuring fal SAM3 prompt: {args.sam3_prompt!r}")
    segmenter = FalSam3Segmenter(args.sam3_prompt, args.sam3_input_size)
    print(f"[INFO] loading FashionAI '{args.clothing_type}'")
    fashion_net = load_net(args.clothing_type, env_id=None)

    import rclpy
    from cv_bridge import CvBridge
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import Image

    rclpy.init()
    node = rclpy.create_node("garment_inspection_roi_mvp")
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
    subscription = node.create_subscription(Image, args.ros_topic, image_callback, qos)
    _ = subscription

    run_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    task_queue: queue.Queue = queue.Queue(maxsize=1)
    result_lock = threading.Lock()
    latest_result: dict[str, Any] = {"value": None}
    active_phase: dict[str, str] = {"value": PHASE_SHOULDER}
    active_scene_generation: dict[str, int] = {"value": 0}
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
            frame, task_phase, attempt, sequence, scene_generation = task
            busy.set()
            try:
                result = analyze_frame(
                    frame, task_phase, attempt, args, segmenter, fashion_net
                )
                result["public"]["frame_sequence"] = sequence
                with result_lock:
                    # A phase key may be pressed while inference is still
                    # running. Never publish coordinates from the old phase.
                    if active_phase["value"] != task_phase:
                        print(f"[INFO] discarded stale result for phase={task_phase}")
                        continue
                    if active_scene_generation["value"] != scene_generation:
                        print("[INFO] discarded stale result because the garment moved")
                        continue
                    latest_result["value"] = result
                    events.append(json_safe(result["public"]))
                cv2.imwrite(str(run_dir / "latest_input.png"), frame)
                cv2.imwrite(str(run_dir / "latest_mask.png"), result["mask"])
                with open(run_dir / "latest_result.json", "w", encoding="utf-8") as stream:
                    json.dump(json_safe(result["public"]), stream, indent=2, ensure_ascii=False)
                print(
                    f"[RESULT] phase={task_phase} state={result['public']['state']} "
                    f"attempt={attempt} total={result['public']['timing_ms']['total']:.0f}ms"
                )
            except Exception as error:
                print(f"[ERROR] inference failed: {error}")
                with result_lock:
                    events.append(
                        {
                            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
                            "phase": task_phase,
                            "state": "INFERENCE_ERROR",
                            "attempt": attempt,
                            "error": str(error),
                        }
                    )
            finally:
                busy.clear()
                task_queue.task_done()

    thread = threading.Thread(target=worker, name="garment-roi-inference", daemon=True)
    thread.start()

    print(f"[INFO] waiting for ROS image: {args.ros_topic}")
    print("[INFO] models run once initially, then only after a settled scene change")
    print("[KEYS] SPACE next/re-check | A motion trigger | 1 shoulder | 2 hem | R reset | Q quit")
    phase = PHASE_SHOULDER
    attempt = 1
    auto_enabled = True
    last_submit = -1e9
    # Even the initial inference waits for the camera and garment to settle.
    # SPACE/phase keys remain explicit manual overrides.
    force_submit = False
    baseline_signature: np.ndarray | None = None
    previous_signature: np.ndarray | None = None
    pending_scene_change = True
    last_motion_time = time.monotonic()
    detected_change_ratio = 1.0
    scene_status = "INITIAL"
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
            signature = scene_signature(frame, args.scene_sample_width)
            adjacent_change_ratio = scene_change_ratio(
                previous_signature,
                signature,
                args.scene_change_pixel_threshold,
            )
            previous_signature = signature
            if adjacent_change_ratio >= args.scene_motion_ratio:
                last_motion_time = now

            detected_change_ratio = scene_change_ratio(
                baseline_signature,
                signature,
                args.scene_change_pixel_threshold,
            )
            changed = (
                baseline_signature is None
                or detected_change_ratio >= args.scene_change_ratio
            )
            if changed and baseline_signature is not None and not pending_scene_change:
                pending_scene_change = True
                active_scene_generation["value"] += 1
                with result_lock:
                    latest_result["value"] = None
                print(
                    "[MOTION] garment/scene change detected "
                    f"({detected_change_ratio * 100.0:.1f}%); waiting for it to settle"
                )

            settled = now - last_motion_time >= args.scene_settle_seconds
            if busy.is_set():
                scene_status = "ANALYZING"
            elif pending_scene_change and not settled:
                scene_status = "MOVING"
            elif pending_scene_change and settled:
                scene_status = "CHANGED"
            else:
                scene_status = "STABLE"

            motion_due = (
                auto_enabled
                and pending_scene_change
                and settled
                and now - last_submit >= args.inference_interval
            )
            if (
                (force_submit or motion_due)
                and phase != PHASE_COMPLETE
                and not busy.is_set()
                and task_queue.empty()
            ):
                scene_generation = active_scene_generation["value"]
                task_queue.put(
                    (frame.copy(), phase, attempt, sequence, scene_generation)
                )
                last_submit = now
                force_submit = False
                baseline_signature = signature.copy()
                pending_scene_change = False
                scene_status = "ANALYZING"
                print(
                    f"[INFERENCE] submitted phase={phase} frame={sequence} "
                    f"scene_generation={scene_generation}"
                )

            with result_lock:
                result = latest_result["value"]
            rois = roi_rectangles(
                frame.shape[1], frame.shape[0], args.roi_top, args.roi_bottom,
                args.roi_margin_x, args.roi_center_overlap
            )
            if phase == PHASE_COMPLETE and result is not None:
                result = dict(result)
                result["public"] = dict(result["public"])
                result["public"]["state"] = "INSPECTION_COMPLETE"
            overlay = render_overlay(
                frame, result, busy.is_set(), auto_enabled,
                scene_status, detected_change_ratio, phase,
                attempt, args.max_attempts, rois, args.display_stage_seconds,
                args.pipeline_panel_width,
            )

            if writer is None and not args.no_record:
                writer = cv2.VideoWriter(
                    str(run_dir / "demo.mp4"),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    args.record_fps,
                    (overlay.shape[1], overlay.shape[0]),
                )
                if not writer.isOpened():
                    print("[WARN] video writer could not be opened; recording disabled")
                    writer = None
                    args.no_record = True
            if writer is not None and now - last_video_write >= 1.0 / args.record_fps:
                writer.write(overlay)
                last_video_write = now

            preview = overlay
            if args.display_width > 0 and preview.shape[1] > args.display_width:
                scale = args.display_width / preview.shape[1]
                preview = cv2.resize(preview, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            cv2.imshow("DUARO ROI garment inspection MVP", preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("a"):
                auto_enabled = not auto_enabled
                print(
                    f"[INFO] motion-triggered inference "
                    f"{'enabled' if auto_enabled else 'disabled'}"
                )
            elif key == ord(" "):
                if (
                    result is not None
                    and time.monotonic() - result["completed_monotonic"]
                    < args.display_stage_seconds * 2.0
                ):
                    print("[INFO] wait until the staged result display reaches STEP 3/3")
                    continue
                current_state = result["public"]["state"] if result is not None else ""
                new_phase = next_phase(phase, current_state)
                if new_phase == phase:
                    attempt = min(args.max_attempts, attempt + 1)
                else:
                    phase = new_phase
                    attempt = 1
                with result_lock:
                    active_phase["value"] = phase
                    latest_result["value"] = None
                force_submit = phase != PHASE_COMPLETE
                print(f"[INFO] phase={phase} attempt={attempt}")
            elif key in (ord("1"), ord("2")):
                phase = {ord("1"): PHASE_SHOULDER, ord("2"): PHASE_HEM}[key]
                attempt = 1
                with result_lock:
                    active_phase["value"] = phase
                    latest_result["value"] = None
                force_submit = True
                print(f"[INFO] manually selected phase={phase}")
            elif key == ord("r"):
                phase = PHASE_SHOULDER
                attempt = 1
                with result_lock:
                    active_phase["value"] = phase
                    latest_result["value"] = None
                force_submit = True
                print("[INFO] sequence reset to SHOULDER")
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
                    "segmentation": {
                        "backend": "fal-ai/sam-3/image",
                        "prompt": args.sam3_prompt,
                        "letterbox_input_size": args.sam3_input_size,
                        "minimum_inference_gap_seconds": args.inference_interval,
                        "motion_trigger": {
                            "scene_change_ratio": args.scene_change_ratio,
                            "pixel_threshold": args.scene_change_pixel_threshold,
                            "adjacent_frame_motion_ratio": args.scene_motion_ratio,
                            "settle_seconds": args.scene_settle_seconds,
                            "sample_width": args.scene_sample_width,
                        },
                    },
                    "display_stage_seconds": args.display_stage_seconds,
                    "pipeline_panel_width": args.pipeline_panel_width,
                    "roi": {
                        "top": args.roi_top,
                        "bottom": args.roi_bottom,
                        "margin_x": args.roi_margin_x,
                        "center_overlap": args.roi_center_overlap,
                    },
                    "max_pair_angle_deg": args.max_pair_angle,
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
