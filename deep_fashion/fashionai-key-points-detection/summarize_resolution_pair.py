"""Summarize the newest FashionAI/FlingBot results for a resolution pair."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np


def newest_result(root: Path) -> tuple[Path, dict]:
    candidates = list(root.glob("*/result.json"))
    if not candidates:
        raise FileNotFoundError(f"No timestamped result.json under {root}")
    path = max(candidates, key=lambda item: item.stat().st_mtime)
    with open(path, encoding="utf-8") as stream:
        return path.parent, json.load(stream)


def sharpness(path: Path) -> float:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return float("nan")
    return float(cv2.Laplacian(image, cv2.CV_64F).var())


def normalized_point(point: dict, image_size: dict) -> tuple[float, float]:
    return point["u"] / image_size["width"], point["v"] / image_size["height"]


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: summarize_resolution_pair.py <comparison-root>")
    root = Path(sys.argv[1])
    summary = {"comparison_root": str(root), "resolutions": {}}

    for resolution in ("1280x720", "1920x1080"):
        fashion_dir, fashion = newest_result(root / resolution / "fashionai")
        fling_dir, fling = newest_result(root / resolution / "flingbot")
        shoulders = fashion["shoulders"]
        best = fling["best_action"]
        summary["resolutions"][resolution] = {
            "fashionai": {
                "result_dir": str(fashion_dir),
                "sam_confidence": fashion["sam2_mask_confidence"],
                "inference_ms": fashion["fashionai_ms"],
                "usable": fashion["shoulder_candidates_usable"],
                "model_input_sharpness": round(sharpness(fashion_dir / "fashionai_model_input_512.png"), 3),
                "shoulder_left_score": shoulders["shoulder_left"]["heatmap_peak_score"],
                "shoulder_right_score": shoulders["shoulder_right"]["heatmap_peak_score"],
                "shoulder_left_uv_normalized": normalized_point(shoulders["shoulder_left"], fashion["image_size"]),
                "shoulder_right_uv_normalized": normalized_point(shoulders["shoulder_right"], fashion["image_size"]),
            },
            "flingbot": {
                "result_dir": str(fling_dir),
                "sam_confidence": fling["sam2_mask_confidence"],
                "inference_ms": fling["flingbot_value_ms"],
                "normalized_256_sharpness": round(sharpness(fling_dir / "flingbot_normalized_input.png"), 3),
                "value": best["value"],
                "rotation_deg": best["rotation_deg"],
                "scale": best["scale"],
                "left_uv_normalized": normalized_point(best["left_grasp"], fling["image_size"]),
                "right_uv_normalized": normalized_point(best["right_grasp"], fling["image_size"]),
            },
        }

    output = root / "summary.json"
    with open(output, "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[DONE] summary saved: {output}")


if __name__ == "__main__":
    main()
