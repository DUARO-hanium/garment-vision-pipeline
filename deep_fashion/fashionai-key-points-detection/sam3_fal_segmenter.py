"""fal.ai SAM3 garment segmentation with reversible letterbox geometry."""

from __future__ import annotations

import base64
import os
import tempfile
import time
import urllib.request
from typing import Any

import cv2
import numpy as np


class FalSam3Segmenter:
    """Upload one frame to fal, request a garment mask, and restore source UVs."""

    ENDPOINT = "fal-ai/sam-3/image"
    FALLBACK_PROMPTS = ("clothing", "cloth", "fabric")

    def __init__(self, prompt: str, input_size: int = 1024) -> None:
        if not os.environ.get("FAL_KEY"):
            raise RuntimeError(
                "FAL_KEY is not set. Run: export FAL_KEY=... (do not paste the key into source code)"
            )
        try:
            import fal_client
        except ImportError as error:
            raise RuntimeError(
                "fal-client is missing. Activate the project venv and run: "
                "python3 -m pip install fal-client"
            ) from error

        self._fal_client = fal_client
        self.prompt = prompt
        self.input_size = max(256, int(input_size))

    def _letterbox(self, frame: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
        source_height, source_width = frame.shape[:2]
        scale = min(self.input_size / source_width, self.input_size / source_height)
        resized_width = max(1, int(round(source_width * scale)))
        resized_height = max(1, int(round(source_height * scale)))
        resized = cv2.resize(
            frame,
            (resized_width, resized_height),
            interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
        )
        pad_left = (self.input_size - resized_width) // 2
        pad_top = (self.input_size - resized_height) // 2
        canvas = np.full((self.input_size, self.input_size, 3), 127, dtype=np.uint8)
        canvas[pad_top : pad_top + resized_height, pad_left : pad_left + resized_width] = resized
        return canvas, {
            "source_width": source_width,
            "source_height": source_height,
            "resized_width": resized_width,
            "resized_height": resized_height,
            "pad_left": pad_left,
            "pad_top": pad_top,
        }

    @staticmethod
    def _download(url: str) -> bytes:
        if url.startswith("data:"):
            _, encoded = url.split(",", 1)
            return base64.b64decode(encoded)
        with urllib.request.urlopen(url, timeout=60) as response:
            return response.read()

    @staticmethod
    def _binary_from_image(image: np.ndarray) -> np.ndarray:
        if image is None:
            raise RuntimeError("SAM3 returned an image that OpenCV could not decode")

        if image.ndim == 3 and image.shape[2] == 4:
            alpha = image[:, :, 3]
            if np.any(alpha < 250):
                return (alpha > 0).astype(np.uint8) * 255

        gray = image if image.ndim == 2 else cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
        high = gray > 127
        low = ~high

        # Raw mask endpoints can encode foreground as either white or black.
        # Prefer the polarity that occupies less of the image border and does
        # not cover almost the entire frame.
        def polarity_cost(candidate: np.ndarray) -> float:
            border = np.concatenate(
                (candidate[0], candidate[-1], candidate[:, 0], candidate[:, -1])
            )
            area = float(np.mean(candidate))
            cost = float(np.mean(border)) * 4.0
            if area < 0.002 or area > 0.98:
                cost += 10.0
            return cost

        binary = high if polarity_cost(high) <= polarity_cost(low) else low
        return binary.astype(np.uint8) * 255

    @staticmethod
    def _largest_component(mask: np.ndarray) -> np.ndarray:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            (mask > 0).astype(np.uint8), connectivity=8
        )
        if count <= 1:
            return mask
        areas = stats[1:, cv2.CC_STAT_AREA]
        selected = 1 + int(np.argmax(areas))
        return (labels == selected).astype(np.uint8) * 255

    def _restore_source_mask(
        self, encoded_mask: bytes, geometry: dict[str, int]
    ) -> np.ndarray:
        decoded = cv2.imdecode(np.frombuffer(encoded_mask, np.uint8), cv2.IMREAD_UNCHANGED)
        square_mask = self._binary_from_image(decoded)
        if square_mask.shape != (self.input_size, self.input_size):
            square_mask = cv2.resize(
                square_mask,
                (self.input_size, self.input_size),
                interpolation=cv2.INTER_NEAREST,
            )

        x1 = geometry["pad_left"]
        y1 = geometry["pad_top"]
        x2 = x1 + geometry["resized_width"]
        y2 = y1 + geometry["resized_height"]
        unpadded = square_mask[y1:y2, x1:x2]
        restored = cv2.resize(
            unpadded,
            (geometry["source_width"], geometry["source_height"]),
            interpolation=cv2.INTER_NEAREST,
        )
        return self._largest_component(restored)

    @staticmethod
    def _score(result: dict[str, Any]) -> float:
        scores = result.get("scores") or []
        if scores and scores[0] is not None:
            return float(scores[0])
        metadata = result.get("metadata") or []
        if metadata and metadata[0].get("score") is not None:
            return float(metadata[0]["score"])
        return 0.0

    def segment(self, frame: np.ndarray) -> tuple[np.ndarray, float, dict[str, Any]]:
        started = time.perf_counter()
        letterboxed, geometry = self._letterbox(frame)
        temporary_path = ""
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temporary:
                temporary_path = temporary.name
            if not cv2.imwrite(temporary_path, letterboxed):
                raise RuntimeError("Could not encode the SAM3 upload image")
            image_url = self._fal_client.upload_file(temporary_path)
        finally:
            if temporary_path and os.path.exists(temporary_path):
                os.unlink(temporary_path)

        prompts = list(dict.fromkeys((self.prompt, *self.FALLBACK_PROMPTS)))
        attempted_prompts: list[str] = []
        prompt_failures: list[str] = []
        selected_prompt = ""
        selected_result: dict[str, Any] | None = None
        selected_mask: np.ndarray | None = None
        selected_area = 0
        minimum_area = max(100, int(frame.shape[0] * frame.shape[1] * 0.002))

        for prompt in prompts:
            attempted_prompts.append(prompt)
            print(
                f"[SAM3] trying prompt {prompt!r} "
                f"({len(attempted_prompts)}/{len(prompts)})"
            )
            result = self._fal_client.subscribe(
                self.ENDPOINT,
                arguments={
                    "image_url": image_url,
                    "prompt": prompt,
                    # Match the fal configuration that was verified manually.
                    # The returned PNG normally carries a transparent/non-mask
                    # background, which _binary_from_image converts to a mask.
                    "apply_mask": True,
                    "output_format": "png",
                    "return_multiple_masks": True,
                    "max_masks": 3,
                    "include_scores": True,
                    "include_boxes": True,
                },
            )
            masks = result.get("masks") or []
            # Some fal response variants place the primary result under `image`
            # even when the `masks` list is empty. Accept either documented form.
            primary = next(
                (
                    item
                    for item in masks
                    if isinstance(item, dict) and item.get("url")
                ),
                None,
            )
            image = result.get("image")
            if primary is None and isinstance(image, dict) and image.get("url"):
                primary = image
            if primary is None:
                prompt_failures.append(f"{prompt}: no mask")
                continue

            mask = self._restore_source_mask(self._download(primary["url"]), geometry)
            area = int(np.count_nonzero(mask))
            if area < minimum_area:
                prompt_failures.append(f"{prompt}: mask too small ({area} pixels)")
                continue

            selected_prompt = prompt
            selected_result = result
            selected_mask = mask
            selected_area = area
            break

        if selected_result is None or selected_mask is None:
            raise RuntimeError(
                "SAM3 returned no usable garment mask after prompt fallbacks "
                f"(attempts={attempted_prompts!r}, failures={prompt_failures!r})"
            )

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        details = {
            "backend": "fal-ai/sam-3/image",
            "prompt": selected_prompt,
            "requested_prompt": self.prompt,
            "attempted_prompts": attempted_prompts,
            "prompt_failures": prompt_failures,
            "selected_confidence": self._score(selected_result),
            "boxes": selected_result.get("boxes") or [],
            "metadata": selected_result.get("metadata") or [],
            "letterbox": geometry,
            "mask_area_pixels": selected_area,
        }
        return selected_mask, elapsed_ms, details
