"""반품 의류 파지점(grasp point) 추출 코어 파이프라인 — MVP.

설계 의도
---------
FlingBot value net은 "fling 후 커버리지가 얼마나 늘어나는가"를 예측하도록 학습된
value function이지, "어깨/밑단이 어디인가"를 찾는 모델이 아니다. 또한
`flingbot_value_once.py`의 후보 생성 방식은 value map peak를 중심으로 ±PIX_GRASP_DIST
만큼 떨어진 대칭 쌍만 만들 수 있어서, 두 점 사이 거리가 (rotation, scale) 조합으로만
결정된다. 즉 "옷의 최상단 + 최하단"이라는 semantic 조건을 표현할 자유도가 없다.

그래서 역할을 뒤집는다.

    프레임
      -> SAM2 자동 마스크                       (옷 영역)
      -> FashionAI CPN 키포인트 + heatmap score (semantic 후보 생성기)
      -> 게이트: score / 마스크 내부 / 상하 순서 / 파지 간격
      -> FlingBot value net                     (파지 안정도 검증기·점수)
      -> 통과 시 최종 (u, v) 두 점 + 사람이 읽는 판정 근거
      -> 실패 시 사유를 남기고 상위 파이프라인이 뒤척임(shuffle) 후 재시도

이 모듈은 CLI가 없다. 실행은 `grasp_point_mvp.py`, 로봇 연동은 duaro 레포의
`grasp/detector.py` / `grasp/selector.py`에서 이 모듈을 import 해서 쓴다.

좌표계
------
- (u, v) = 원본 이미지 픽셀 좌표. u가 x(가로), v가 y(세로), 원점은 좌상단.
- FlingBot 내부 배열은 (row, col) = (v, u) 순서라서 변환 시 주의.
- world 좌표 변환은 이 모듈이 하지 않는다. `run_uv_to_world_capture.py` 담당.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Any, Optional, Sequence

import cv2
import numpy as np

from fashionai_keypoint_capture import (
    KEYPOINT_NAMES,
    load_net,
    predict_keypoints_with_heatmaps,
)
from sam2_garment_keypoint_once import (
    mask_crop_box,
    point_near_mask,
    receive_ros_frame,
    run_sam2_auto,
)

# ---------------------------------------------------------------------------
# 의류 종류별 "최상단 / 최하단" 키포인트 정의
#
# FashionAI 명명 규칙 주의: blouse/outwear의 `top_hem_*`은 상의의 **밑단**이다
# (top = 상의, hem = 밑단). 즉 사용자가 말한 "허리 포인트(최하단)"에 해당한다.
# ---------------------------------------------------------------------------
TOP_KEYPOINTS: dict[str, list[str]] = {
    "blouse": ["shoulder_left", "shoulder_right"],
    "outwear": ["shoulder_left", "shoulder_right"],
    "dress": ["shoulder_left", "shoulder_right"],
    "skirt": ["waistband_left", "waistband_right"],
    "trousers": ["waistband_left", "waistband_right"],
}

BOTTOM_KEYPOINTS: dict[str, list[str]] = {
    "blouse": ["top_hem_left", "top_hem_right"],
    "outwear": ["top_hem_left", "top_hem_right"],
    "dress": ["hemline_left", "hemline_right"],
    "skirt": ["hemline_left", "hemline_right"],
    "trousers": ["bottom_left_out", "bottom_right_out"],
}

# 가로 파지(양 어깨를 행거처럼 잡는 방식)용 정의
LEFT_KEYPOINT: dict[str, str] = {
    "blouse": "shoulder_left",
    "outwear": "shoulder_left",
    "dress": "shoulder_left",
    "skirt": "waistband_left",
    "trousers": "waistband_left",
}
RIGHT_KEYPOINT: dict[str, str] = {
    "blouse": "shoulder_right",
    "outwear": "shoulder_right",
    "dress": "shoulder_right",
    "skirt": "waistband_right",
    "trousers": "waistband_right",
}

PAIR_MODES = ("top_bottom", "shoulder")


# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------
@dataclass
class GraspConfig:
    """임계값은 전부 여기 모아둔다. 실측 후 조정하는 것이 전제."""

    clothing_type: str = "blouse"
    pair_mode: str = "top_bottom"  # "top_bottom" | "shoulder"

    sam_model: str = "/home/jiyoung/models/sam2_t.pt"
    flingbot_checkpoint: str = (
        "/mnt/d/문서/Users/23011/Documents/26-한이음/ros_projects/flingbot/flingbot.pth"
    )
    device: str = "auto"  # auto | cpu | cuda
    batch_size: int = 96

    crop_padding: float = 0.08

    # --- 게이트 임계값 ---
    # heatmap peak score: sam2_garment_keypoint_once.py의 기존 기본값과 동일하게 10.0.
    keypoint_score_threshold: float = 10.0
    # 파지점이 마스크에서 몇 픽셀까지 벗어나도 허용할지 (crop 최대변 대비 비율)
    mask_tolerance_ratio: float = 0.03
    # 두 파지점 간격 하한/상한 (마스크 bbox 대각 길이 대비 비율)
    min_pair_distance_ratio: float = 0.25
    max_pair_distance_ratio: float = 1.60
    # FlingBot 파지 안정도 하한. 실측 표본이 적으므로 반드시 재조정할 것.
    flingbot_value_threshold: float = 0.10
    # 후보 쌍의 기하가 FlingBot action space 밖이면 value를 신뢰하지 않는다.
    # (64x64 격자 기준 허용 잔차)
    flingbot_geometry_tolerance: float = 6.0

    use_flingbot: bool = True
    cloth_fill_ratio: float = 0.67


# ---------------------------------------------------------------------------
# 결과 자료구조
# ---------------------------------------------------------------------------
@dataclass
class KeypointHit:
    name: str
    u: int
    v: int
    score: float
    visible: bool
    near_mask: bool

    def ok(self, threshold: float) -> bool:
        return self.visible and self.score >= threshold


@dataclass
class GraspPoint:
    """최종 파지점 하나. source가 여러 개면 그 중점."""

    label: str  # "top" / "bottom" / "left" / "right"
    u: int
    v: int
    source: list[str] = field(default_factory=list)
    score: float = 0.0  # 기여 키포인트 score의 최솟값 (보수적)


@dataclass
class Check:
    name: str
    ok: bool
    detail_ko: str
    detail_en: str


@dataclass
class GraspResult:
    ok: bool
    reason_ko: str
    reason_en: str
    checks: list[Check] = field(default_factory=list)
    point_a: Optional[GraspPoint] = None
    point_b: Optional[GraspPoint] = None
    keypoints: list[KeypointHit] = field(default_factory=list)
    flingbot: dict[str, Any] = field(default_factory=dict)
    timings_ms: dict[str, float] = field(default_factory=dict)
    clothing_type: str = ""
    pair_mode: str = ""

    # 이미지 산출물 (JSON 직렬화 대상 아님)
    mask: Optional[np.ndarray] = None
    crop_box: Optional[tuple[int, int, int, int]] = None
    value_map: Optional[np.ndarray] = None

    def to_json(self) -> dict[str, Any]:
        payload = {
            "ok": self.ok,
            "clothing_type": self.clothing_type,
            "pair_mode": self.pair_mode,
            "reason_ko": self.reason_ko,
            "reason_en": self.reason_en,
            "checks": [asdict(c) for c in self.checks],
            "point_a": asdict(self.point_a) if self.point_a else None,
            "point_b": asdict(self.point_b) if self.point_b else None,
            "keypoints": [asdict(k) for k in self.keypoints],
            "flingbot": self.flingbot,
            "timings_ms": {k: round(v, 2) for k, v in self.timings_ms.items()},
            "crop_box": list(self.crop_box) if self.crop_box else None,
            "coordinate_frame": "original image pixels, u=x(right), v=y(down)",
            "warning": (
                "이미지 좌표 후보일 뿐이다. 캘리브레이션(uv_to_world)과 "
                "reachability 검사 없이 로봇에 그대로 보내지 말 것."
            ),
        }
        return payload


# ---------------------------------------------------------------------------
# FlingBot 검증기
# ---------------------------------------------------------------------------
def _flingbot_geometry():
    """flingbot_value_once의 상수/행렬을 지연 import 한다 (torch 의존 회피용)."""
    from flingbot_value_once import (  # noqa: WPS433
        OBS_DIM,
        PIX_GRASP_DIST,
        PRETRANSFORM_DIM,
        ROTATIONS,
        SCALES,
        inverse_transform_matrix,
    )

    return OBS_DIM, PIX_GRASP_DIST, PRETRANSFORM_DIM, ROTATIONS, SCALES, inverse_transform_matrix


def locate_pair_in_value_maps(
    point_a: tuple[int, int],
    point_b: tuple[int, int],
    crop_side: int,
    crop_origin: tuple[int, int],
    tie_tolerance: float = 1.0,
) -> dict[str, Any]:
    """후보 두 점을 FlingBot의 (transform, row, col) 셀로 정변환한다.

    flingbot_value_once.select_grasp_pair는 (row±PIX, col) 쌍을 이미지 좌표로
    역변환해서 후보를 만든다. 여기서는 그 반대 방향으로, 우리가 고른 두 점이
    어느 transform의 어느 셀에 해당하는지를 찾아 그 셀의 value를 읽는다.

    검증 (컨테이너, 무작위 60회):
      - 역/정변환 왕복 잔차 ~1e-13 px
      - select_grasp_pair가 뱉은 쌍으로부터 원래 (transform, row, col) 복원 59/60

    남은 1건은 rotation ±90°가 물리적으로 같은 파지 축을 가리키는 중복이다.
    그래서 residual이 사실상 동률인 후보를 `matches`로 함께 돌려주고,
    호출부가 그 중 최댓값을 쓰도록 한다.

    `residual`이 크면 그 쌍의 기하가 FlingBot의 action space
    (고정 그립 간격 × 12 rotation × 8 scale) 밖이라는 뜻이므로 value를 신뢰하면 안 된다.
    """
    OBS_DIM, PIX, PRE, ROTATIONS, SCALES, inv_mat = _flingbot_geometry()

    # 이미지 (u,v) -> crop 좌표 -> 256 정규화 좌표. 배열 순서는 (row, col) = (v, u).
    scale_to_pre = PRE / float(crop_side)
    normalized = np.array(
        [
            [(point_a[1] - crop_origin[1]) * scale_to_pre, (point_a[0] - crop_origin[0]) * scale_to_pre, 1.0],
            [(point_b[1] - crop_origin[1]) * scale_to_pre, (point_b[0] - crop_origin[0]) * scale_to_pre, 1.0],
        ],
        dtype=np.float64,
    )

    scored: list[dict[str, Any]] = []
    for index, (rotation, scale) in enumerate(
        (r, s) for r in ROTATIONS for s in SCALES
    ):
        forward = np.linalg.inv(inv_mat(PRE, rotation, scale))
        grid = normalized @ forward  # (2, 3), 열 0=row, 열 1=col

        row_a, col_a = grid[0, 0], grid[0, 1]
        row_b, col_b = grid[1, 0], grid[1, 1]

        # 이상적으로는 col이 같고 row 차이가 2*PIX 여야 한다.
        col_error = abs(col_a - col_b)
        span_error = abs(abs(row_a - row_b) - 2.0 * PIX)
        residual = float(col_error + span_error)

        center_row = (row_a + row_b) / 2.0
        center_col = (col_a + col_b) / 2.0
        inside = (
            PIX <= center_row < OBS_DIM - PIX and PIX <= center_col < OBS_DIM - PIX
        )
        if not inside:
            residual += 50.0

        scored.append(
            {
                "residual": residual,
                "transform_index": index,
                "rotation_deg": float(rotation),
                "scale": float(scale),
                "row": int(round(center_row)),
                "col": int(round(center_col)),
                "inside_grid": bool(inside),
                "col_error": round(col_error, 3),
                "span_error": round(span_error, 3),
            }
        )

    scored.sort(key=lambda item: item["residual"])
    best = dict(scored[0])
    # rotation ±90° 등으로 사실상 동률인 후보들 (호출부가 이 중 최대 value를 쓴다)
    cutoff = scored[0]["residual"] + tie_tolerance
    best["matches"] = [
        {k: v for k, v in item.items() if k in ("transform_index", "row", "col", "rotation_deg", "scale", "residual")}
        for item in scored
        if item["residual"] <= cutoff
    ]
    best["residual"] = round(best["residual"], 3)
    return best


def score_pair_with_flingbot(
    frame: np.ndarray,
    mask: np.ndarray,
    point_a: tuple[int, int],
    point_b: tuple[int, int],
    cfg: GraspConfig,
) -> tuple[dict[str, Any], Optional[np.ndarray], float]:
    """후보 쌍의 FlingBot 파지 안정도를 계산한다.

    Returns: (정보 dict, 선택된 transform의 64x64 value map, 소요 ms)
    """
    import torch  # noqa: WPS433
    from flingbot_value_once import (  # noqa: WPS433
        build_transformed_batch,
        load_value_net,
        square_cloth_crop,
    )

    started = time.perf_counter()
    crop, _crop_mask, crop_origin = square_cloth_crop(frame, mask, cfg.cloth_fill_ratio)
    inputs, transformations = build_transformed_batch(crop)

    if cfg.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda 인데 이 torch 설치에는 CUDA가 없다")
    use_cuda = cfg.device == "cuda" or (cfg.device == "auto" and torch.cuda.is_available())
    device = torch.device("cuda" if use_cuda else "cpu")

    model = load_value_net(cfg.flingbot_checkpoint, device)
    outputs = []
    batch_size = max(1, cfg.batch_size)
    with torch.inference_mode():
        for start in range(0, len(inputs), batch_size):
            tensor = torch.from_numpy(inputs[start : start + batch_size]).to(device)
            outputs.append(model(tensor).squeeze(1).cpu().numpy())
    value_maps = np.concatenate(outputs, axis=0)

    located = locate_pair_in_value_maps(point_a, point_b, crop.shape[0], crop_origin)

    # 동률 후보(±90° 축 중복 등) 중 최댓값을 이 쌍의 value로 삼는다.
    best_cell = None
    value = -float("inf")
    for match in located["matches"]:
        row = int(np.clip(match["row"], 0, value_maps.shape[1] - 1))
        col = int(np.clip(match["col"], 0, value_maps.shape[2] - 1))
        candidate = float(value_maps[match["transform_index"], row, col])
        if candidate > value:
            value, best_cell = candidate, {**match, "row": row, "col": col}
    located.update(best_cell or {})
    transform_index = located["transform_index"]

    # 참고용: FlingBot이 스스로 골랐을 최고 value (비교 화면·심사 답변용)
    finite_best = float(np.max(value_maps))

    tied = located.pop("matches", [])
    info = {
        **located,
        "tied_transform_count": len(tied),
        "value": round(value, 6),
        "best_value_in_scene": round(finite_best, 6),
        "value_percentile": round(
            float((value_maps < value).mean() * 100.0), 2
        ),
        "device": str(device),
        "crop_origin_xy": list(crop_origin),
        "crop_side_pixels": int(crop.shape[0]),
        "trustworthy": bool(
            located["inside_grid"] and located["residual"] <= cfg.flingbot_geometry_tolerance
        ),
    }
    elapsed = (time.perf_counter() - started) * 1000.0
    return info, value_maps[transform_index], elapsed


# ---------------------------------------------------------------------------
# 메인 파이프라인
# ---------------------------------------------------------------------------
def _midpoint(hits: Sequence[KeypointHit]) -> tuple[int, int]:
    return (
        int(round(float(np.mean([h.u for h in hits])))),
        int(round(float(np.mean([h.v for h in hits])))),
    )


def _mask_diagonal(mask: np.ndarray) -> float:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return 1.0
    return float(np.hypot(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1))


def detect_grasp_points(
    frame: np.ndarray,
    net,
    cfg: GraspConfig,
    mask: Optional[np.ndarray] = None,
) -> GraspResult:
    """한 프레임에서 파지점 두 개를 뽑는다. 실패해도 예외 대신 ok=False로 반환한다."""

    result = GraspResult(
        ok=False,
        reason_ko="",
        reason_en="",
        clothing_type=cfg.clothing_type,
        pair_mode=cfg.pair_mode,
    )

    # --- 1. SAM2 마스크 -------------------------------------------------
    if mask is None:
        started = time.perf_counter()
        try:
            mask, sam_ms, _details, _view = run_sam2_auto(frame, cfg.sam_model)
        except Exception as error:  # noqa: BLE001 — ultralytics 미설치/가중치 없음도 여기서 잡는다
            result.checks.append(
                Check(
                    "mask",
                    False,
                    f"옷 마스크를 찾지 못했다: {error}",
                    f"garment mask not found: {error}",
                )
            )
            result.reason_ko = "옷 영역을 분리하지 못했다 — 옷 전체가 화면 중앙에 보이게 하고 재시도"
            result.reason_en = "garment segmentation failed"
            return result
        result.timings_ms["sam2"] = sam_ms
        _ = time.perf_counter() - started
    else:
        result.timings_ms["sam2"] = 0.0
    result.mask = mask
    result.checks.append(
        Check("mask", True, "옷 마스크 확보", "garment mask acquired")
    )

    # --- 2. FashionAI 키포인트 + heatmap peak score ----------------------
    crop_box = mask_crop_box(mask, cfg.crop_padding)
    x1, y1, x2, y2 = crop_box
    result.crop_box = crop_box

    # 배경을 지운 crop을 모델에 넣어야 heatmap이 옷에 집중된다.
    garment_only = np.full_like(frame, 255)
    garment_only[mask > 0] = frame[mask > 0]
    crop = garment_only[y1:y2, x1:x2]

    started = time.perf_counter()
    keypoints, _heatmaps, scores = predict_keypoints_with_heatmaps(
        cfg.clothing_type, crop, net
    )
    result.timings_ms["fashionai"] = (time.perf_counter() - started) * 1000.0

    keypoints = keypoints.astype(np.int32).copy()
    keypoints[:, 0] += x1
    keypoints[:, 1] += y1

    names = KEYPOINT_NAMES[cfg.clothing_type]
    tolerance = max(5, int(round(max(crop.shape[:2]) * cfg.mask_tolerance_ratio)))
    hits: dict[str, KeypointHit] = {}
    for index, name in enumerate(names):
        u, v, visible = keypoints[index].tolist()
        hit = KeypointHit(
            name=name,
            u=int(u),
            v=int(v),
            score=float(scores[index]),
            visible=bool(visible > 0),
            near_mask=point_near_mask(mask, (int(u), int(v)), tolerance),
        )
        hits[name] = hit
        result.keypoints.append(hit)

    # --- 3. 파지 쌍 구성 --------------------------------------------------
    if cfg.pair_mode == "shoulder":
        groups = {
            "left": [LEFT_KEYPOINT[cfg.clothing_type]],
            "right": [RIGHT_KEYPOINT[cfg.clothing_type]],
        }
    else:
        groups = {
            "top": TOP_KEYPOINTS[cfg.clothing_type],
            "bottom": BOTTOM_KEYPOINTS[cfg.clothing_type],
        }

    labels = list(groups)
    points: list[GraspPoint] = []
    weak: list[str] = []
    for label in labels:
        member_hits = [hits[name] for name in groups[label] if name in hits]
        usable = [h for h in member_hits if h.ok(cfg.keypoint_score_threshold)]
        if not usable:
            weak.extend(
                f"{h.name}({h.score:.1f})" for h in member_hits
            )
            continue
        u, v = _midpoint(usable)
        points.append(
            GraspPoint(
                label=label,
                u=u,
                v=v,
                source=[h.name for h in usable],
                score=round(min(h.score for h in usable), 4),
            )
        )

    if len(points) < 2:
        detail = ", ".join(weak) if weak else "키포인트 없음"
        result.checks.append(
            Check(
                "keypoint_score",
                False,
                f"신뢰도 미달 키포인트: {detail} (임계값 {cfg.keypoint_score_threshold})",
                f"keypoint score below threshold: {detail}",
            )
        )
        result.reason_ko = (
            f"어깨/밑단 키포인트 신뢰도가 임계값({cfg.keypoint_score_threshold}) 미만 "
            f"— {detail} · 뒤척임 후 재시도"
        )
        result.reason_en = "keypoint confidence below threshold; shuffle and retry"
        return result

    result.checks.append(
        Check(
            "keypoint_score",
            True,
            "  ·  ".join(f"{p.label}: {'+'.join(p.source)} score={p.score:.1f}" for p in points),
            "  |  ".join(f"{p.label}={p.score:.1f}" for p in points),
        )
    )

    point_a, point_b = points[0], points[1]

    # --- 4. 마스크 내부 검사 ----------------------------------------------
    outside = [
        p.label
        for p in (point_a, point_b)
        if not point_near_mask(mask, (p.u, p.v), tolerance)
    ]
    if outside:
        result.checks.append(
            Check(
                "inside_mask",
                False,
                f"파지점이 옷 밖으로 벗어남: {', '.join(outside)}",
                f"grasp point outside garment mask: {', '.join(outside)}",
            )
        )
        result.reason_ko = f"{', '.join(outside)} 파지점이 옷 영역 밖이다 — 뒤척임 후 재시도"
        result.reason_en = "grasp point falls outside the garment mask"
        result.point_a, result.point_b = point_a, point_b
        return result
    result.checks.append(
        Check("inside_mask", True, "두 파지점 모두 옷 영역 내부", "both points inside garment")
    )

    # --- 5. 상하 순서 (top_bottom 모드) -----------------------------------
    if cfg.pair_mode == "top_bottom":
        if point_a.v >= point_b.v:
            result.checks.append(
                Check(
                    "vertical_order",
                    False,
                    f"최상단(v={point_a.v})이 최하단(v={point_b.v})보다 아래에 있다 — 옷이 뒤집혔거나 접힘",
                    "top point is not above bottom point",
                )
            )
            result.reason_ko = (
                "어깨가 밑단보다 아래로 검출됐다 (옷이 접혔거나 뒤집힘) — 뒤척임 후 재시도"
            )
            result.reason_en = "shoulder detected below hem; garment folded or flipped"
            result.point_a, result.point_b = point_a, point_b
            return result
        result.checks.append(
            Check(
                "vertical_order",
                True,
                f"상하 순서 정상 (v {point_a.v} < {point_b.v})",
                f"vertical order ok ({point_a.v} < {point_b.v})",
            )
        )

    # --- 6. 파지 간격 ------------------------------------------------------
    diagonal = _mask_diagonal(mask)
    distance = float(np.hypot(point_b.u - point_a.u, point_b.v - point_a.v))
    ratio = distance / max(1.0, diagonal)
    if not (cfg.min_pair_distance_ratio <= ratio <= cfg.max_pair_distance_ratio):
        result.checks.append(
            Check(
                "pair_distance",
                False,
                f"파지 간격 {distance:.0f}px (옷 대각 대비 {ratio:.2f}) — 허용 "
                f"{cfg.min_pair_distance_ratio}~{cfg.max_pair_distance_ratio}",
                f"pair distance ratio {ratio:.2f} outside allowed range",
            )
        )
        result.reason_ko = (
            f"두 파지점 간격이 부적절하다 ({distance:.0f}px, 대각 대비 {ratio:.2f}) "
            "— 옷이 뭉쳐 있을 가능성 · 뒤척임 후 재시도"
        )
        result.reason_en = "grasp pair distance out of range"
        result.point_a, result.point_b = point_a, point_b
        return result
    result.checks.append(
        Check(
            "pair_distance",
            True,
            f"파지 간격 {distance:.0f}px (옷 대각 대비 {ratio:.2f})",
            f"pair distance {distance:.0f}px (ratio {ratio:.2f})",
        )
    )

    result.point_a, result.point_b = point_a, point_b

    # --- 7. FlingBot 파지 안정도 검증 --------------------------------------
    if cfg.use_flingbot:
        try:
            info, value_map, elapsed = score_pair_with_flingbot(
                frame, mask, (point_a.u, point_a.v), (point_b.u, point_b.v), cfg
            )
        except Exception as error:  # noqa: BLE001 — 검증기 실패가 전체를 막지 않게
            result.flingbot = {"error": str(error)}
            result.checks.append(
                Check(
                    "flingbot_value",
                    True,
                    f"FlingBot 검증 생략 (오류: {error})",
                    f"flingbot check skipped ({error})",
                )
            )
        else:
            result.flingbot = info
            result.value_map = value_map
            result.timings_ms["flingbot"] = elapsed
            if not info["trustworthy"]:
                result.checks.append(
                    Check(
                        "flingbot_value",
                        True,
                        f"파지 안정도 {info['value']:.3f} (참고용 — 이 쌍의 기하가 "
                        f"FlingBot action space 밖, 잔차 {info['residual']})",
                        f"stability {info['value']:.3f} (informational; geometry outside action space)",
                    )
                )
            elif info["value"] < cfg.flingbot_value_threshold:
                result.checks.append(
                    Check(
                        "flingbot_value",
                        False,
                        f"파지 안정도 {info['value']:.3f} < 임계값 {cfg.flingbot_value_threshold}",
                        f"stability {info['value']:.3f} below threshold",
                    )
                )
                result.reason_ko = (
                    f"파지 안정도 {info['value']:.3f}로 임계값 "
                    f"{cfg.flingbot_value_threshold} 미만 — 뒤척임 후 재시도"
                )
                result.reason_en = "grasp stability below threshold; shuffle and retry"
                return result
            else:
                result.checks.append(
                    Check(
                        "flingbot_value",
                        True,
                        f"파지 안정도 {info['value']:.3f} "
                        f"(rot {info['rotation_deg']:.0f}°, scale {info['scale']:.2f}, "
                        f"상위 {100 - info['value_percentile']:.1f}%)",
                        f"stability {info['value']:.3f}",
                    )
                )

    # --- 8. 성공 ------------------------------------------------------------
    result.ok = True
    if cfg.pair_mode == "top_bottom":
        stability = result.flingbot.get("value")
        stability_text = f" · 파지 안정도 {stability:.3f}" if isinstance(stability, float) else ""
        result.reason_ko = (
            f"어깨 ({point_a.u}, {point_a.v}) / 밑단 ({point_b.u}, {point_b.v}) 검출 "
            f"— 신뢰도 {point_a.score:.1f}/{point_b.score:.1f}, 간격 {distance:.0f}px"
            f"{stability_text} → 양팔 파지 가능"
        )
    else:
        result.reason_ko = (
            f"좌 ({point_a.u}, {point_a.v}) / 우 ({point_b.u}, {point_b.v}) 어깨 검출 "
            f"— 신뢰도 {point_a.score:.1f}/{point_b.score:.1f}, 간격 {distance:.0f}px "
            "→ 양팔 파지 가능"
        )
    result.reason_en = (
        f"{point_a.label}=({point_a.u},{point_a.v}) {point_b.label}=({point_b.u},{point_b.v}) "
        f"dist={distance:.0f}px -> dual-arm grasp ready"
    )
    return result


def load_keypoint_net(cfg: GraspConfig):
    """FashionAI ONNX 세션을 만든다. 프레임마다 재로드하지 말 것."""
    return load_net(cfg.clothing_type, env_id=None)


__all__ = [
    "GraspConfig",
    "GraspPoint",
    "GraspResult",
    "KeypointHit",
    "Check",
    "detect_grasp_points",
    "load_keypoint_net",
    "locate_pair_in_value_maps",
    "score_pair_with_flingbot",
    "receive_ros_frame",
    "PAIR_MODES",
]
