"""반품 의류 파지점 추출 MVP — 실행 진입점 + 시연 영상 렌더러.

사용 예
-------
  # 1) ROS 카메라 한 장
  python grasp_point_mvp.py --ros-topic /camera1/image_raw

  # 2) 이미 찍어둔 테스트 이미지 한 장
  python grasp_point_mvp.py --input real_test5.jpg

  # 3) 폴더 전체를 돌려서 시연 영상까지 뽑기  ← 영상 만들 때 이걸 쓴다
  python grasp_point_mvp.py --input-dir . --pattern "real_test*.jpg" --video demo.mp4

  # 4) 키포인트만 빠르게 확인 (FlingBot 생략, torch 불필요)
  python grasp_point_mvp.py --input real_test5.jpg --no-flingbot

산출물
------
  outputs/grasp_point_mvp/<timestamp>/
    frames/<name>_result.png     원본 + 마스크 + 키포인트 + 파지점 + 판정 자막
    frames/<name>_value.png      FlingBot value map (선택된 transform)
    json/<name>.json             좌표·점수·게이트 통과 내역
    demo.mp4                     --video 지정 시
    summary.json

주의: 여기서 나오는 (u, v)는 이미지 좌표다. 로봇에 보내려면
run_uv_to_world_capture.py로 world 변환 + reachability 검사를 거쳐야 한다.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from garment_grasp_pipeline import (
    PAIR_MODES,
    GraspConfig,
    GraspResult,
    detect_grasp_points,
    load_keypoint_net,
    receive_ros_frame,
)
from fashionai_keypoint_capture import KEYPOINT_NAMES

# ---------------------------------------------------------------------------
# 한글 렌더링 (OpenCV putText는 한글을 못 그린다 → PIL + 시스템 폰트)
# ---------------------------------------------------------------------------
FONT_CANDIDATES = [
    "/mnt/c/Windows/Fonts/malgun.ttf",           # WSL에서 윈도우 맑은고딕
    "/mnt/c/Windows/Fonts/malgunsl.ttf",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
]

_FONT_CACHE: dict[int, object] = {}
_FONT_PATH: str | None = None
_FONT_WARNED = False


def _font(size: int):
    """한글 폰트를 찾아 캐시한다. 없으면 None (ASCII 폴백)."""
    global _FONT_PATH, _FONT_WARNED
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    try:
        from PIL import ImageFont
    except ImportError:
        if not _FONT_WARNED:
            print("[WARN] Pillow가 없어 한글 자막을 영문으로 대체한다 (pip install pillow)")
            _FONT_WARNED = True
        _FONT_CACHE[size] = None
        return None

    if _FONT_PATH is None:
        for path in FONT_CANDIDATES:
            if Path(path).exists():
                _FONT_PATH = path
                break
    if _FONT_PATH is None:
        if not _FONT_WARNED:
            print("[WARN] 한글 폰트를 못 찾아 자막을 영문으로 대체한다")
            _FONT_WARNED = True
        _FONT_CACHE[size] = None
        return None

    _FONT_CACHE[size] = ImageFont.truetype(_FONT_PATH, size)
    return _FONT_CACHE[size]


def text_size(text: str, size: int) -> tuple[int, int]:
    """렌더링될 텍스트의 (폭, 높이) 근사치. 라벨을 화면 안에 붙이는 데 쓴다."""
    font = _font(size)
    if font is None:
        return int(len(text) * size * 0.58), size
    left, top, right, bottom = font.getbbox(text)
    return int(right - left), int(bottom - top)


def draw_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    size: int = 28,
    color: tuple[int, int, int] = (255, 255, 255),
    ascii_fallback: str | None = None,
    shadow: bool = False,
) -> np.ndarray:
    """BGR 이미지에 한글 텍스트를 그린다. 폰트가 없으면 ascii_fallback을 cv2로 그린다.

    shadow=True면 밝은 배경 위에서도 읽히도록 검은 외곽선을 함께 그린다.
    """
    font = _font(size)
    if font is None:
        body = ascii_fallback or text.encode("ascii", "replace").decode("ascii")
        baseline = (origin[0], origin[1] + size)
        if shadow:
            cv2.putText(image, body, baseline, cv2.FONT_HERSHEY_SIMPLEX,
                        size / 34.0, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(image, body, baseline, cv2.FONT_HERSHEY_SIMPLEX,
                    size / 34.0, color, 2, cv2.LINE_AA)
        return image

    from PIL import Image, ImageDraw

    pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    if shadow:
        draw.text(origin, text, font=font, fill=(0, 0, 0),
                  stroke_width=max(2, size // 9), stroke_fill=(0, 0, 0))
    draw.text(origin, text, font=font, fill=color[::-1])
    image[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    return image


def wrap_text(text: str, max_chars: int) -> list[str]:
    lines: list[str] = []
    current = ""
    for token in text.split(" "):
        if len(current) + len(token) + 1 > max_chars and current:
            lines.append(current)
            current = token
        else:
            current = f"{current} {token}".strip()
    if current:
        lines.append(current)
    return lines


# ---------------------------------------------------------------------------
# 오버레이 렌더링
# ---------------------------------------------------------------------------
OK_GREEN = (90, 220, 90)
FAIL_RED = (70, 70, 245)
CYAN = (255, 220, 40)
POINT_A_COLOR = (255, 140, 60)   # 어깨 / 좌
POINT_B_COLOR = (90, 90, 255)    # 밑단 / 우


def render_result(frame: np.ndarray, result: GraspResult, title: str) -> np.ndarray:
    canvas = frame.copy()
    height, width = canvas.shape[:2]
    scale = width / 1920.0
    bar_height_hint = int(64 * scale)  # 상단 타이틀 바 높이 (라벨이 가려지지 않게)

    # 1) 마스크 오버레이 + 윤곽
    if result.mask is not None:
        tint = np.zeros_like(canvas)
        tint[:, :] = (40, 200, 40)
        blended = cv2.addWeighted(canvas, 0.78, tint, 0.22, 0)
        canvas[result.mask > 0] = blended[result.mask > 0]
        contours, _ = cv2.findContours(result.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, (0, 230, 0), max(2, int(3 * scale)))

    # 2) 전체 키포인트를 옅게 (후보 생성기가 무엇을 봤는지 보여주는 용도)
    for hit in result.keypoints:
        if not hit.visible:
            continue
        cv2.circle(canvas, (hit.u, hit.v), max(3, int(5 * scale)), (200, 200, 200), -1)

    # 3) 최종 파지점 두 개
    if result.point_a and result.point_b:
        a = (result.point_a.u, result.point_a.v)
        b = (result.point_b.u, result.point_b.v)
        line_color = OK_GREEN if result.ok else (120, 120, 120)
        cv2.line(canvas, a, b, line_color, max(3, int(5 * scale)), cv2.LINE_AA)
        for point, color, label in (
            (a, POINT_A_COLOR, result.point_a.label),
            (b, POINT_B_COLOR, result.point_b.label),
        ):
            radius = max(8, int(16 * scale))
            cv2.circle(canvas, point, radius, color, -1)
            cv2.circle(canvas, point, radius + max(2, int(4 * scale)), (255, 255, 255), max(2, int(3 * scale)))

            caption = f"{label} ({point[0]}, {point[1]})"
            label_size = max(16, int(34 * scale))
            caption_width, caption_height = text_size(caption, label_size)
            margin = radius + int(10 * scale)
            # 화면 밖으로 잘리지 않게 좌/우, 상/하를 뒤집어 배치한다.
            text_x = point[0] + margin
            if text_x + caption_width > width - int(10 * scale):
                text_x = point[0] - margin - caption_width
            text_x = max(int(10 * scale), min(text_x, width - caption_width - int(10 * scale)))
            text_y = point[1] - margin - caption_height
            if text_y < bar_height_hint:
                text_y = point[1] + margin
            text_y = max(bar_height_hint, min(text_y, height - caption_height - int(10 * scale)))
            draw_text(
                canvas,
                caption,
                (text_x, text_y),
                size=label_size,
                color=color,
                ascii_fallback=f"{label} ({point[0]},{point[1]})",
                shadow=True,
            )

    # 4) 상단 타이틀 바
    bar_height = bar_height_hint
    cv2.rectangle(canvas, (0, 0), (width, bar_height), (24, 24, 24), -1)
    status = "PASS" if result.ok else "RETRY"
    status_color = OK_GREEN if result.ok else FAIL_RED
    draw_text(
        canvas,
        f"{title}   |   {result.clothing_type} / {result.pair_mode}",
        (int(18 * scale), int(12 * scale)),
        size=max(16, int(32 * scale)),
        color=(240, 240, 240),
        ascii_fallback=f"{title} | {result.clothing_type}/{result.pair_mode}",
    )
    draw_text(
        canvas,
        status,
        (width - int(150 * scale), int(12 * scale)),
        size=max(18, int(34 * scale)),
        color=status_color,
        ascii_fallback=status,
    )

    # 5) 하단 판정 근거 패널 — 시연 영상 자막이 되는 부분
    panel_lines: list[tuple[str, tuple[int, int, int], str]] = []
    for check in result.checks:
        mark = "O" if check.ok else "X"
        color = OK_GREEN if check.ok else FAIL_RED
        panel_lines.append((f"[{mark}] {check.name}: {check.detail_ko}", color, f"[{mark}] {check.name}: {check.detail_en}"))

    reason_lines = wrap_text(result.reason_ko, 78)
    panel_text_size = max(14, int(26 * scale))
    line_height = int(panel_text_size * 1.45)
    panel_height = line_height * (len(panel_lines) + len(reason_lines) + 1) + int(28 * scale)
    panel_top = height - panel_height

    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, panel_top), (width, height), (18, 18, 18), -1)
    canvas = cv2.addWeighted(overlay, 0.80, canvas, 0.20, 0)

    y = panel_top + int(12 * scale)
    for text, color, fallback in panel_lines:
        draw_text(canvas, text, (int(20 * scale), y), size=panel_text_size, color=color, ascii_fallback=fallback)
        y += line_height

    y += int(6 * scale)
    for index, line in enumerate(reason_lines):
        prefix = "→ " if index == 0 else "   "
        draw_text(
            canvas,
            prefix + line,
            (int(20 * scale), y),
            size=int(panel_text_size * 1.15),
            color=status_color,
            ascii_fallback=(prefix + result.reason_en) if index == 0 else "",
        )
        y += line_height

    return canvas


def render_value_map(result: GraspResult, size: int = 512) -> np.ndarray | None:
    if result.value_map is None:
        return None
    normalized = cv2.normalize(result.value_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    info = result.flingbot
    if "row" in info and "col" in info:
        cv2.circle(colored, (int(info["col"]), int(info["row"])), 2, (255, 255, 255), 1)
    colored = cv2.resize(colored, (size, size), interpolation=cv2.INTER_NEAREST)
    draw_text(
        colored,
        f"FlingBot value={info.get('value', float('nan')):.3f}",
        (12, 10),
        size=22,
        color=(255, 255, 255),
        ascii_fallback=f"value={info.get('value', 0):.3f}",
    )
    return colored


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SAM2 + FashionAI 후보 생성 → 게이트 → FlingBot 검증으로 파지점 두 개를 뽑는다."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="이미지 파일 한 장")
    source.add_argument("--input-dir", help="이미지 폴더 (배치 · 영상용)")
    source.add_argument("--ros-topic", help="ROS 2 sensor_msgs/Image 토픽, 예: /camera1/image_raw")

    parser.add_argument("--pattern", default="*.jpg", help="--input-dir와 함께 쓰는 파일 패턴")
    parser.add_argument("--clothing-type", choices=sorted(KEYPOINT_NAMES), default="blouse")
    parser.add_argument("--pair-mode", choices=PAIR_MODES, default="top_bottom",
                        help="top_bottom=어깨(최상단)+밑단(최하단) / shoulder=양 어깨")
    parser.add_argument("--sam-model", default="/home/jiyoung/models/sam2_t.pt")
    parser.add_argument(
        "--flingbot-checkpoint",
        default="/mnt/d/문서/Users/23011/Documents/26-한이음/ros_projects/flingbot/flingbot.pth",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--no-flingbot", action="store_true", help="FlingBot 검증 생략 (빠른 확인용)")

    parser.add_argument("--keypoint-score-threshold", type=float, default=10.0)
    parser.add_argument("--flingbot-value-threshold", type=float, default=0.10)
    parser.add_argument("--min-pair-distance-ratio", type=float, default=0.25)
    parser.add_argument("--max-pair-distance-ratio", type=float, default=1.60)

    parser.add_argument("--output-dir", default="outputs/grasp_point_mvp")
    parser.add_argument("--video", help="시연 영상 파일명 (예: demo.mp4)")
    parser.add_argument("--video-fps", type=float, default=30.0)
    parser.add_argument("--video-hold-seconds", type=float, default=3.0,
                        help="영상에서 프레임 한 장을 몇 초 보여줄지")
    parser.add_argument("--video-width", type=int, default=1280)
    parser.add_argument("--ros-timeout", type=float, default=15.0)
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> GraspConfig:
    return GraspConfig(
        clothing_type=args.clothing_type,
        pair_mode=args.pair_mode,
        sam_model=args.sam_model,
        flingbot_checkpoint=args.flingbot_checkpoint,
        device=args.device,
        use_flingbot=not args.no_flingbot,
        keypoint_score_threshold=args.keypoint_score_threshold,
        flingbot_value_threshold=args.flingbot_value_threshold,
        min_pair_distance_ratio=args.min_pair_distance_ratio,
        max_pair_distance_ratio=args.max_pair_distance_ratio,
    )


def collect_inputs(args: argparse.Namespace) -> list[tuple[str, np.ndarray]]:
    if args.input:
        frame = cv2.imread(args.input, cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"이미지를 읽지 못했다: {args.input}")
        return [(Path(args.input).stem, frame)]

    if args.input_dir:
        paths = sorted(Path(args.input_dir).glob(args.pattern))
        if not paths:
            raise RuntimeError(f"패턴에 맞는 파일이 없다: {args.input_dir}/{args.pattern}")
        frames = []
        for path in paths:
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if frame is None:
                print(f"[WARN] 건너뜀 (읽기 실패): {path}")
                continue
            frames.append((path.stem, frame))
        return frames

    print(f"[INFO] {args.ros_topic} 에서 프레임 한 장 대기 중")
    frame = receive_ros_frame(args.ros_topic, args.ros_timeout)
    return [(args.ros_topic.strip("/").replace("/", "_"), frame)]


def main() -> int:
    args = parse_args()
    cfg = build_config(args)

    run_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    (run_dir / "frames").mkdir(parents=True, exist_ok=True)
    (run_dir / "json").mkdir(parents=True, exist_ok=True)

    items = collect_inputs(args)
    print(f"[INFO] 입력 {len(items)}장 · FashionAI '{cfg.clothing_type}' 모델 로드")
    net = load_keypoint_net(cfg)

    rendered: list[np.ndarray] = []
    summary = []
    for index, (name, frame) in enumerate(items, start=1):
        print(f"[{index}/{len(items)}] {name} ({frame.shape[1]}x{frame.shape[0]})")
        try:
            result = detect_grasp_points(frame, net, cfg)
        except Exception as error:  # noqa: BLE001 — 배치 도중 한 장 때문에 멈추지 않게
            print(f"[ERROR] {name}: {error}")
            traceback.print_exc()
            summary.append({"name": name, "ok": False, "error": str(error)})
            continue

        canvas = render_result(frame, result, title=name)
        cv2.imwrite(str(run_dir / "frames" / f"{name}_result.png"), canvas)
        value_image = render_value_map(result)
        if value_image is not None:
            cv2.imwrite(str(run_dir / "frames" / f"{name}_value.png"), value_image)
        with open(run_dir / "json" / f"{name}.json", "w", encoding="utf-8") as stream:
            json.dump(result.to_json(), stream, indent=2, ensure_ascii=False)

        rendered.append(canvas)
        summary.append(
            {
                "name": name,
                "ok": result.ok,
                "reason_ko": result.reason_ko,
                "point_a": [result.point_a.u, result.point_a.v] if result.point_a else None,
                "point_b": [result.point_b.u, result.point_b.v] if result.point_b else None,
                "flingbot_value": result.flingbot.get("value"),
                "timings_ms": result.timings_ms,
            }
        )
        status = "PASS" if result.ok else "RETRY"
        print(f"    -> {status}: {result.reason_ko}")

    passed = sum(1 for item in summary if item.get("ok"))
    with open(run_dir / "summary.json", "w", encoding="utf-8") as stream:
        json.dump(
            {
                "config": {
                    "clothing_type": cfg.clothing_type,
                    "pair_mode": cfg.pair_mode,
                    "keypoint_score_threshold": cfg.keypoint_score_threshold,
                    "flingbot_value_threshold": cfg.flingbot_value_threshold,
                    "use_flingbot": cfg.use_flingbot,
                },
                "total": len(summary),
                "passed": passed,
                "items": summary,
            },
            stream,
            indent=2,
            ensure_ascii=False,
        )

    if args.video and rendered:
        write_video(rendered, run_dir / args.video, args)

    print(f"\n[DONE] {passed}/{len(summary)} 통과 · 결과: {run_dir}")
    print("[SAFETY] 이미지 좌표 후보다. uv_to_world 변환과 reachability 검사 전에는 로봇에 보내지 말 것.")

    if not args.no_show and len(rendered) == 1:
        preview = rendered[0]
        factor = min(1.0, 1500.0 / preview.shape[1], 850.0 / preview.shape[0])
        cv2.imshow("grasp point MVP (아무 키나 누르면 종료)",
                   cv2.resize(preview, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


def write_video(frames: list[np.ndarray], path: Path, args: argparse.Namespace) -> None:
    target_width = args.video_width
    first = frames[0]
    target_height = int(round(first.shape[0] * target_width / first.shape[1]))
    target_height += target_height % 2  # H.264 호환

    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), args.video_fps, (target_width, target_height)
    )
    if not writer.isOpened():
        print(f"[WARN] 영상 파일을 열지 못했다: {path}")
        return

    hold = max(1, int(round(args.video_fps * args.video_hold_seconds)))
    for frame in frames:
        resized = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
        for _ in range(hold):
            writer.write(resized)
    writer.release()
    print(f"[INFO] 시연 영상 저장: {path}")


if __name__ == "__main__":
    sys.exit(main())
