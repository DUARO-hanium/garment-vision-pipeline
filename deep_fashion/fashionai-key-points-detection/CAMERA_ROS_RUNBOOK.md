# 카메라 → WSL → ROS 2 연결 런북

Arducam B0196를 Windows에서 WSL로 넘기고, gscam으로 `/camera1/image_raw`에 퍼블리시한 뒤
파지점 추출을 돌리는 전체 순서. 명령은 전부 이 저장소의 기존 스크립트 규약을 따른다.

---

## 0. 터미널 역할 (고정)

| 터미널 | 역할 | 특징 |
|---|---|---|
| **PS** | Windows PowerShell (관리자) | usbipd로 USB를 WSL에 넘긴다 |
| **A** | WSL — 카메라 | `run_gscam_b0196.sh` 실행 후 **계속 점유**. 다른 명령 쓰지 말 것 |
| **B** | WSL — 확인 | `ros2 topic` 으로 상태 점검 |
| **C** | WSL — 추론 | `run_grasp_point_mvp.sh` 등 실행 |

터미널 A를 다른 용도로 재사용하는 것이 사고의 가장 흔한 원인이다. gscam은 포그라운드로
`/dev/video0`을 잡고 있고, Ctrl+C 하면 그 순간 모든 구독자가 프레임을 못 받는다.

---

## 1. USB를 WSL로 넘기기 — 터미널 PS

```powershell
usbipd list
# BUSID  VID:PID    DEVICE                          STATE
# 2-3    0c45:636d  Arducam B0196 / USB Camera      Not shared     <- BUSID 확인

usbipd bind   --busid 2-3      # 최초 1회만. 관리자 권한 필요
usbipd attach --wsl --busid 2-3
```

> `usbipd attach --wsl` 이 없다고 하면 구버전이다: `usbipd wsl attach --busid 2-3`

**attach는 다음 경우에 전부 풀린다** — 그때마다 `usbipd attach`를 다시 해야 한다:

- WSL 종료 / `wsl --shutdown` / 재부팅
- Windows 절전·최대 절전에서 복귀
- USB 케이블을 뺐다 꽂음
- 카메라가 스스로 USB 리셋 (max8mp 모드에서 관찰됨)

작업 끝나고 Windows에서 카메라를 쓰려면: `usbipd detach --busid 2-3`

---

## 2. WSL에서 장치 확인 — 터미널 B

```bash
ls -l /dev/video*                                  # /dev/video0 이 보여야 한다
lsusb | grep -i -E "arducam|camera"
v4l2-ctl -d /dev/video0 --list-formats-ext | head -30   # MJPG 1920x1080 지원 확인
```

`/dev/video0`이 없으면 → **1번으로 돌아가서 재attach**. 다른 걸 먼저 의심하지 말 것.

---

## 3. 카메라 퍼블리시 — 터미널 A

```bash
cd /mnt/d/문서/Users/23011/Documents/26-한이음/ros_projects
./run_gscam_b0196.sh quality        # 1920x1080 MJPG @ 30fps  ← 기본으로 이걸 쓴다
```

| 모드 | 해상도 | 언제 |
|---|---|---|
| `quality` | 1920×1080 @30 | **기본.** 기존 실측 결과가 전부 이 해상도다 |
| `realtime` | 1280×720 @30 | CPU 부하가 문제될 때 |
| `max8mp` | 3264×2448 @15 | **쓰지 말 것.** 스크립트 주석대로 JPEG 깨짐 + USB 끊김. 진단용 단발성만 |

환경변수로 조정 가능:

```bash
ARDUCAM_DEVICE=/dev/video2 ./run_gscam_b0196.sh quality   # 장치 번호가 다를 때
ARDUCAM_ROTATE=none        ./run_gscam_b0196.sh quality   # 회전 끄기 (기본 clockwise)
```

스크립트가 `fuser`로 장치 점유를 먼저 검사한다. "already in use"가 나오면
cheese·ffmpeg·OpenCV `VideoCapture`·이전 gscam 중 하나가 살아 있는 것이다.

---

## 4. 퍼블리시 확인 — 터미널 B

```bash
source /opt/ros/jazzy/setup.bash

ros2 topic list | grep camera1
# /camera1/camera_info
# /camera1/image_raw

ros2 topic hz /camera1/image_raw                 # ~30 Hz 나와야 정상
ros2 topic echo /camera1/camera_info --once --no-arr
```

`ros2 topic list`에는 보이는데 `hz`가 안 올라오면 GStreamer 파이프라인이 죽은 것이다.
터미널 A의 로그를 볼 것.

---

## 5. 파지점 추출 — 터미널 C

```bash
cd /mnt/d/문서/Users/23011/Documents/26-한이음/ros_projects/DUARO-hanium/Detection/deep_fashion/fashionai-key-points-detection

./run_grasp_point_mvp.sh                 # 라이브 프레임 한 장
./run_grasp_point_mvp.sh --demo --fast   # 저장 이미지 배치 (카메라 불필요)
```

스크립트가 ROS 소싱 → venv 활성화 순으로 처리하고, 실행 전에 토픽 존재 여부를 확인한다.

---

## 행동강령

**① 소싱 순서는 ROS 먼저, venv 다음.**
저장소의 모든 실행 스크립트가 이 순서다.

```bash
source /opt/ros/jazzy/setup.bash
source /home/jiyoung/.venvs/sam2/bin/activate
```

뒤집으면 venv의 numpy와 ROS의 `cv_bridge`가 어긋나서 import 시점에 깨진다.
그리고 ROS setup 스크립트를 소싱하는 동안에는 `set -u`를 켜지 말 것 —
setup 스크립트가 아직 정의되지 않은 변수를 참조한다. (`set -eo pipefail` → 소싱 → `set -u`)

**② `/dev/video0`은 한 번에 하나만 연다.**
gscam이 잡고 있는 동안 다른 프로그램이 같은 장치를 열면 둘 다 깨진다.
추론 스크립트는 카메라를 직접 열지 않고 **ROS 토픽만 구독**하므로 gscam과 충돌하지 않는다.
카메라를 직접 여는 코드(`cv2.VideoCapture(0)`)를 새로 짜지 말 것.

**③ `/dev/video0`이 없으면 usbipd부터 의심한다.**
WSL 재시작·절전 복귀·케이블 재삽입 후에는 항상 풀려 있다. 드라이버나 gscam을 먼저 건드리지 말 것.

**④ 회전 설정이 파지 판정에 직접 영향을 준다.**
`ARDUCAM_ROTATE=clockwise`(기본)가 GStreamer 파이프라인 안에서 적용되므로
`/camera1/image_raw`는 이미 회전된 영상이다. 파이프라인의 `vertical_order` 게이트
(어깨 v < 밑단 v)는 이 방향을 전제로 한다. **회전 설정을 바꾸면 이 게이트가 반대로 뒤집힌다.**
저장된 이미지로 튜닝한 임계값을 라이브에 그대로 쓸 때 방향이 같은지 반드시 확인할 것.

**⑤ 지금 MVP는 `image_raw`, world 변환은 `image_rect`를 쓴다.**
`DUARO-hanium-total_keypoint_uv_to_world/README.md`가 명시하듯 `uv_to_world` 런타임은
**rectified** 영상의 UV를 전제로 한다. 지금 파지점 MVP는 `/camera1/image_raw`를 쓰므로,
world 좌표까지 이어붙일 때는 rectify 노드를 띄우고 토픽을 바꿔야 한다:

```bash
CAMERA_TOPIC=/camera1/image_rect ./run_grasp_point_mvp.sh
```

이걸 안 맞추면 캘리브레이션이 아무리 정확해도 좌표가 렌즈 왜곡만큼 어긋난다.

**⑥ 해상도를 바꾸면 임계값을 다시 본다.**
`pair_distance` 게이트는 마스크 대각 길이 대비 비율이라 해상도에 둔감하지만,
`keypoint_score_threshold`(heatmap peak 절댓값)와 SAM2 처리 시간은 해상도에 따라 변한다.

**⑦ 종료는 역순.**
추론(C) → 확인(B) → gscam(A, Ctrl+C) → 필요 시 `usbipd detach`.
gscam을 먼저 죽이면 구독 중인 스크립트가 타임아웃까지 매달린다.

---

## 증상별 대처

| 증상 | 원인 | 대처 |
|---|---|---|
| `/dev/video0 is missing` | usbipd attach 풀림 | 1번 재실행 |
| `already in use` + `fuser -v` 출력 | 다른 프로세스가 장치 점유 | 그 프로세스 종료 후 재시도 |
| `ros2 topic list`에 `/camera1/*` 없음 | gscam이 죽었거나 소싱 안 됨 | 터미널 A 로그 확인, B에서 ROS 소싱 확인 |
| 토픽은 있는데 `hz` 0 | GStreamer 파이프라인 정지 | 터미널 A 로그 확인, gscam 재시작 |
| `No image received ... within 15.0 seconds` | 위 두 가지 중 하나 | 4번으로 먼저 확인 |
| `ModuleNotFoundError: rclpy` | venv만 활성화하고 ROS 소싱 안 함 | 행동강령 ① |
| JPEG 깨짐 / USB 끊김 | `max8mp` 모드 | `quality`로 돌아갈 것 |
