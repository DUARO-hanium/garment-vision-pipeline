# DUARO 의류 비전 파이프라인

반품 의류의 펼치기·파지 좌표를 카메라 영상에서 찾기 위해 작성한 코드입니다. 기존 `DUARO-hanium/Detection` 저장소에 올리지 않았던 로컬 작업을 별도 저장소로 모았습니다.

현재 사용하는 흐름은 다음과 같습니다.

```text
Arducam → ROS /camera1/image_raw → 장면 정지 감지
→ SAM3 옷 마스크 → FashionAI 어깨·밑단 좌표와 점수
→ 점수 통과: 좌표를 옷 안쪽으로 보정하여 GRASP 후보 출력
→ 점수 미달: 마스크의 주축과 수평 구간으로 UNFOLD 후보 출력
```

FlingBot의 가치 신경망을 이용한 코드는 이전 실험으로 보관합니다. **현재 실행하는 판정 경로에는 FlingBot을 사용하지 않습니다.**

## 주요 파일

- `run_gscam_b0196.sh`, `configure_arducam_b0196.sh`: Arducam B0196 연결과 ROS 2 `gscam` 영상 발행.
- `deep_fashion/fashionai-key-points-detection/run_garment_inspection_mvp.sh`: 현재 실시간 시연 프로그램의 실행 스크립트.
- `deep_fashion/fashionai-key-points-detection/garment_inspection_roi_mvp.py`: ROS 영상 수신, 옷의 움직임·정지 감지, 어깨·밑단 단계, 점수 판정, 펼치기 좌표 계산, 화면 표시, JSON·영상 저장을 담당하는 메인 코드. 어깨 또는 밑단의 두 점이 점수 기준을 통과하면 각도·ROI 검사를 생략하고 좌표를 옷 안쪽으로 보정합니다. 통과하지 못하면 SAM3 마스크에서 로봇 가까운 쪽의 수평 파지 구간을 찾습니다.
- `deep_fashion/fashionai-key-points-detection/sam3_fal_segmenter.py`: fal.ai의 SAM3를 호출하고, 마스크가 없을 때 텍스트 프롬프트를 재시도하며, 결과 마스크를 원본 카메라 좌표계로 복원합니다.
- `deep_fashion/fashionai-key-points-detection/fashionai_keypoint_capture.py`, `fashionai_key_points_detection_utils.py`: FashionAI의 512×512 전처리, ONNX Runtime 추론, 히트맵 좌표 해석 및 원본 영상 좌표 변환을 위한 수정 코드.
- `deep_fashion/fashionai-key-points-detection/sam2_garment_keypoint_once.py`: 로컬 SAM2와 FashionAI를 이미지 한 장에 실행하던 이전 실험. 현재 메인 코드에서는 이 파일의 마스크 보조 함수만 가져오며 SAM2 모델 자체는 실행하지 않습니다.
- `deep_fashion/fashionai-key-points-detection/flingbot_value_once.py`: FlingBot 가치 신경망을 단일 프레임에 적용한 독립 실험. 별도 체크포인트가 필요합니다.
- `deep_fashion/fashionai-key-points-detection/garment_inspection_mvp.py`, `garment_grasp_pipeline.py`, `grasp_point_mvp.py`: 이전 파지·펼치기 전략을 비교하기 위해 보관한 실험 코드.
- `deep_fashion/fashionai-key-points-detection/capture_ros_frame_once.py`, `run_resolution_pair.sh`, `summarize_resolution_pair.py`: 카메라 한 장 저장과 해상도 비교 도구.
- `util/model_utils.py`: FashionAI 모델 파일 확인·다운로드에 필요한 보조 코드.

## 실행 환경

기본 실행 스크립트는 Ubuntu/WSL, ROS 2 Jazzy, `/dev/video0`에 연결한 UVC 카메라, ROS `gscam` 패키지, `/home/jiyoung/.venvs/sam2` 가상환경을 기준으로 작성했습니다. 다른 PC에서는 실행 스크립트의 가상환경 경로와 카메라 장치 경로를 변경해야 합니다.

주요 Python 의존성은 `numpy`, `opencv-python`, `onnxruntime`, `fal-client`이며, ROS 환경에는 `rclpy`, `sensor_msgs`, `cv_bridge`가 필요합니다. 과거 SAM2·FlingBot 실험에는 PyTorch, SciPy 및 해당 모델의 의존성이 추가로 필요합니다.

### 1. 카메라 발행

첫 번째 터미널에서 저장소 루트 기준으로 실행합니다.

```bash
ARDUCAM_ROTATE=none bash ./run_gscam_b0196.sh realtime
```

이 터미널을 종료하지 않고 유지합니다.

### 2. 의류 비전 실행

두 번째 터미널에서 fal.ai의 **실제 API 키를 로컬 환경 변수로만** 등록합니다. 키를 코드나 Git 커밋에 넣지 마세요.

```bash
cd deep_fashion/fashionai-key-points-detection
read -rsp 'FAL API 키: ' FAL_KEY; echo; export FAL_KEY
bash ./run_garment_inspection_mvp.sh --ros-topic /camera1/image_raw --clothing-type blouse --inference-interval 5.0 --sam3-prompt shirt --keypoint-threshold 150 --keypoint-inset-ratio 0.018 --max-attempts 3 --display-stage-seconds 0 --pipeline-panel-width 0
```

`--display-stage-seconds 0`은 단계별 화면 연출을 생략하고, `--pipeline-panel-width 0`은 왼쪽 흐름도 패널을 숨깁니다. 추론 로직은 그대로 실행됩니다.

FashionAI 실행에는 `blouse_100.onnx`가 필요합니다. 없으면 모델 보조 코드가 원래 배포 주소에서 다운로드를 시도합니다. 다른 옷 종류의 가중치도 별도로 준비해야 합니다. 이 저장소에는 작은 `blouse_100.onnx.prototxt`만 포함하고 `.onnx`, `.pth` 등의 가중치는 포함하지 않았습니다. FlingBot 실험에는 별도로 구한 `flingbot.pth`와 로컬 SAM2 모델의 경로를 실행 옵션으로 지정해야 합니다.

## 결과 해석과 주의 사항

- SAM3는 외부 유료 API입니다. 분할을 위해 카메라 프레임이 fal.ai 서버로 전송되며, 추론에는 수 초가 걸릴 수 있습니다. 이 코드는 지연 시간이 보장되는 로봇 제어 루프가 아닙니다.
- FashionAI 점수는 히트맵의 최대 활성값입니다. 확률이나 백분율이 아닙니다.
- 표시되는 `GRASP`와 `UNFOLD`는 **영상 픽셀 `(u, v)` 기준 후보 좌표**입니다. 로봇팔 구동 명령이 아닙니다. 실제 파지에는 카메라 보정, 월드 좌표 변환, 팔 도달성·충돌 검사와 실물 검증이 추가로 필요합니다.
- `outputs/`, 촬영 사진·영상, API 키와 모델 가중치는 저장소에서 제외했습니다.

## 원본 코드와 모델

FashionAI 연동 코드는 `ailia-models`의 FashionAI 예제를 바탕으로 수정했습니다. FlingBot 실험은 가치 신경망 구조와 가중치 로딩 경로를 분리해 평가하며, 원래 시뮬레이터와 로봇 동작 코드는 포함하지 않습니다. 모델 및 체크포인트를 별도로 사용할 때는 각각의 원본 프로젝트의 라이선스와 배포 조건을 확인하세요. 이 저장소는 해당 가중치를 재배포하지 않습니다.
