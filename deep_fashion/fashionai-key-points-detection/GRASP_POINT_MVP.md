# 반품 의류 파지점 추출 MVP

검수 대상 의류에서 **어디를 잡아야 안정적으로 grasp가 되는지**를 이미지 좌표로
뽑아내는 최소 구현. 시연 영상용 오버레이 렌더링까지 포함한다.

## 왜 이 순서인가 (설계 근거)

처음 구상은 "FlingBot value net으로 어깨/허리를 찾을 때까지 threshold를 넘게 반복하고,
찾으면 FashionAI로 확인"이었다. 세 가지 이유로 순서를 뒤집었다.

1. **FlingBot value net의 학습 목적이 다르다.**
   그것은 *fling 후 커버리지가 얼마나 늘어나는가*를 예측하는 value function이지
   "어깨/밑단이 어디인가"를 찾는 모델이 아니다. 실제로 `outputs/flingbot_value_once/20260822_145503`
   결과는 어깨가 아니라 **밑단 두 점** (505,938)-(1300,938)을 골랐다. 모델이 틀린 게 아니라
   원래 그렇게 학습됐다.

2. **후보 생성 구조상 semantic 조건을 표현할 수 없다.**
   `flingbot_value_once.select_grasp_pair`는 value map peak를 중심으로 `±PIX_GRASP_DIST(8)`
   떨어진 대칭 쌍만 만든다. 두 점 사이 거리가 `crop_side × scale / 4`로, 즉 (rotation, scale)
   조합으로만 결정된다 — 옷의 실제 어깨-밑단 거리로 결정되지 않는다.

3. **"threshold 넘을 때까지 반복"은 무한루프가 된다.**
   value net은 같은 프레임에 deterministic이다. 장면이 안 바뀌면 100번 돌려도 같은 답이다.
   루프가 의미를 가지려면 사이에 상태를 바꾸는 액션(뒤척임)이 필요하고, 그 자리는
   duaro `main.py`의 `Phase.SHUFFLE`이다.

그래서 **FashionAI = semantic 후보 생성기, FlingBot = 파지 안정도 검증기**로 역할을 나눴다.

```
프레임
 └─ SAM2 자동 마스크                        옷 영역 분리
     └─ FashionAI CPN 키포인트 + heatmap score
         └─ 게이트 5종
             ① keypoint_score   신뢰도 ≥ 임계값
             ② inside_mask      파지점이 옷 영역 안
             ③ vertical_order   어깨(v) < 밑단(v)
             ④ pair_distance    간격이 옷 대각 대비 0.25~1.60
             ⑤ flingbot_value   파지 안정도 ≥ 임계값
                 └─ 전부 통과 → (u,v) 두 점 + 사람이 읽는 판정 근거
                    하나라도 실패 → 사유를 남기고 뒤척임 후 재시도
```

## 파일

| 파일 | 역할 |
|---|---|
| `garment_grasp_pipeline.py` | 코어 로직 (CLI 없음). duaro 레포에서 import 해서 쓴다 |
| `grasp_point_mvp.py` | 실행 진입점 + 오버레이/영상 렌더러 |
| `run_grasp_point_mvp.sh` | 실행 스크립트 |

기존 파일을 그대로 재사용한다 — 새로 학습하거나 다시 구현한 것은 없다:
`sam2_garment_keypoint_once.run_sam2_auto` / `mask_crop_box` / `point_near_mask` / `receive_ros_frame`,
`fashionai_keypoint_capture.predict_keypoints_with_heatmaps` / `load_net` / `KEYPOINT_NAMES`,
`flingbot_value_once.load_value_net` / `square_cloth_crop` / `build_transformed_batch` / `inverse_transform_matrix`.

## 실행

```bash
source /home/jiyoung/.venvs/sam2/bin/activate
cd /mnt/d/문서/Users/23011/Documents/26-한이음/ros_projects/DUARO-hanium/Detection/deep_fashion/fashionai-key-points-detection

# ROS 카메라 한 장 (run_gscam_b0196.sh가 떠 있어야 함)
./run_grasp_point_mvp.sh

# 저장된 이미지 한 장
./run_grasp_point_mvp.sh real_test5.jpg

# 시연 영상 (real_test*.jpg 전부 → demo.mp4)
./run_grasp_point_mvp.sh --demo

# FlingBot 생략, 키포인트만 빠르게 (torch 불필요)
./run_grasp_point_mvp.sh --demo --fast
```

산출물:

```
outputs/grasp_point_mvp/<timestamp>/
  frames/<name>_result.png   원본 + 마스크 + 키포인트 + 파지점 + 판정 자막
  frames/<name>_value.png    FlingBot value map
  json/<name>.json           좌표·점수·게이트 통과 내역
  demo.mp4                   --demo 일 때
  summary.json               전체 통과율
```

## 임계값 — 반드시 실측 후 조정할 것

`GraspConfig`에 모여 있고 CLI로도 덮어쓸 수 있다.

| 항목 | 기본값 | 근거 / 조정 방법 |
|---|---|---|
| `keypoint_score_threshold` | 10.0 | `sam2_garment_keypoint_once.py`의 기존 기본값을 그대로 가져옴. `--fast`로 여러 장 돌려 json의 `keypoints[].score` 분포를 보고 조정 |
| `flingbot_value_threshold` | 0.10 | **표본이 3장뿐이라 사실상 임시값이다.** 실측 결과 하나가 0.369였다. 10장 이상 돌려 `flingbot.value` 분포를 보고 정할 것 |
| `min/max_pair_distance_ratio` | 0.25 / 1.60 | 마스크 bbox 대각 길이 대비 |
| `mask_tolerance_ratio` | 0.03 | 파지점이 마스크에서 벗어나도 허용할 여유 |

## 알려진 한계

- **모델이 `blouse_100.onnx` 하나뿐이다.** 티셔츠·맨투맨도 blouse로 처리한다. MVP에서는 허용,
  나중에 카테고리 분류를 앞에 붙이거나 `--clothing-type`을 수동 지정.
- **FashionAI는 평평하게 펼쳐진 옷 사진으로 학습됐다.** 구겨진 반품 의류에서는 score가 떨어진다.
  이건 버그가 아니라 게이트 ①이 걸러내야 하는 정상 상황이고, 그래서 뒤척임 재시도가 필요하다.
- **FlingBot 안정도는 참고 점수다.** 후보 쌍의 기하가 FlingBot action space
  (고정 그립 간격 × 12 rotation × 8 scale) 밖이면 `flingbot.trustworthy=false`로 표시되고
  게이트를 막지 않는다. json에서 이 값을 확인할 것.
- **SAM2가 CPU에서 10~13초 걸린다.** 시연 파이프라인에서 병목이다. GPU를 쓰거나
  마스크를 배경 차분/색 분리로 대체하는 것이 다음 개선 지점.
- **여기 나오는 (u, v)는 이미지 좌표다.** 로봇에 보내려면
  `DUARO-hanium-total_keypoint_uv_to_world/run_uv_to_world_capture.py`로 world 변환을 하고
  양팔 reachability를 확인해야 한다. 이 단계 없이 좌표를 그대로 명령하지 말 것.

## 검증 기록

`locate_pair_in_value_maps`(우리 후보 쌍 → FlingBot value map 셀 정변환)는
`flingbot_value_once`의 역변환과 왕복 일치해야 한다. 무작위 테스트 결과:

- 역/정변환 왕복 잔차: **~1e-13 px**
- `select_grasp_pair`가 뱉은 쌍으로부터 원래 (transform, row, col) 복원: **60/60**
  (rotation ±90°가 같은 파지 축을 가리키는 중복이 있어, 동률 후보를 모두 보고
  그 중 최대 value를 쓴다)
- 로그된 실제 결과 `(505,938)-(1300,938)`, `distance=795.0`을 수식으로 정확히 재현

## 다음 단계

1. `--fast`로 `real_test*.jpg` 12장을 돌려 keypoint score 분포 확인 → 임계값 확정
2. FlingBot 포함으로 다시 돌려 `flingbot.value` 분포 확인 → 안정도 임계값 확정
3. `--demo`로 시연 영상 생성
4. duaro 레포 `grasp/detector.py` · `grasp/selector.py`에서 이 모듈을 import 해 `main.py`에 연결
5. `run_uv_to_world_capture.py`와 이어 붙여 world 좌표까지
