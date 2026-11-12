# 설계 A: LayaNav 코드 기반 NaVILA 상위 모델 교체

> 이 문서는 Claude Code에게 주는 작업 지시서다. 저장소 `rvlab-univ/laya_Nav` 안에서 작업한다.
> 짝이 되는 문서는 `design_B_laya_native.md`(Laya 원본 형식을 그대로 쓰는 안)이고, 두 안은 같은 데이터, 같은 평가로 비교한다.

## 0. 한 줄 요약

NaVILA의 상위 VLA(8B, 문장 생성)를 약 0.6B 결정 모델로 바꾼다. 기존 LayaNav v4의 인코더 입력 구조(과거 프레임, 포즈 기억, grounding)를 그대로 쓰고, 선택지만 "행동 3 + 바닥 패치 196"에서 **NaVILA 중간 수준 행동 10개**로 바꾼다. 언어 인코더는 ModernBERT-large로 키우고, 현재 프레임에 DINOv2 특징을 더한다. NaVILA의 하위 보행 정책은 건드리지 않는다.

## 1. 배경: 코드에서 확인한 사실

아래는 2026-11-11에 각 저장소 main을 읽고 확인한 내용이다. 구현 전에 한 번 더 대조한다.

### NaVILA (`AnjieCheng/NaVILA`, 커밋 76b98f2)

- 상위 모델은 `a8cheng/navila-llama3-8b-8f`. 비전 타워는 `google/siglip-so400m-patch14-384`, 프레임 8장(`scripts/train/sft_8frames.sh`).
- 프레임 선택: 과거 전체에서 균등하게 7장 + 현재 1장. 부족하면 앞쪽을 검은 이미지로 채운다(`evaluation/vlnce_baselines/navila_trainer.py`의 `sample_and_pad_images`).
- 출력은 문장이고 평가 코드가 정규식으로 해석한다(`navila_trainer.py` 222-283행).
  - `stop` / `move forward N cm` / `turn left N degree` / `turn right N degree`
  - N은 25·50·75 cm, 15·30·45°로 맞춰진다. 즉 **행동은 10개**다.
  - 실행은 Habitat 저수준 행동을 N/25회 또는 N/15회 반복한다.
- 학습 주석 형식: `{"video_id", "frames": [...], "q": 지시문, "a": 답 문장}`(`llava/data/dataset.py`의 `LazyVLNCEDataset`).

### LayaNav (`rvlab-univ/laya_Nav`, 커밋 6f94c26)

- 모델: `internnav/model/basemodel/laya_s2/laya_s2.py`(`LayaS2`, 판단), `laya_nav.py`(경로 헤드), `grounding.py`, `agent.py`.
- 데이터: `internnav/dataset/laya_s2_dataset.py`. InternData-N1 에피소드에서 샘플을 만든다. 에피소드마다 저수준 행동 열 `item["actions"]`(0 STOP, 1 FORWARD 0.25 m, 2 LEFT 15°, 3 RIGHT 15°)과 포즈가 있다.
- 등록 이름 `r2r_125cm_0_30` 등은 카메라 높이 125 cm, 정면 0°, 내려다보기 30°를 뜻한다. **정면 프레임은 `pitch_1` 쪽**이다(`frame_path`).
- 평가: `internnav/habitat_extensions/vln/laya_s2_eval.py`. `habitat_vln_evaluator.py` 119·195행에서 `model_args.mode`로 분기한다. 설정은 `scripts/eval/configs/habitat_laya_nav_cfg.py`.
- v4 검증 결과(`docs/laya_nav/results/laya_nav_v4_c2.md`): acc_goal 0.47, acc_stop은 0.79에서 0.64로 하락, 판단 과적합. 폐루프 SR/SPL은 아직 없다.

## 2. 범위

**한다**
- 새 모델 `LayaVLA`, 새 데이터셋, 학습 스크립트, 폐루프 평가 루프, 테스트.
- 기존 `laya_s2/` 코드와 체크포인트는 그대로 둔다. 새 디렉터리에 만들고 필요한 함수만 가져다 쓴다.

**하지 않는다**
- NaVILA 하위 보행 정책 수정, 실제 Go2 탑재.
- 내려다보기 뷰, offset head, latent 디코더, 경로 헤드(모두 뺀다).
- 실행 중인 학습을 건드리는 일. 연구실 서버는 공용이므로 GPU 사용 전에 `nvidia-smi`로 확인한다.

## 3. 모델

### 3.1 파일

| 새 파일 | 내용 |
|---|---|
| `internnav/model/basemodel/laya_vla/laya_vla.py` | `LayaVLAConfig`, `LayaVLA`, `compute_loss` |
| `internnav/model/basemodel/laya_vla/agent.py` | 평가용 에이전트(프레임 특징 캐시 포함) |
| `internnav/dataset/laya_vla_dataset.py` | 중간 수준 행동 라벨, 샘플 생성, collate |
| `scripts/train/laya_vla/train_laya_vla.py` | 학습 |
| `scripts/train/laya_vla/diagnose.py` | 오프라인 진단 |
| `internnav/habitat_extensions/vln/laya_vla_eval.py` | 폐루프 평가 루프 |
| `scripts/eval/configs/habitat_laya_vla_cfg.py` | 평가 설정 (`mode="laya_vla"`) |
| `tests/laya_vla/` | CPU 테스트(작은 인코더) |

`laya_s2`에서 그대로 가져다 쓸 것: `grounding.py` 전체, `history_ids`, `memory_features`, `robot_poses`, `mirror_instruction`, `mirror_poses`, `make_image_transform`, `mismatched`, `match_loss`.

### 3.2 설정 기본값

```python
text_encoder   = "answerdotai/ModernBERT-large"   # d=1024, 영어 전용
vision_encoder = "google/siglip2-base-patch16-224"
geo_encoder    = "facebook/dinov2-with-registers-base"  # None이면 끔
geo_image_size = 196          # 패치 14 → 14×14, SigLIP2 격자와 일치
image_size     = 224
num_history    = 32; hist_recent = 16; hist_pool = 4
hist_pose      = True; motion_tokens = 8; hist_grad = False
cur_pool       = 14           # 현재 프레임 196토큰, 풀링 없음
grounding      = True; match_head = True; grounding_chunks = 6
max_text_len   = 256
head_layers    = 2
actions = ["stop",
           "move forward 25 cm", "move forward 50 cm", "move forward 75 cm",
           "turn left 15 degree", "turn left 30 degree", "turn left 45 degree",
           "turn right 15 degree", "turn right 30 degree", "turn right 45 degree"]
```

### 3.3 입력 열

```
[CLS] 지시문 [SEP] | 선택지 10개 | 움직임 토큰 8 | 과거 32장×16 | 현재 196
```

길이는 약 1,030토큰이다(지시문 256, 선택지 약 60 기준).

- **선택지**: 각 행동을 `[MASK] + 행동 문구 토큰`으로 넣는다. 문구는 토크나이저로 한 번만 토큰화하고, 임베딩은 인코더의 단어 임베딩 표에서 가져온다. `[MASK]` 위치가 그 선택지의 점수 자리다. `action_emb` 학습 벡터는 쓰지 않는다.
- **세그먼트 임베딩**: 지시문 / 선택지·움직임 / 과거 / 현재 4종. 내려다보기용은 없앤다.
- **과거 프레임**: v4 그대로. 4×4 평균, `hist_slot_emb`, `hist_pose_proj`.
- **현재 프레임**: 196토큰 + `pos_cur`. grounding 지도는 현재 프레임에만 더한다.
- 영상 투영은 768 → 1024로 바뀐다.

### 3.4 DINOv2 결합

```python
cur_tok = vis_proj(siglip_patches) + geo_proj(dino_patches)   # [B, 196, 1024]
```

- `geo_proj`는 `Linear(768,1024) → GELU → Linear(1024,1024)`이고 마지막 층을 0으로 초기화한다. 저장소의 기존 add-on 방식(`ground_proj`)과 같아서, DINOv2 없이 학습한 체크포인트에 나중에 붙일 수 있다.
- DINOv2는 얼린다. 모듈 트리와 체크포인트 밖에 두는 것도 `grounder`와 같은 방식으로 한다.
- 입력은 196 px로 따로 리사이즈하고 **ImageNet 평균·표준편차**로 정규화한다(SigLIP2는 0.5/0.5). 같은 원본 이미지에서 두 번 변환한다.
- 출력에서 CLS와 register 토큰 4개를 버리고 패치 196개만 쓴다.
- 과거 프레임에는 적용하지 않는다.

### 3.5 출력 헤드

| 헤드 | 입력 → 출력 | 비고 |
|---|---|---|
| `scorer` | 선택지 `[MASK]` 벡터 10개 → 점수 10 | `LayerNorm → Linear → GELU → Linear(1)`, softmax |
| `progress_head` | `[CLS]` → 0~1 | 신규. `LayerNorm → Linear(1) → sigmoid` |
| `act_head` | `[CLS]` + 특징 4개 → 2 | 특징: 1등 확률, 1·2등 차이, 정규화 엔트로피, 진행도 예측값(모두 detach) |
| `match_head` | `[CLS]` → 1 | v3 그대로 |

### 3.6 손실

```
L = L_dec + 0.1·L_prog + 0.1·L_esc + 0.2·L_match
```

- `L_dec`: 목표 분포 `t`에 대해 `-(t·log q) + 0.5·spherical`. `laya_s2.compute_loss`의 식을 그대로 쓴다. `t`는 단계 1에서 one-hot, 단계 3부터 `(1-α)·onehot + α·교사 확률`(α 기본 0.5).
- STOP 샘플은 5배 반복한다(기존 `repeat_stops`). v4에서 STOP 정확도가 계속 떨어졌으므로 검증에서 STOP 정밀도와 재현율을 따로 본다.
- `L_prog`: Huber. 정답은 `현재 프레임 번호 / 에피소드 길이`.
- `L_esc`: "1등이 틀렸는가" 이진 분류(기존과 같음).
- v5 장치는 처음부터 켠다: `--mirror 0.5 --memory_dropout 0.5 --w_change 3`. 좌우 반전 시 행동 4~6과 7~9를 맞바꾼다.

## 4. 데이터와 라벨

### 4.1 중간 수준 행동 라벨

InternData-N1의 저수준 행동 열에서 만든다. 서버에 이미 받아 둔 r2r, rxr, scalevln을 그대로 쓴다.

```python
def mid_level_label(actions, t):
    """프레임 t에서 시작하는 같은 행동의 연속 길이(최대 3)로 10개 중 하나를 고른다."""
    a = actions[t]
    if a == 0: return 0                                  # stop
    run = 1
    while run < 3 and t + run < len(actions) and actions[t + run] == a:
        run += 1
    return {1: 0, 2: 3, 3: 6}[a] + run                   # 1..3 전진, 4..6 좌, 7..9 우
```

- 행동 열의 인덱싱은 `enumerate_samples`와 똑같이 맞춘다(`actions = item["actions"][1:] + [0]`).
- 샘플 위치: 에피소드를 처음부터 탐욕적으로 잘랐을 때의 경계(추론 때 실제로 판단하는 지점)는 모두 쓰고, 그 밖의 프레임은 `--offgrid_ratio`(기본 0.3) 비율로 섞는다.
- **영어 지시문만 쓴다.** RxR에는 힌디어·텔루구어 지시문이 섞여 있고 ModernBERT-large는 영어 전용이다. `laya_s2_dataset._NOT_ENGLISH`로 거른다.
- 검증 분할은 기존처럼 장면 단위(`--val_split scene`).

### 4.2 교사 라벨 (단계 3)

- 같은 샘플을 NaVILA 8B에 넣어 행동 10개의 확률을 얻는다. 프롬프트와 프레임 선택은 `navila_trainer.py`와 똑같이 한다.
- 방법: 답 문장 10개(`"The next action is move forward 75 cm."` 등) 각각의 로그 가능도를 구해 softmax 한다. 프롬프트 부분의 KV 캐시는 공유한다.
- 답 문장의 정확한 표기는 `NaVILA-Dataset/R2R/annotations.json`의 `"a"` 값을 세어 확인한다.
- 저장 형식은 `TeacherLatentStore`와 같게 한다(`keys.txt` + `probs.npy`, 키는 `sample_key`).
- NaVILA는 별도 conda 환경(transformers 4.37.2)이 필요하다. 추출 스크립트는 `scripts/train/laya_vla/extract_teacher_probs.py`.

## 5. 작업 순서와 통과 기준

각 단계가 통과해야 다음으로 간다. 수치는 모두 실제로 재서 `docs/laya_vla/`에 기록한다.

| 단계 | 내용 | 통과 기준 |
|---|---|---|
| 0 | 폐루프 평가 루프 | 정답 행동을 그대로 재생했을 때 R2R val_unseen 50 에피소드에서 SR 95% 이상 |
| 1 | 기본 모델 학습 (DINOv2 끔, one-hot) | 학습 손실 감소, 검증 10지선다 정확도 기록, 폐루프 SR/SPL 측정 |
| 2 | DINOv2 켬 | 단계 1과 같은 조건에서 비교표 작성 |
| 3 | 교사 soft label | 교사 일치율과 SR 변화 기록 |
| 4 | DAgger | 모델이 직접 주행한 상태에 교사 라벨을 붙여 재학습, SR 변화 기록 |
| 5 | 온도 보정 + escalate | ECE, "8B 호출 비율 대 SR" 곡선 |

단계 0을 가장 먼저 한다. 서버 Habitat에 libEGL이 필요하다(`docs/laya_nav/model_improvement/next_candidates.md` 참고).

### 5.1 폐루프 평가 루프 (단계 0)

```
매 판단:
  d = agent.decide(instruction, history, cur_image, poses, actions)   # 행동 번호 0..9
  stop이면 env.step(STOP)
  아니면 저수준 행동을 k번 실행 (전진 25k cm → FORWARD k회, 회전 15k° → 회전 k회)
  실행한 저수준 프레임을 모두 history에 넣는다
```

- LOOKDOWN / LOOKUP 호출과 경로 헤드 관련 코드는 없다.
- 지표는 기존 평가기의 SR, SPL, OS, NE, nDTW에 판단 1회 시간(ms)과 에피소드당 판단 횟수를 더한다.
- `habitat_vln_evaluator.py`의 mode 분기에 `"laya_vla"`를 추가한다.

### 5.2 오프라인 진단 (`diagnose.py`)

- 10지선다 정확도, 종류(정지/전진/좌/우) 정확도
- **전진 25/50/75 cm 혼동 행렬**(DINOv2 효과를 보는 주된 지표)
- STOP 정밀도·재현율, ECE
- 지름길 검사: 기억 입력 끔, 지시문 좌우 바꿈, 좌우 반전 샘플(기존 `diagnose_decision.py`의 검사를 옮긴다)
- 학습 정확도와 검증 정확도의 차이(용량 부족인지 일반화 문제인지 구분)

## 6. 테스트 (`tests/laya_vla/`)

기존 `tests/laya_s2/`처럼 작은 인코더로 CPU에서 돈다.

1. `mid_level_label`: 손으로 만든 행동 열에서 기대 라벨, 좌우 반전 대응
2. 순전파 출력 모양: `logits [B,10]`, `progress [B]`, `act_logits [B,2]`
3. `geo_proj` 0 초기화: DINOv2를 켜고 끈 출력이 초기 상태에서 같음
4. DINOv2 없이 저장한 체크포인트를 켠 설정으로 불러오기
5. 선택지 순서를 바꾸면 점수도 같은 순서로 바뀜
6. 포즈로 라벨 검증: FORWARD 한 번에 이동 약 0.25 m, 회전 한 번에 약 15°

## 7. 선택 실험 (본 작업 뒤)

- **인코더 크기**: mmBERT-base 대 ModernBERT-large. 같은 데이터로 한 번씩.
- **CLIP 교체**: `openai/clip-vit-base-patch16`. `grounding.py`의 패치 임베딩은 SigLIP의 attention-pooling 헤드에 맞춰져 있으므로, CLIP 쪽은 grounding을 끄고 비교하거나 MaskCLIP 방식을 따로 구현한다.
- **극좌표 히트맵 행동 공간**(교수님 피드백): 각도 7 × 거리 3 격자 + 제자리 회전 6 + STOP으로 선택지를 늘린다. 선택지가 텍스트라 모델 구조는 그대로다. LiDAR 점유로 막힌 칸을 softmax에서 제외할 수 있다.
- **소형 VLM 기준선**: 2~3B VLM을 같은 데이터로 미세조정한 결과.

## 8. 확인하지 못한 것

- NaVILA가 연속된 같은 행동을 최대 3개까지 합쳐 라벨을 만들었다는 것은 논문 기억이다. 4.2의 주석 집계로 확인한다.
- InternData-N1의 카메라 설정과 NaVILA VLN-CE 평가의 카메라 설정이 같은지 확인하지 않았다. 교사 라벨 품질에 영향을 줄 수 있다.
- ModernBERT는 층 3개마다 전역 attention이고 나머지는 국소 attention(창 128토큰)인 것으로 기억한다. 선택지 토큰과 현재 프레임 토큰이 멀리 떨어져 있으니, 배치 순서를 바꾼 실험(선택지를 열의 끝에 두기)을 한 번 해볼 가치가 있다.
- DINOv2의 Hugging Face 이름(`facebook/dinov2-with-registers-base`)과 그 클래스를 지원하는 transformers 버전은 기억 기준이다. 서버 환경에서 불러와 확인한다.
- 파라미터 수(약 0.6B)와 속도는 추정이다. 구현 후 실제 값을 잰다.
- 이 설계는 실행해 본 적이 없다. 성능 목표는 "NaVILA와 동급"이 아니라 "성능-속도 곡선 위의 한 점"이다.
