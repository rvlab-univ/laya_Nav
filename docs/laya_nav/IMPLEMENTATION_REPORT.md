# LayaNav 구현 보고서: System 1과 System 2를 하나의 모델로

- 저장소: `rvlab-univ/laya_Nav` (`main`), 기반 코드: InternNav `7a5c624`
- 대상 모델: InternVLA-N1 **DualVLN** (System 2 = Qwen2.5-VL-7B, System 1 = `nextdit_async`)
- 작성 시점: 2026-10-02, 커밋 `275dbe4` 기준

---

## 1. 요약

| 항목 | 내용 |
|---|---|
| 목표 | DualVLN의 System 2(7B VLM)와 System 1(DiT 경로 생성기)을 **Laya 구조를 뼈대로 하는 하나의 모델**로 합치고, **계산량을 실제로 줄인다** |
| 결과물 | `LayaNav`: 가중치 하나, 체크포인트 하나, 호출 두 가지(`forward`=판단, `plan`=경로) |
| 핵심 변경 | System 1의 별도 이미지 인코더와 **10스텝 × CFG 2배 × 32샘플 DiT 샘플링**을 없애고, 판단 모델이 이미 계산한 특징을 읽는 **10M 파라미터 경로 디코더(1회 실행)**로 교체 |
| 경로 계산 1회 (3090, bf16) | **202 ms → 10.8 ms (약 19배)** |
| step당 계산 | Laya-S2 + DualVLN System 1 **55~60 ms** → LayaNav **7.5~12.4 ms (약 5~7배)** |
| 파라미터 | Laya-S2 436M + System 1 약 91M = **약 527M → 446.5M** |
| 학습 | 7B teacher 없이 가능. C1(경로 헤드만) → C2(전체) |
| 검증 상태 | 단위 테스트 13개 통과, 3090에서 C1 → C2 → 저장 → 평가 에이전트 로드까지 실행 확인. **실제 데이터 학습과 Habitat 평가는 아직 수행 전** |

---

## 2. 배경: 어디서 계산이 드는가

Laya-S2(가벼운 System 2)를 붙인 뒤 각 부품의 시간을 RTX 3090(bf16, 배치 1)에서 측정했다. 가중치는 무작위지만 계산 시간은 실제와 같다.

| 부품 | 시간 | 비고 |
|---|---|---|
| Laya-S2 판단 1회 | 40.3 ms | 이미지 10장 인코딩 13.6 ms + mmBERT·헤드 약 27 ms |
| DualVLN System 1 `generate_traj` 1회 | **201.8 ms** | 샘플 수 32 / 8 / 1에서 197~202 ms로 거의 같음 |
| ├ System 1 이미지 처리 | 7.9 ms | Depth-Anything ViT-S(2장) + 메모리 인코더 + QFormer, **System 1 시간의 4%** |
| └ DiT 1스텝 (배치 64) | 24.4 ms | GPU가 실제로 일한 시간 16.8 ms, **스텝당 커널 약 1,091개** |
| DiT 1스텝, `torch.compile`(CUDA graph) 적용 | 4.3 ms | 참고용 (실행 최적화는 이번 범위에서 제외) |

**측정에서 얻은 결론**

1. System 2를 가볍게 바꾼 뒤에는 **System 1이 step당 계산의 대부분**을 차지한다. 4 step마다 재계획하므로 step당 약 50 ms다.
2. System 1 시간의 대부분은 **DiT 반복 샘플링**이다. 이미지 처리는 4%에 불과하다. 그래서 "이미지 인코더만 공유"하는 B안으로는 계산이 거의 줄지 않는다.
3. DiT는 32개 경로를 뽑아 놓고 **평균 하나만 쓴다**(`traj_to_actions`). 다양한 후보를 내는 diffusion의 장점이 마지막에 사라진다. 그렇다면 **평균 경로를 한 번에 회귀하는 헤드**로 바꿔도 동작은 거의 같고, 계산은 크게 줄어든다.

→ 이 측정에 따라 **C안(단일 모델, 경로 생성 방식 교체)**으로 진행했다.

---

## 3. 설계

### 3.1 구조

```
LayaNav(                                         # 446.5M
  ── 공유 인코더 (Laya-S2와 동일) ──
  (vision): SiglipVisionTransformer      92.9M   # SigLIP2-B/16 @224, 패치 14×14 = 196개
  (vis_proj): Sequential                         # 768-d로 투영
  (text): ModernBertModel (mmBERT-base)  306.9M  # 지시문 + 이미지 토큰 융합 (임베딩 표 197M 포함)
  (seg_emb, pos_*, hist_slot_emb, action_*)
  (head): TransformerEncoder 2층                 # Laya의 head
  ── 판단 헤드 (System 2 역할, Laya-S2와 동일) ──
  (scorer)        → [STOP, ←, →] + 패치 196개를 softmax 하나로
  (offset_head)   → 패치 안의 정확한 위치
  (act_head)      → 직접 결정 / escalate (Laya의 확신도 특징 4개)
  (latent_queries, goal_xy_emb, latent_decoder, latent_out)  → 목표 조건부 latent 4×768
  ── 경로 헤드 (System 1 역할, 새로 추가) ──      10.1M
  (traj_mem_proj, traj_latent_proj)              # 768 → 384
  (traj_seg, traj_patch_pos)                     # 메모리 / 결정 시점 프레임 / 현재 프레임 구분, 패치 위치
  (traj_queries): 32 × 384                       # 경로 점 하나당 쿼리 하나
  (traj_decoder): TransformerDecoder 4층, d=384
  (traj_out): LayerNorm + Linear → (32, 3)
)
```

### 3.2 호출 두 가지

| 호출 | 언제 | 입력 | 출력 |
|---|---|---|---|
| `forward(...)` | System 2 시점 (판단 필요 시) | 지시문, 과거 8장, 현재 정면, 현재 내려다보기 | 행동 또는 목표 픽셀, `goal_token`, `latent`, `down_feat` |
| `plan_memory(out)` | 목표가 정해진 직후 1회 | `goal_token`, `latent` | **계획 메모리** `[1+4, 384]` (다음 판단까지 보관) |
| `plan(memory, goal_feat, cur_feat)` | System 1 시점 (재계획마다) | 계획 메모리, 결정 시점 내려다보기 특징(보관), **현재 내려다보기 1장만 새로 인코딩** | 경로 `[32, 3]` |

- **두 장면 비교 기능 유지:** 원래 System 1이 하던 "목표를 정한 시점의 장면 vs 지금 장면" 비교는 그대로 유지한다. 별도 인코더 대신 **이미 계산한 공유 인코더 특징**(`down_feat`)을 재사용한다.
- **기존 파이프라인 재사용:** 경로 형식은 DualVLN과 같다. 0.1 m 간격 32스텝의 `(dx, dy, dyaw)`이고, `dx, dy`는 4배 스케일이다. 그래서 `traj_to_actions`와 평가 루프를 바꾸지 않고 쓴다.

### 3.3 왜 회귀 헤드인가

- **기존과 같은 결과를 내는 방식:** 기존 파이프라인은 경로 샘플 32개의 **평균**을 행동으로 바꾼다. 회귀 헤드는 이 평균에 해당하는 조건부 기댓값을 직접 학습한다.
- **목표까지의 거리:** 경로 정답에는 목표 도착 후 변화량 0이 채워져 있다. 그래서 "목표까지 남은 거리"도 같이 학습된다.
- **갈림길 대비:** 경로가 여러 갈래인 상황에 대비한 대안도 남겨 두었다. Laya 방식으로 경로 후보 K개를 선택지로 두고 `scorer`로 고르는 방식이다(7절).

---

## 4. 구현 내역

### 4.1 파일별 변경

| 파일 | 변경 |
|---|---|
| `internnav/model/basemodel/laya_s2/laya_nav.py` (신규) | `LayaNav`, `LayaNavConfig`, `plan_memory` / `encode_frame` / `plan`, 학습용 `forward(traj_pixels, traj_mask)`, `from_laya_s2` / `load_any`, `traj_loss` |
| `internnav/model/basemodel/laya_s2/laya_s2.py` | `forward`가 `goal_token`, `down_feat`도 반환. `config_class`로 하위 클래스 로딩 지원. **기존 동작은 변화 없음** |
| `internnav/dataset/laya_s2_dataset.py` | 목표 샘플에 `goal_len`과 `poses`(에피소드 단위 공유 참조, 복사 없음)를 추가. `with_traj=True`일 때 경로 학습 쌍을 생성(`trajectory_frame_ids`, `trajectory_target`) |
| `scripts/train/laya_s2/train_laya_nav.py` (신규) | C1/C2 학습. Laya-S2 학습 스크립트의 공용 함수(scene split, 스케줄)를 재사용 |
| `internnav/model/basemodel/laya_s2/agent.py` | 에이전트 API를 `decide` / `start_goal` / `plan`으로 통일하고 `LayaNavAgent`, `load_agent` 추가 |
| `internnav/habitat_extensions/vln/laya_s2_eval.py` | 평가 루프가 System 1 종류와 무관하게 `agent.plan`을 호출 |
| `internnav/habitat_extensions/vln/habitat_vln_evaluator.py` | `mode='laya_nav'` 등록 |
| `scripts/eval/configs/habitat_laya_nav_cfg.py` (신규) | LayaNav 평가 설정 (`LAYA_NAV` 환경변수로 체크포인트 지정) |
| `scripts/eval/bash/compare_laya_s2.sh` | LayaNav 체크포인트가 있으면 3개 설정을 함께 평가하고 비교 |
| `tests/laya_s2/test_laya_nav.py` (신규) | 4개 테스트 (5절) |

### 4.2 경로 학습 데이터

원래 System 1 학습(`NavPixelGoalDataset`, `pixel_goal_only`)과 **같은 규칙**으로 만든다.

- **샘플 구성:** 목표 샘플(시작 프레임 `start`, 목표까지 `goal_len` 프레임)마다 출발 지점 후보 `trajectory_frame_ids(goal_len)`(2프레임 간격, 최대 12개) 중 하나를 고른다. 학습 시에는 무작위, 검증 시에는 가운데 것을 쓴다.
- **입력:** 그 지점의 **내려다보기 프레임**
- **정답:** 그 지점에서 목표까지 남은 경로. 원본 함수 `get_trajectory_relative_to_frame`(카메라 pitch 보정) → `interpolate_and_resample_trajectory`(0.1 m 재표본, 4배 스케일) → `clip_or_pad(32)`
- **결정 시점 프레임:** 판단 입력의 `down_pixels`를 그대로 쓴다. 원래 System 1의 `pix_goal_image`와 같은 프레임이다.
- **방어 처리:** 에피소드 범위를 벗어나는 출발 지점은 제외한다(`_traj_start_ids`).

### 4.3 손실

| 손실 | 정의 | 단계 |
|---|---|---|
| `l_traj` | 스텝별 `(dx, dy, dyaw)` Huber | C1, C2 |
| 누적 경로 (`ade`) | 누적 좌표(m)의 평균 L2 거리. 작은 오차가 쌓여 경로가 틀어지는 것을 막음 | C1, C2 |
| 판단 손실 | Laya-S2와 동일 (log + spherical score, 위치 보정, escalate) | C2 |
| latent 따라 하기 | 7B `cond_projector` 출력 모방 (`--teacher_latents`를 줄 때만) | C2 (선택) |

보고 지표: `ade`(평균 변위 오차, m), `fde`(최종 변위 오차, m). 검증 지표는 해당 샘플 수로 가중 평균한다.

### 4.4 학습 단계

| 단계 | 시작점 | 학습 대상 | 데이터 |
|---|---|---|---|
| **C1** | 학습된 Laya-S2 체크포인트 | **경로 헤드만** (10.1M), 나머지는 고정 + eval 모드(dropout 끔) | 목표 샘플만 |
| **C2** | C1 결과 | 전체 | 전체 샘플 (목표 / 회전 / 정지) |

- **DDP 대응:** 학습 시 경로 헤드를 모델 `forward` 안에서 실행한다. 경로 샘플이 없는 배치에서도 학습 대상 파라미터에 0 손실을 연결해서, C1의 `backward`가 실패하거나 DDP가 멈추는 일을 막는다.
- **7B teacher가 필요 없다.** 이후 rxr이나 scalevln을 추가할 때 7B latent 추출(r2r만으로도 수 시간) 없이 바로 학습할 수 있다.

### 4.5 평가 연동

에이전트 세 종류가 같은 API를 쓴다.

```
decide(지시문, 과거, 현재, 내려다보기)  → 행동 또는 목표           [timers.s2]
start_goal(결정, 내려다보기, 깊이)       → System 1 조건 고정
plan(현재 내려다보기, 깊이)              → 경로 [N, 32, 3]          [timers.s1]
```

| 모드 | System 2 | System 1 |
|---|---|---|
| `dual_system` (기준선) | Qwen2.5-VL-7B | DualVLN DiT |
| `laya_s2` | Laya-S2 | DualVLN DiT (단독 추출, 원본과 결과 동일 검증) |
| `laya_nav` | LayaNav `forward` | LayaNav `plan` |

- **공정한 비교:** 평가 루프는 기준선과 **같은 제어 흐름**을 쓴다. 내려다보기 촬영, 4 step마다 재계획, 8 step 후 재판단, 같은 에피소드 부분집합(`EVAL_EPISODES`)이 모두 같다. 그래서 세 모드의 차이는 모델뿐이다.
- **재계획 주기:** 경로 계산이 싸져서 매 step 재계획도 가능하다. 다만 비교의 공정성을 위해 기본값은 기준선과 같은 4 step으로 두었다.

---

## 5. 검증

### 5.1 단위 테스트 (`tests/laya_s2`, 13개 모두 통과, transformers 4.51.0 / diffusers 0.32.2)

| 테스트 | 확인 내용 |
|---|---|
| `test_forward_plan_and_c1_gradients` | 학습용 `forward`가 경로 샘플만 골라 계획함. **C1에서 기울기가 `traj_*` 파라미터에만** 흐름. 추론 경로(`forward` → `plan_memory` → `plan`)가 학습용 `forward`와 같은 경로를 냄 |
| `test_init_from_laya_s2_and_save_load` | Laya-S2 체크포인트에서 시작하면 판단 출력이 **원래 Laya-S2와 동일**. LayaNav 저장 후 다시 불러도 경로가 동일 |
| `test_trajectory_target_straight_line` | 실제 데이터 변환의 역으로 만든 카메라 pose(직진 0.25 m/프레임)에서 정답 경로가 **2.0 m 직진, 첫 스텝 0.4(=0.1 m×4)**. 중간 출발 시 남은 거리 1.0 m |
| `test_dataset_with_traj_and_agent` | 데이터셋이 경로 쌍을 만들고, pose는 복사 없이 공유됨. 에피소드 범위 밖 출발 지점을 제외. `LayaNavAgent`의 `decide` → `start_goal` → `plan` 동작 |
| 기존 9개 | Laya-S2, 단독 System 1(원본과 비트 단위 동일), 데이터 로딩, 에피소드 부분집합 등. **변경 후에도 모두 통과** |

### 5.2 학습 스크립트 실행 (3090, 작은 모델 + 실제 형식의 가짜 데이터)

LeRobot 형식(parquet + `episodes.jsonl`, pose는 4×4 중첩 리스트)의 씬 2개와 작은 Laya-S2 체크포인트로 실행했다.

```
stage c1 | params 16.9M (trajectory head 0.1M, trainable 0.1M) | train 14 ...
[ep 0 step 2/6] ... l_traj=0.0736 ade=1.4364 fde=3.0481
[ep 1 step 6/6] ... l_traj=0.0477 ade=0.6402 fde=1.3455      ← C1: 경로 헤드만 학습, 오차 감소
stage c2 | params 16.9M (trajectory head 0.1M, trainable 16.9M) | train 24 ...
[val step 3] ... ade=1.1457 fde=2.8011 ...                    ← scene 단위 검증 경로 동작
```
- **C1 → C2 → 체크포인트 저장 → `load_agent(mode='laya_nav')` → `decide`/`plan` → `traj_to_actions`**까지 이어지는 것을 확인했다.
- 숫자 자체는 의미가 없다. 무작위 이미지로 몇 step만 돌린 결과다.

### 5.3 실측 계산량 (실제 크기, RTX 3090, bf16, 배치 1)

| | Laya-S2 + DualVLN System 1 | **LayaNav** |
|---|---|---|
| 판단 1회 | 40.3 ms | 38.8 ms |
| 경로 계산 1회 | 201.8 ms | **10.8 ms** (프레임 인코딩 6.6 ms + 경로 헤드 3.3 ms) |
| step당 (판단 4 step마다, 경로 4 step마다) | 60.5 ms | **12.4 ms** (4.9배) |
| step당 (판단 8 step마다, 경로 4 step마다) | 55.5 ms | **7.5 ms** (7.4배) |
| step당 (판단 8 step마다, **경로 매 step**) | 206.8 ms | **15.6 ms** |
| 파라미터 | 약 527M (436.3M + 약 91M) | 446.5M |

- **H200에서는 차이가 더 클 수 있다.** DiT는 커널 실행 오버헤드에 묶여 있어서, GPU가 빨라져도 CPU 쪽 오버헤드는 그대로 남기 때문이다.
- **7B 기준선 시간은 아직 없다.** Habitat 비교 평가에서 측정한다(`progress.json`의 `s2_time`, `s1_time`).

---

## 6. 구현 과정에서 발견하고 고친 것

| 문제 | 원인 | 조치 |
|---|---|---|
| 단독 System 1과 원본의 경로가 0.09 → 0.31만큼 다름 | 원본은 이미지 정규화 상수(`_resnet_mean/std`)가 **정확한 fp32**인데, 단독 버전은 bf16 반올림값(0.484375)을 씀 | 상수에서 다시 생성. 가중치 609개 / 단계 5개 / 최종 경로 **모두 차이 0** 확인 |
| 검증 `acc_goal`, `latent_cos` 과소평가 | 해당 샘플이 없는 배치를 0으로 평균 | 샘플 수 가중 평균 |
| 검증이 학습과 같은 집에서 이뤄짐 | 에피소드 단위 분할 | **집(scene) 단위** 분할이 기본값 (r2r 61개 중 4개를 따로 둠) |
| 에피소드 밖 출발 지점 | 목표 프레임이 에피소드 끝을 넘는 경우 | 출발 지점을 에피소드 범위로 제한 |
| C1에서 경로 샘플 없는 배치 | 손실이 그래프와 끊겨 `backward` 실패 또는 DDP 정지 가능 | 학습 대상 파라미터에 0 손실 연결 |

---

## 7. 한계와 위험

| 항목 | 내용 | 대응 |
|---|---|---|
| **실제 데이터 미학습** | 경로 헤드는 아직 r2r로 학습하지 않았다 | 서버에서 C1부터 실행 (8절) |
| **Habitat 미실행** | `laya_nav` 평가 루프는 이 PC에 Habitat이 없어 정적 검사와 단위 테스트까지만 했다 | 서버 첫 실행 시 확인 |
| 갈림길(다중 경로) | 회귀는 평균을 낸다. 다만 기존도 32샘플 평균이라 손해는 아님 | Laya 방식 **경로 후보 K개 + scorer 선택**(정답에 가장 가까운 후보만 학습)으로 확장 가능 |
| 경로 품질 | DiT보다 정밀도가 낮을 수 있다 | 3개 설정 비교로 **경로 헤드 교체의 영향만** 따로 측정 |
| 판단 성능 흔들림 (C2) | 공유 인코더가 경로 손실의 영향을 받는다 | C1 → C2 전후 `acc_*` 비교, 필요시 경로 손실 가중치 조정 |
| 다중 GPU | DDP 경로는 단일 GPU에서만 실행해 봤다 | 현재 서버는 GPU 1장 |
| 일반화 | r2r 61개 집만 학습 | rxr / scalevln 추가 (7B 없이 가능) |

---

## 8. 서버 실행 순서

현재 Laya-S2 학습(`train_laya_s2.sh`)이 끝난 뒤:

```bash
cd /home1/irteam/laya_Nav && conda activate laya && git pull
export CC=$(which gcc)

# C1: 경로 헤드만 (Laya-S2 고정)
setsid nohup python scripts/train/laya_s2/train_laya_nav.py --stage c1 \
    --init_from checkpoints/laya_s2/last --vln_dataset_use r2r_125cm_0_30 \
    --output_dir checkpoints/laya_nav_c1 --batch_size 64 --num_workers 16 \
    > logs/train_laya_nav_c1.log 2>&1 < /dev/null &

# C2: 전체 (C1 결과에서)
setsid nohup python scripts/train/laya_s2/train_laya_nav.py --stage c2 \
    --init_from checkpoints/laya_nav_c1/last --vln_dataset_use r2r_125cm_0_30 \
    --output_dir checkpoints/laya_nav_c2 --batch_size 64 --num_workers 16 \
    > logs/train_laya_nav_c2.log 2>&1 < /dev/null &

# 비교 평가: DualVLN / Laya-S2 + DiT / LayaNav, 같은 300개 에피소드
EVAL_EPISODES=300 bash scripts/eval/bash/compare_laya_s2.sh
```

로그에서 볼 지표:
- **C1, C2 공통:** `[val step …]`의 `ade`, `fde`(m). 처음 보는 집 4개에서 측정한 값이다.
- **C2 추가:** `acc_action`, `acc_goal`이 C1(=Laya-S2) 대비 떨어지지 않는지

---

## 9. 다음 단계

1. 서버에서 C1과 C2를 학습하고, 3개 설정을 비교 평가한다(성공률, SPL, step당 시간).
2. 결과에 따라:
   - **경로 품질이 부족하면:** 경로 후보 K개 + scorer 선택(Laya 구조 확장), 또는 1~2스텝 flow 헤드
   - **일반화가 부족하면:** rxr / scalevln 추가 (7B 추출 없이)
   - **판단 성능이 부족하면:** 7B를 escalate용으로만 쓰는 하이브리드

---

## 부록 A. 커밋 이력 (`7a5c624` 이후)

| 커밋 | 내용 |
|---|---|
| `b8f506d` | Laya-S2: 비자기회귀 경량 System 2 (모델, 데이터, latent 추출, 학습, 단독 System 1, 평가 모드) |
| `5cb35cf` | 서버 설치 스크립트, 영상 디코더 import 선택화 |
| `4170ac0` | 목표 좌표 순서 확인 도구, latent 추출 소량 옵션 |
| `8fd391c` | 단일 GPU 지원, 평가 에피소드 부분집합 |
| `a288cf0` | flash-attn 빌드된 wheel 설치 |
| `bf18727` | 좌표 확인 도구: 값 범위, 묶음 그림 |
| `a0862ea` | Triton 실행용 gcc 설치 |
| `d5edaf5`, `86deea1`, `f4e5456` | 단독 System 1의 정규화 상수 문제 진단과 수정, 가중치·단계별 비교 검증 |
| `889b86e` | 검증 지표 샘플 수 가중 평균 |
| `c86b53b`, `5070769` | 설치 스크립트: HF 로그인 사전 확인, 전체 항목 점검 |
| `ed156cf` | 검증을 집 단위로 분할 |
| `275dbe4` | **LayaNav**: 단일 모델 (경로 헤드, C1/C2 학습, `laya_nav` 평가 모드) |

## 부록 B. 측정 환경

- RTX 3090 24 GB, torch 2.12 (cu130), transformers 4.51.0, diffusers 0.32.2
- 시간: CUDA 동기화 후 20~30회 평균, 워밍업 5회, `torch.autocast(bfloat16)`, 배치 1
- 인코더: `jhu-clsp/mmBERT-base`, `google/siglip2-base-patch16-224` (실제 크기, 무작위 head)
- DualVLN System 1: 실제 구조(`nextdit_async`), 무작위 가중치 (계산 시간 측정용)
