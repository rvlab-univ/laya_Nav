# LayaNav 구현 보고서: System 1과 System 2를 하나의 모델로

- 저장소: `rvlab-univ/laya_Nav` (`main`), 기반 코드: InternNav `7a5c624`
- 대상 모델: InternVLA-N1 **DualVLN** (System 2 = Qwen2.5-VL-7B, System 1 = `nextdit_async`)
- 작성 시점: 2026-10-02, 커밋 `275dbe4` 기준
- 갱신: 2026-10-06. System 1 / System 2 구분 표시, C2 학습 결과와 현재 개선점·한계(11절) 추가

---

## 1. 요약

| 항목 | 내용 |
|---|---|
| 목표 | DualVLN의 System 2(7B VLM)와 System 1(DiT 경로 생성기)을 **Laya 구조를 뼈대로 하는 하나의 모델**로 합치고, **계산량을 실제로 줄인다** |
| 결과물 | `LayaNav`: 가중치 하나, 체크포인트 하나, 호출 두 가지(`forward`=판단, `plan`=경로) |
| 핵심 변경 | System 1의 별도 이미지 인코더와 **10스텝 × CFG 2배 × 32샘플 DiT 샘플링**을 없애고, 판단 모델이 이미 계산한 특징을 읽는 **10M 파라미터 경로 디코더(1회 실행)**로 교체 |
| 경로 계산 1회 (3090, bf16) | **202 ms → 10.8 ms (약 19배)** |
| step당 계산 | Laya-S2 + DualVLN System 1 **55~60 ms** → LayaNav **7.5~12.4 ms (약 5~7배)** |
| 파라미터 | Laya-S2 436M + System 1 약 91M = **약 527M → 439.4M** (사용하지 않던 SigLIP 풀링 헤드 7.1M 제거 포함) |
| 학습 | 7B teacher 없이 가능. C1(경로 헤드만) → C2(전체) |
| 검증 상태 | C2 학습 완료 (`laya_nav_mix_c2`, r2r + rxr + scalevln, 96,318 step, 2026-10-05). 오프라인 검증 결과는 11절. **Habitat 폐루프 평가(SR, SPL)는 아직 수행 전** |
| 현재 병목 | **System 2(목표 선택)**. System 1 경로 헤드는 정답 목표 기준 ADE 0.09 m로 동작하지만, 모델이 직접 고른 목표로는 0.37 m (11절) |

### 1.1 System 1 / System 2 구분

LayaNav는 가중치 하나지만 역할은 DualVLN과 같이 둘로 나뉜다. 이 문서의 모든 변경과 결과에는 어느 쪽인지 표시했다.

| 구분 | LayaNav 모듈 | 호출 시점 | DualVLN에서 대응하는 부분 |
|---|---|---|---|
| **System 2** (판단: 행동 또는 목표 픽셀) | `vis_proj`, `text`(mmBERT), `seg_emb`·`pos_*`·`action_*`, `head`, `scorer`, `offset_head`, `act_head`, `latent_*`, `goal_xy_emb` (= Laya-S2) | `forward`, 재판단마다 | Qwen2.5-VL-7B + `cond_projector` |
| **System 1** (경로: 32스텝 궤적) | `traj_*` 전부 (`traj_mem_proj`, `traj_latent_proj`, `traj_goal_xy`, `traj_goal_mark`, `traj_seg`, `traj_patch_pos`, `traj_fuse`, `traj_queries`, `traj_decoder`, `traj_out`) | `plan`, 재계획마다 | Depth-Anything ViT-S + MemoryEncoder + QFormer + NextDiT |
| **공유** | `vision` (SigLIP2) | 둘 다 (System 2: 10장, System 1: 현재 내려다보기 1장) | 없음 (DualVLN은 System마다 인코더가 따로 있음) |

**변경 이력을 System별로 보면**

| 시점 · 커밋 | 변경 | 대상 |
|---|---|---|
| 10/02 `b8f506d` | 7B VLM을 Laya-S2(비자기회귀 판단 모델)로 교체 | **System 2** |
| 10/02 `275dbe4` | DiT 경로 생성기를 경로 헤드(회귀, 1회 실행)로 교체해 단일 모델(LayaNav) 구성 | **System 1** |
| 10/02 `fe9dbf2` | 사용하지 않던 SigLIP 풀링 헤드 제거 | 공유 인코더 |
| 10/03 `31fb3a8` | 경로 헤드 v2 (`traj_fuse`, 목표 표시, 출발 지점 4개) | **System 1** |
| 10/03 `31fb3a8` | 큰 데이터셋 로딩, 버그 수정, 검증 상한 | 데이터 (공통) |
| 10/05 (서버) | C2: 전체 학습 (r2r + rxr + scalevln) | System 1 + System 2 + 공유 |

---

## 2. 배경: 어디서 계산이 드는가 (System 1을 바꾼 이유)

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
LayaNav(                                         # 439.4M
  ── 공유 인코더 (Laya-S2와 동일) ──
  (vision): SiglipVisionTransformer      85.8M   # SigLIP2-B/16 @224, 패치 14×14 = 196개 (풀링 헤드 제거)
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

### 3.3 왜 회귀 헤드인가 (System 1)

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

### 4.2 경로 학습 데이터 (System 1)

원래 System 1 학습(`NavPixelGoalDataset`, `pixel_goal_only`)과 **같은 규칙**으로 만든다.

- **샘플 구성:** 목표 샘플(시작 프레임 `start`, 목표까지 `goal_len` 프레임)마다 출발 지점 후보 `trajectory_frame_ids(goal_len)`(2프레임 간격, 최대 12개) 중 하나를 고른다. 학습 시에는 무작위, 검증 시에는 가운데 것을 쓴다.
- **입력:** 그 지점의 **내려다보기 프레임**
- **정답:** 그 지점에서 목표까지 남은 경로. 원본 함수 `get_trajectory_relative_to_frame`(카메라 pitch 보정) → `interpolate_and_resample_trajectory`(0.1 m 재표본, 4배 스케일) → `clip_or_pad(32)`
- **결정 시점 프레임:** 판단 입력의 `down_pixels`를 그대로 쓴다. 원래 System 1의 `pix_goal_image`와 같은 프레임이다.
- **방어 처리:** 에피소드 범위를 벗어나는 출발 지점은 제외한다(`_traj_start_ids`).

### 4.3 손실

| 손실 | 정의 | 대상 | 단계 |
|---|---|---|---|
| `l_traj` | 스텝별 `(dx, dy, dyaw)` Huber | System 1 | C1, C2 |
| 누적 경로 (`ade`) | 누적 좌표(m)의 평균 L2 거리. 작은 오차가 쌓여 경로가 틀어지는 것을 막음 | System 1 | C1, C2 |
| 판단 손실 (`l_dec`, `l_off`, `l_esc`) | Laya-S2와 동일 (log + spherical score, 위치 보정, escalate) | System 2 | C2 |
| latent 따라 하기 (`l_lat`) | 7B `cond_projector` 출력 모방 (`--teacher_latents`를 줄 때만) | System 2 | C2 (선택) |

C2에서는 공유 인코더(SigLIP2)가 System 1과 System 2 손실을 함께 받는다.

보고 지표: `ade`(평균 변위 오차, m), `fde`(최종 변위 오차, m). 검증 지표는 해당 샘플 수로 가중 평균한다.

### 4.4 학습 단계

| 단계 | 시작점 | 학습 대상 | 데이터 |
|---|---|---|---|
| **C1** | 학습된 Laya-S2 체크포인트 | **System 1만** (경로 헤드), System 2와 공유 인코더는 고정 + eval 모드(dropout 끔) | 목표 샘플만 |
| **C2** | C1 결과 | **System 1 + System 2 + 공유 인코더 전체** | 전체 샘플 (목표 / 회전 / 정지) |

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
| 파라미터 | 약 527M (436.3M + 약 91M) | 439.4M |

- **H200에서는 차이가 더 클 수 있다.** DiT는 커널 실행 오버헤드에 묶여 있어서, GPU가 빨라져도 CPU 쪽 오버헤드는 그대로 남기 때문이다.
- **7B 기준선 시간은 아직 없다.** Habitat 비교 평가에서 측정한다(`progress.json`의 `s2_time`, `s1_time`).

---

## 6. 구현 과정에서 발견하고 고친 것

| 문제 | 원인 | 조치 | 대상 |
|---|---|---|---|
| 단독 System 1과 원본의 경로가 0.09 → 0.31만큼 다름 | 원본은 이미지 정규화 상수(`_resnet_mean/std`)가 **정확한 fp32**인데, 단독 버전은 bf16 반올림값(0.484375)을 씀 | 상수에서 다시 생성. 가중치 609개 / 단계 5개 / 최종 경로 **모두 차이 0** 확인 | DualVLN System 1 (비교 기준선) |
| 검증 `acc_goal`, `latent_cos` 과소평가 | 해당 샘플이 없는 배치를 0으로 평균 | 샘플 수 가중 평균 | 평가 지표 |
| 검증이 학습과 같은 집에서 이뤄짐 | 에피소드 단위 분할 | **집(scene) 단위** 분할이 기본값 (r2r 61개 중 4개를 따로 둠) | 평가 (검증 분할) |
| 에피소드 밖 출발 지점 | 목표 프레임이 에피소드 끝을 넘는 경우 | 출발 지점을 에피소드 범위로 제한 | System 1 (경로 데이터) |
| C1에서 경로 샘플 없는 배치 | 손실이 그래프와 끊겨 `backward` 실패 또는 DDP 정지 가능 | 학습 대상 파라미터에 0 손실 연결 | System 1 (C1 학습) |
| SigLIP 풀링 헤드(`vision.head`, 7.1M)가 매 인코딩마다 실행되고 결과는 버려짐 | 이미지 전체 요약 벡터(`pooler_output`)용 부속인데, 우리는 패치 특징만 사용 | 헤드 제거(`use_head=False`). 패치 특징은 원본과 **차이 0**, 미사용 파라미터 0개, 인코딩 1회당 약 0.5~0.6 ms 절약. 이전 체크포인트는 해당 키를 무시하고 불러옴. **Laya 구성(`head`, `scorer`, `act_head`)은 변화 없음** | 공유 인코더 |

---

## 7. 한계와 위험 (작성 당시 10/02 기준)

> **현재 상태는 11절을 보라.** 아래 표는 실제 데이터로 학습하기 전에 예상한 위험이다. "실제 데이터 미학습"과 "일반화(r2r만)"는 C2로 해소됐고, "경로 품질"은 정답 목표 기준으로 문제가 없음이 확인됐다.

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

> 2026-10-03 이후 C1/C2는 `scripts/train/laya_s2/train_laya_nav.sh`로 실행한다(10절). 아래는 첫 C1 당시의 명령이다.

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

## 9. 다음 단계 (작성 당시 10/02 기준, 현재는 11.4절)

1. 서버에서 C1과 C2를 학습하고, 3개 설정을 비교 평가한다(성공률, SPL, step당 시간).
2. 결과에 따라:
   - **경로 품질이 부족하면:** 경로 후보 K개 + scorer 선택(Laya 구조 확장), 또는 1~2스텝 flow 헤드
   - **일반화가 부족하면:** rxr / scalevln 추가 (7B 추출 없이)
   - **판단 성능이 부족하면:** 7B를 escalate용으로만 쓰는 하이브리드

---

## 10. 후속 변경 (2026-10-03): System 1 경로 헤드 보강과 큰 데이터셋

> **대상: System 1** (경로 헤드)과 데이터 로딩. System 2(판단 모델)는 바꾸지 않았다.

서버에서 돌린 첫 C1(r2r) 결과가 좋지 않았다. 정답 경로 생성 규칙은 원본 System 1과 같다(프레임 정렬, pose 구간, pitch 보정, 재표본 모두 확인). 차이는 학습 구성에 있었다.

| | 원본 System 1 | 첫 C1 | 변경 |
|---|---|---|---|
| 목표 샘플당 출발 지점 | 최대 12개 전부 | 무작위 1개 | `--traj_starts` (기본 4) |
| 두 장면 비교 | 두 프레임 패치 전체에 self-attention (`memory_encoder`) | 경로 쿼리가 각 프레임을 따로 읽을 뿐, 두 프레임 패치가 서로를 보지 못함 | 메모리 전체에 self-attention 2층 (`traj_fuse_layers`) |
| 목표 위치 | 7B hidden state에 들어 있음 | 결정 모델의 은닉 상태에 섞여 있음 | 목표 프레임의 목표 주변 패치에 표시 + 목표 좌표 토큰 (`traj_goal_mark`) |
| 이미지 인코더 | 함께 학습 | C1에서는 고정 | 그대로 (C2에서 학습) |

- **비용 (5060 Ti 실측)**: 경로 헤드 10.1M → 13.7M, 경로 계산 1회 +0.9 ms, step당 +0.2 ms. C1 학습 step은 출발 지점 4개 때문에 약 +40%.
- **호환**: 옛 체크포인트는 설정에 새 필드가 없으면 옛 헤드로 그대로 로드된다. 첫 C1과 같은 설정은 `--traj_fuse_layers 0 --traj_goal_mark 0 --traj_starts 1`이다.
- **진단 스크립트 (`scripts/train/laya_s2/eval_traj.py`)**: 처음 보는 집의 모든 출발 지점에서 ade, fde, 첫 0.5 m 방향 오차를 잰다. "평균 경로"와 "정지" 기준선을 함께 내고, 출발 위치별·남은 거리별로 나눠 보여 준다. 평균 경로 기준선보다 확실히 낫지 않으면 경로 헤드가 목표나 장면을 읽지 못하고 있다는 뜻이다. 학습 로그의 `[val step]`은 이제 출발 지점 4개 평균이라 첫 C1 값과 직접 비교할 수 없으니, 체크포인트 간 비교는 이 스크립트로 한다.

**큰 데이터셋(rxr, scalevln) 대응**

| 문제 | 조치 |
|---|---|
| 카메라 pose를 프레임마다 Python 중첩 리스트로 저장 (프레임당 약 570 B, dataloader worker마다 복사될 수 있음) | float32 배열 `[T, 4, 4]` (프레임당 64 B) |
| 해당 카메라 설정의 열이 없는 에피소드가 **이전 에피소드의 pose와 목표를 그대로 씀** (원본 코드의 버그) | 그 에피소드는 건너뛰고 경고 출력 |
| 장면을 스레드로 읽어서 샘플 순서가 실행마다 다름 → `%30` 부분집합이 매번 다르고, GPU 여러 장일 때 rank마다 목록이 다름 | 샘플 정렬 후 seed 고정 샘플링 |
| scalevln은 5% 집 분할만으로 수천 채가 검증으로 빠져 검증이 오래 걸림 | `--max_val_samples` (기본 5000, 고정 무작위 부분집합) |
| 다운로드가 아카이브를 전부 받은 뒤 풀어서 디스크가 데이터의 약 2배 필요 | 아카이브 하나씩 받고 풀고 지움. 중단되면 같은 명령을 다시 실행하면 이어서 받음 |

**서버 실행**

```bash
DATASETS="rxr scalevln" bash scripts/setup/setup_laya_s2_server.sh train_data   # 압축 기준 rxr 911 GB, scalevln 1.3 TB
bash scripts/setup/setup_laya_s2_server.sh check

# C1 -> 진단 -> C2 -> 진단. 처음 한 번은 첫 C1(checkpoints/laya_nav_c1)도 진단해서 기준으로 남긴다
setsid nohup bash scripts/train/laya_s2/train_laya_nav.sh > logs/train_laya_nav_mix.log 2>&1 < /dev/null &
#   VLN_DATASETS=r2r_125cm_0_30,rxr_125cm_0_30,scalevln_125cm_0_30%30   # 데이터셋별 비율 (기본: 세 데이터셋 전부)
#   TAG=laya_nav_mix  BATCH=64  INIT=checkpoints/laya_s2/last  VAL_RATIO=0.05
# 진단 결과: logs/<TAG>_c1_eval_traj.log, logs/<TAG>_c2_eval_traj.log, logs/laya_nav_c1_eval_traj.log

LAYA_NAV=checkpoints/laya_nav_mix_c2/last EVAL_EPISODES=300 bash scripts/eval/bash/compare_laya_s2.sh
```

- **기본 데이터셋**은 Habitat R2R 평가와 카메라가 같은 세 가지(1.25 m, 내려다보기 30°)다. 60 cm 설정은 모델에 카메라 정보 입력이 없어서 섞지 않았다.
- **C2 메모리**: 출발 지점 4개 때문에 C2에서 이미지 인코더 활성값이 약 40% 늘어난다. OOM이 나면 `BATCH=48` 또는 `--traj_starts 2`로 줄인다.
- **확인 범위**: 테스트 19개 통과. 가짜 LeRobot 데이터로 실제 크기 C1 학습, 진단 스크립트, 실행 스크립트, 다운로드 루프(가짜 Hub)를 확인했다. 실제 크기 C2는 5060 Ti(16 GB)에 올라가지 않아 작은 모델로만 확인했다. **실제 데이터에서 경로 품질이 나아지는지는 서버 결과로 확인해야 한다.**

---

## 11. 현재 모델의 개선점과 한계 (2026-10-06, C2 결과 기준)

- 대상 체크포인트: `checkpoints/laya_nav_mix_c2` (r2r + rxr + scalevln, 경로 헤드 v2, 96,318 step, 21시간)
- 수치 출처: [results/laya_nav_mix_c2.md](results/laya_nav_mix_c2.md) (서버 `logs/post_train/c2_report.md`, `laya_nav_mix_c2_eval_traj.log`)
- 검증: 학습에 쓰지 않은 집 4곳 (`PX4nDJXEHrG`, `ULsKaCPVFJR`, `VLzqgDo317F`, `rPc6DW4iMge`), 목표 샘플 8,112개, 경로 출발 지점 56,523개
- **모든 수치는 오프라인 검증이다.** Habitat 폐루프 성공률(SR, SPL)과 7B 기준선의 실제 시간은 아직 재지 않았다.

### 11.1 C2 결과 요약

| 지표 | 대상 | 값 |
|---|---|---|
| 경로 ADE / FDE, 정답 목표 사용 | System 1 | **0.09 m / 0.12 m**, 방향 오차 4.5° |
| 경로 ADE / FDE, 모델이 고른 목표 사용 | System 2 → System 1 | **0.37 m / 0.55 m**, 방향 오차 14.2° |
| 기준선: 평균 궤적 / 정지 | – | 0.52 / 0.81 m (7.5°) · 0.95 / 1.29 m |
| 목표 패치 정확도 `acc_goal` | System 2 | 최고 0.4651 (86k), 최종 0.4469 |
| 행동 정확도 `acc_action` | System 2 | 최고 0.7429 (28k), 최종 0.5543. 같은 시기 학습 쪽은 약 0.93 |
| 검증 loss | 전체 | 최저 3.3414 (28k), 최종 5.0796 (판단 손실 `l_dec` 상승이 원인) |

### 11.2 개선점 (DualVLN 대비)

**System 2: 판단**

| 개선 | 근거 |
|---|---|
| 판단 방식이 7B 자기회귀 생성 3회(내려다보기 요청 → 좌표 텍스트 → latent 재계산)에서 **forward 1회**로 바뀜 | 입력 약 2,300 → 391 토큰. 판단 1회 38.8 ms (RTX 3090). 7B 판단 시간은 미측정 |
| 파라미터 약 8.3B → **429.2M** (공유 인코더 포함) | 실제 설정으로 모델을 만들어 셈 |
| 행동 3개와 목표 패치 196개를 **softmax 하나**로 고름 | Laya `scorer` 구조 |
| 자기 판단이 틀릴지 예측하는 `act_head` (escalate) | DualVLN에는 없음. 하이브리드(애매할 때만 7B 호출)의 기반 |
| **7B teacher 없이** 데이터를 늘려 학습 가능 | C2에서 rxr, scalevln 추가 학습 완료 |

**System 1: 경로**

| 개선 | 근거 |
|---|---|
| 별도 이미지 인코더(Depth-Anything + MemoryEncoder + QFormer) 제거. 공유 SigLIP2로 현재 1장만 인코딩 | 2절, 3.2절 |
| DiT 10스텝 × 배치 64 + 32개 평균 → **디코더 1회** | 경로 계산 1회 202 → 10.8 ms (v1, RTX 3090). v2는 v1보다 약 6% 느림 |
| 경로 헤드 v2로 경로 품질 확보 | 정답 목표 기준 ADE 0.09 m. 평균 궤적(0.52 m)의 약 1/6. 검증 오차가 학습 끝(96k)까지 계속 줄어 과적합 없음 |

**모델 전체**

| 개선 | 근거 |
|---|---|
| 단일 모델, 체크포인트 하나 | 443.0M (DualVLN 약 8.4B의 약 1/19) |
| step당 계산 | Laya-S2 + DualVLN System 1의 55~60 ms → 7.5~12.4 ms (RTX 3090) |

### 11.3 한계

**System 2: 판단 (현재 가장 큰 병목)**

| 한계 | 근거 | 영향 |
|---|---|---|
| **목표 선택이 부정확** | 처음 보는 집에서 `acc_goal` 0.45 | 모델이 고른 목표로는 경로 오차가 0.09 → 0.37 m로 4배 가까이 커지고, 방향 오차(14.2°)는 평균 궤적(7.5°)보다도 나쁨. 경로 오차의 주원인 |
| **판단의 과적합** | 검증 `acc_action` 0.74 (28k) → 0.55 (최종), 검증 loss 3.34 → 5.08. 학습 쪽은 약 0.93 | 학습한 집에 맞춰지고 있음. 최종 체크포인트의 판단이 28k보다 나쁨 |
| 지시문과 화면을 잇는 능력(grounding)을 처음부터 학습 | 텍스트 인코더(mmBERT)와 이미지 인코더(SigLIP2)를 이번 학습으로 처음 연결. 7B는 인터넷 규모 사전학습으로 이미 갖고 있음 | 위 두 수치와 일치하는 원인 후보 (분리 검증은 안 함) |
| 진행 상황 정보가 적음 | 과거 화면을 장당 16토큰으로 압축 (7B는 196토큰) | 긴 지시문의 진행 단계와 STOP 판단이 약할 수 있음 (가설, 폐루프에서 확인 필요) |
| 목표 해상도 | 224 입력, 패치 하나가 원본 약 46×34 px | 위치 보정(`offset_head`)이 있지만 7B보다 거침 |

**System 1: 경로**

| 한계 | 근거 | 영향 |
|---|---|---|
| System 2 목표에 그대로 의존 | 정답 목표 0.09 m vs 모델 목표 0.37 m | 목표가 틀리면 경로도 그대로 틀림. 지금 System 1만 고쳐서는 전체 성능이 크게 오르지 않음 |
| 갈림길에서 경로 하나만 냄 | 회귀는 평균을 냄 (DualVLN도 32개 평균이라 기존보다 나빠지진 않음) | 여러 갈래가 가능한 장면에서 중간으로 가는 경로가 나올 수 있음 |
| 깊이 정보를 쓰지 않음 | 원래 System 1은 깊이 추정으로 사전학습된 Depth-Anything 특징을 썼음. LayaNav는 의미 위주 SigLIP2 특징 | 오프라인 지표에서는 문제가 드러나지 않음(정답 목표 기준 0.09 m). 장애물 회피는 폐루프에서 확인 필요 |

**모델 전체**

| 한계 | 내용 |
|---|---|
| **최적 체크포인트가 System마다 다름** | 하나의 모델이라 System 2가 가장 좋은 시점(28k~46k)과 System 1이 가장 좋은 시점(96k)을 한 체크포인트로 동시에 얻을 수 없다 |
| 폐루프 미검증 | 성공률(SR, SPL), 충돌, 7B 대비 실제 시간 모두 아직 없음 |
| 고정된 학습 조건 | 카메라 1.25 m, 정면과 30° 아래, 화각 79°, 영어 R2R 스타일 지시문으로만 학습. 모델에 카메라 정보 입력이 없어 다른 조건에는 일반화를 기대하기 어렵다 |

### 11.4 개선 방향 (우선순위 순)

| 순위 | 대상 | 방법 | 겨냥하는 한계 |
|---|---|---|---|
| 1 | 공통 → System 2 | **C3: System 2가 가장 좋은 체크포인트(28k~46k)에서 System 2와 공유 인코더를 고정하고, System 1 경로 헤드만 다시 학습** (C1 방식) | 최적 시점 불일치, 판단 과적합 |
| 2 | System 2 | 판단 과적합 줄이기: 판단 부분 학습률 낮추기, 조기 종료, 데이터 증강 | 과적합 |
| 3 | System 2 | 목표 선택 강화: 7B의 목표 확률 분포를 따라 하는 distillation, 과거 토큰 늘리기(4×4 → 7×7), 입력 해상도 384 | 목표 선택, 진행 상황 |
| 4 | System 2 | 하이브리드: `act_head`가 불확실하다고 할 때만 7B 호출 | 목표 선택 (속도 이득은 줄어듦) |
| 5 | System 1 | 경로 후보 K개 + `scorer` 선택 (Laya 구조 확장), 필요하면 깊이 입력 추가 | 갈림길, 형상 정보 |
| – | 전체 | 후보 체크포인트를 Habitat에서 DualVLN과 같은 에피소드로 비교 (`compare_laya_s2.sh`) | 폐루프 미검증 |

지금 수치로 보면 **System 1은 충분히 동작하고, 개선할 곳은 System 2**다. System 1 쪽 작업(5순위)은 System 2 목표 정확도가 오른 뒤에 효과가 드러난다.

---

## 부록 A. 커밋 이력 (`7a5c624` 이후)

| 커밋 | 내용 |
|---|---|
| `b8f506d` | Laya-S2: 비자기회귀 경량 System 2 (모델, 데이터, latent 추출, 학습, 단독 System 1, 평가 모드) · System 2 |
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
| `275dbe4` | **LayaNav**: 단일 모델 (경로 헤드, C1/C2 학습, `laya_nav` 평가 모드) · System 1 |
| `d99177e` | 구현 보고서 |
| `fe9dbf2` | SigLIP 풀링 헤드 제거 · 공유 인코더 |
| `31fb3a8` | 경로 헤드 v2, 큰 데이터셋 학습 · System 1, 데이터 |
| `9aa19c2` | 경로 헤드 v2 문서 (구조, 첫 C1 실패 원인) · System 1 |

## 부록 B. 측정 환경

- RTX 3090 24 GB, torch 2.12 (cu130), transformers 4.51.0, diffusers 0.32.2
- 시간: CUDA 동기화 후 20~30회 평균, 워밍업 5회, `torch.autocast(bfloat16)`, 배치 1
- 인코더: `jhu-clsp/mmBERT-base`, `google/siglip2-base-patch16-224` (실제 크기, 무작위 head)
- DualVLN System 1: 실제 구조(`nextdit_async`), 무작위 가중치 (계산 시간 측정용)
