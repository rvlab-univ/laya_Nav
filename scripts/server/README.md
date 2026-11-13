# 서버 실행 순서 (H200, 웹 터미널)

- 모든 명령은 저장소 폴더 `~/light_nav`에서 실행한다.
- 오래 걸리는 명령은 `nohup … &`로 돌리고, 진행은 `tail -n 20 logs/<이름>.log`로 본다. 브라우저가 끊겨도 계속 돈다.
- 모든 스크립트는 같은 명령을 다시 실행하면 이어서 한다.
- 서버는 공용이다. 스크립트가 시작할 때 `nvidia-smi` 결과를 로그에 남긴다. 다른 사람의 작업이 GPU를 쓰고 있으면 기다린다.

## 0. 코드 받기

```bash
cd ~ && git clone --recurse-submodules https://github.com/rvlab-univ/laya_Nav.git light_nav
cd ~/light_nav && mkdir -p logs
```

갱신은 `git pull && git submodule update --init`이다.

## 1. 환경 (둘을 차례로, 합쳐 30분 안팎)

```bash
nohup bash -c "bash scripts/server/setup_student_env.sh && bash scripts/server/setup_teacher_env.sh" > logs/setup.log 2>&1 &
```

| 환경 | 내용 |
|---|---|
| 학생 `lightnav` | Python 3.11, torch 2.7.1 (cu126), transformers 4.51.0, laya 0.4.31 (서브모듈). 드라이버가 CUDA 12.8을 지원하면 `TORCH_CUDA=cu128` |
| 교사 `navila` | NaVILA의 설치 방법 그대로: Python 3.10, torch 2.3.0, flash-attn 2.5.8, transformers 4.37.2 + NaVILA 교체 파일, deepspeed 0.9.5. Habitat은 넣지 않는다 |
| 교사 체크포인트 | `a8cheng/navila-llama3-8b-8f`, 약 17 GB → `assets/navila-llama3-8b-8f` |

## 2. 데이터 (몇 시간)

```bash
nohup bash scripts/server/download_data.sh > logs/download.log 2>&1 &
```

- 받는 것: NaVILA-Dataset의 R2R·RxR 주석과 프레임, ScanQA 주석과 영상
- 공간: 압축 파일을 풀고 나서 지운다. 푸는 동안 약 90 GB가 필요하고, 끝나면 약 45 GB가 남는다. 압축 파일을 남기려면 `KEEP_TAR=1`.
- 검증 분리는 git에 들어 있다(`splits/val_scenes.json`). VLN-CE 파일은 받지 않아도 된다.

## 3. 확인

```bash
bash scripts/server/check.sh               # TESTS=1이면 학생 테스트도 돌린다 (1분 안팎)
```

MISS로 나오는 줄이 없어야 다음으로 간다.

## 4. 교사 라벨

먼저 1,000개로 확인한다(수십 분).

```bash
nohup bash scripts/server/teacher_check.sh > logs/teacher_check.log 2>&1 &
```

로그 끝에서 세 가지를 본다.

| 줄 | 기대 | 다르면 |
|---|---|---|
| `generate check: 20/20 …` | 점수 1순위가 NaVILA가 실제로 생성한 답과 같다 | 전체를 돌리지 않는다. 로그의 `check …` 줄을 가져온다 |
| `… samples/s` | 프로세스 하나의 속도 | 걸리는 시간 = 679,532 ÷ (속도 × SHARDS) |
| `runs/teacher_check/stats.json`의 `top1_equals_label` | 높다 (NaVILA가 학습한 샘플이다) | 프롬프트나 전처리가 틀렸을 수 있다 |

확인이 끝나면 전체를 돌린다.

```bash
SHARDS=4 nohup bash scripts/server/teacher_labels.sh > logs/teacher.log 2>&1 &
```

- 프로세스 하나가 8B 모델 한 벌(약 18 GB)을 올린다. SHARDS=4면 약 72 GB다. 학습과 함께 돌릴 때는 `nvidia-smi`로 메모리를 본다.
- 프로세스마다 로그가 `logs/teacher_shard_<i>.log`에 남는다.
- 결과는 `runs/teacher/`에 생긴다: `keys.txt`, `teacher_probs.npy`, `stats.json`.

## 5. 학습

먼저 1,000 step으로 속도를 잰다.

```bash
NAME=N1_timing nohup bash scripts/server/train.sh N1 --max_steps 1000 > logs/N1_timing.log 2>&1 &
```

- 걸리는 시간 = 로그의 `sec_per_step` × 전체 step 수(시작할 때 `optimizer steps`로 출력됨)
- 메모리가 모자라면 끝에 `--grad_checkpointing`을 붙이거나 `--batch 8 --grad_accum 8`로 바꾼다.

본 실험:

```bash
nohup bash scripts/server/train.sh N1 > logs/N1.log 2>&1 &     # Laya 체크포인트에서 시작
nohup bash scripts/server/train.sh N2 > logs/N2.log 2>&1 &     # ModernBERT-large 원본에서 시작
```

**축소안 (2026-11-13)**: N1·N2가 지시문을 읽지 않는 것으로 진단되어(설계 B 7.2.1절) 아래 두 실행으로 줄였다.

```bash
nohup bash scripts/server/train.sh A > logs/A.log 2>&1 &     # 짝 맞추기 20% + 좌우 반전, 16,000 step
nohup bash scripts/server/train.sh B > logs/B.log 2>&1 &     # A + 교사 분포 (교사 라벨이 끝난 뒤)
```

- A는 교사 라벨이 필요 없다. 다만 GPU 메모리 약 50 GB를 쓴다. 교사 라벨을 `SHARDS=6`(약 108 GB)으로 돌리는 중이면 함께 올라가지 않는다.
- `MATCH`(짝 맞추기 비율, 기본 0.2), `STEPS`(기본 16000), `ALPHA`(기본 0.5)로 바꿀 수 있다.
- `status.sh`가 검증마다 "same answer with another instruction"을 보여 준다. 0.9 이상이면 지시문을 읽지 않는 것이다.

**A의 첫 검증 뒤 (2026-11-13, 설계 B 7.2.2절)**: A도 지시문을 읽지 않았다(같은 답 0.96, 짝 맞추기 0.50).

```bash
bash scripts/server/teacher_signal.sh                        # B가 도움이 될지: 같은 경로의 지시문끼리 NaVILA 분포 차이 (CPU, 몇 분)
bash scripts/server/match_eval.sh A > logs/match_eval.log 2>&1   # A가 학습 경로의 짝 맞추기는 맞히는지 (GPU 약 4 GB)
MATCH=0 nohup bash scripts/server/train.sh C > logs/C.log 2>&1 &   # 지시문 대조 손실 (CONTRAST, 기본 1.0), 짝 맞추기 없이
```

- A는 같은 step의 N2보다 정확도와 정지 재현율이 낮았고(4,000 step: 0.275 / 0.00 대 0.337 / 0.52), 짝 맞추기도 배우지 못했다. 그래서 C는 `MATCH=0`으로 돌린다.
- C가 지시문을 쓰기 시작하는지는 `status.sh`의 `contrast gap`(20 step마다)으로 본다. 0보다 커지면 쓰기 시작한 것이다.

**실행 D (2026-11-13 밤, 설계 B 7.2.3절)**: 로컬 실험에서 영상 토큰 539개로는 지시문 비교를 배우지 못했고, 11개로는 배웠다.

```bash
pkill -u irteam -f "scripts/train.py --out /home/irteam/light_nav/runs/C "    # C를 멈춤 (같은 구조라 기대가 낮음)
git pull
nohup bash scripts/server/train.sh D > logs/D.log 2>&1 &
```

- 시작할 때 학습 프레임 4,000장으로 태그 통계를 잰다. 로그의 `most frequent tags`에 bedroom, bathroom 같은 단어가 보이면 정상이다.
- 처음 200 step은 짝 맞추기 예열이다(손실 약 0.69). 그 뒤 `contrast gap`이 나온다.
- 검증 2,000 step마다: 짝 맞추기가 0.5보다 오르는지, 같은 답 비율이 0.9보다 내려가는지, 정확도가 N2(2,000 step 0.300, 4,000 step 0.337)와 비슷한지 본다.

- C는 A와 함께 올라가지 않는다(교사 라벨 2개 + A가 이미 약 104 GB). A를 멈춘 뒤에 시작한다.
- 교사 분포까지 넣으려면: `NAME=BC nohup bash scripts/server/train.sh B --contrast 1.0 > logs/BC.log 2>&1 &`

이전 계획:

- N4(좌우 반전)는 N1·N2 다음이다. N2가 이기면 `START=none bash scripts/server/train.sh N4 …`로 돌린다.
- N5(교사 분포)는 교사 라벨이 끝난 뒤다.
- N3, N6, N8은 아직 구현하지 않았다. N7(짝 맞추기 10%)은 2026-11-13에 구현했다.
- 결과는 `runs/<이름>/log.jsonl`(학습·검증 기록)과 `runs/<이름>/last/`(체크포인트)에 남는다.

## 진단

학습된 모델이 지시문과 영상을 실제로 쓰는지 잰다(검증 5,000개, 모델마다 수십 분, GPU 약 4 GB).

```bash
nohup bash scripts/server/diagnose.sh N1 N2 > logs/diagnose.log 2>&1 &
tail -n 30 logs/diagnose.log
```

결과는 `runs/diag/N1-N2.json`에 남는다. `shuf_text` 줄의 `same`이 0.9 이상이면 지시문을 읽지 않는 것이다.

## 진행 상황 보기

```bash
bash scripts/server/status.sh
```

한 화면에 아래 내용이 모두 나온다.
- 학습마다: 실행 중인지, step / 전체 step, step당 시간, 남은 시간, 최근 손실(ce), 최근 검증 정확도
- 교사 라벨: shard마다 진행 상황, 전체 속도와 남은 시간
- GPU: 메모리, 사용률, 프로세스별 메모리

## 결과 가져오기

- **수치**: `tail -n 3 runs/<이름>/log.jsonl`, `cat runs/teacher/stats.json`
- **체크포인트**: 모델만 묶으면 약 2 GB다. 옵티마이저 상태(`train_state.pt`)는 이어서 학습할 때만 필요하다. 묶은 파일은 v3·v4 체크포인트 때와 같은 방법으로 받는다.

  ```bash
  tar -cf N1_model.tar --exclude=train_state.pt -C runs/N1 last
  ```
