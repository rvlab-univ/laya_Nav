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

- N4(좌우 반전)는 N1·N2 다음이다. N2가 이기면 `START=none bash scripts/server/train.sh N4 …`로 돌린다.
- N5(교사 분포)는 교사 라벨이 끝난 뒤다.
- N3, N6, N7, N8은 아직 구현하지 않았다.
- 결과는 `runs/<이름>/log.jsonl`(학습·검증 기록)과 `runs/<이름>/last/`(체크포인트)에 남는다.

## 결과 가져오기

- **수치**: `tail -n 3 runs/<이름>/log.jsonl`, `cat runs/teacher/stats.json`
- **체크포인트**: 모델만 묶으면 약 2 GB다. 옵티마이저 상태(`train_state.pt`)는 이어서 학습할 때만 필요하다. 묶은 파일은 v3·v4 체크포인트 때와 같은 방법으로 받는다.

  ```bash
  tar -cf N1_model.tar --exclude=train_state.pt -C runs/N1 last
  ```
