# VLN-CE 평가 (이 PC)

NaVILA 논문과 같은 조건으로 학생을 평가한다. 평가 루프는 NaVILA의 VLN-CE 코드와 설정을 그대로 쓰고, NaVILA 8B를 부르던 자리에서 학생 프로세스에 묻는다. 설계와 기준은 [VLN-CE 평가 설계](https://claude.ai/artifact/EQ7No7dYevdu7xAAMWoEMi)에 정리했다.

| 프로세스 | 환경 | 하는 일 |
|---|---|---|
| 평가 루프 `scripts/eval_vlnce.py` | `env_vlnce`: Python 3.8, habitat-sim·habitat-lab 0.1.7 | 장면, 에피소드, 행동 실행, 지표 계산 |
| 학생 `lightnav/server.py` | `env`: Python 3.11, torch 2.7, GPU | 판단 한 번에 NaVILA 문장 하나를 돌려줌 |

## 준비 (한 번)

```bash
cd /mnt/storage/light_nav
bash scripts/eval/setup_vlnce_env.sh > logs/setup_vlnce.log 2>&1     # env_vlnce 만들기
```

| 데이터 | 위치 | 출처 |
|---|---|---|
| R2R-CE 에피소드, 정답 경로 | `data/vlnce/r2r/val_unseen{,_gt}.json.gz` | VLN-CE `R2R_VLNCE_v1-3_preprocessed.zip` (Google Drive `1fo8F4NKgZDH-bPSdVU3cONAkt5EW-tyr`) |
| RxR-CE 에피소드, 정답 경로 | `data/vlnce/rxr/val_unseen/` | VLN-CE `RxR_VLNCE_v0.zip` (Google Drive `145xzLjxBaNTbVgBfQ8e9EsBAV8W-SM0t`) |
| MP3D 장면 90개 | `data/scene_datasets/mp3d/` | Hugging Face `InternRobotics/Scene-N1`의 `mp3d_ce.tar.gz` (약관 동의 필요, CC BY-NC-SA 4.0) |

## 평가

```bash
# 1. 학생 띄우기 (체크포인트 폴더는 서버에서 받은 runs/<이름>/last)
HF_HOME=$PWD/assets/hf_cache nohup env/bin/python -m lightnav.server --ckpt runs/N1/last --port 8765 > logs/server_N1.log 2>&1 &

# 2. 평가 (끊기면 같은 명령으로 이어서 한다)
nohup env_vlnce/bin/python scripts/eval_vlnce.py --split r2r --agent http --out runs/eval/N1/r2r > logs/eval_N1_r2r.log 2>&1 &
nohup env_vlnce/bin/python scripts/eval_vlnce.py --split rxr --agent http --out runs/eval/N1/rxr > logs/eval_N1_rxr.log 2>&1 &

# 3. 요약, 모델 비교 (같은 에피소드끼리 쌍 부트스트랩)
env_vlnce/bin/python scripts/eval_summary.py runs/eval/N1/r2r runs/eval/N2/r2r --pair
```

- `--episodes 100`을 붙이면 고정 seed로 고른 100개만 평가한다(동작 확인과 중간 점검용).
- 결과는 `runs/eval/<모델>/<세트>/episodes.jsonl`에 에피소드마다 한 줄로 남는다(지표, 판단 수, 답별 횟수, 판단 시간).

## 점검 결과 (2026-11-13)

| 점검 | 결과 | 판정 |
|---|---|---|
| G3 정답 재생, R2R 50개 | SR 100.0, SPL 96.5, nDTW 100.0 | 통과 |
| G3 정답 재생, RxR 50개 (30° 회전) | SR 94.0, nDTW 98.0. 실패 3개는 목표까지 경로가 없는 에피소드 | 통과 |
| 하한선 "전진 75 cm"만, R2R 100개 | SR 0.0, OS 3.0, 매번 500 step | 기록 |
| 학습 전 학생, HTTP 연결 | 끝까지 돌고 판단 시간 중앙값 55.5 ms (RTX 5060 Ti, 이미지 8장 전부 인코딩) | 통과 |

- **RxR 정답 행동은 30° 회전 기준이다.** 15°로 재생하면 SR이 10%로 떨어진다. RxR 공식 설정이 30° 회전이기 때문이다.
- 그래서 G3는 30°(`--turn_angle 30`)로 확인했다. 학생 평가는 NaVILA와 같게 15°로 한다. 정답 경로로 계산하는 지표(nDTW)는 행동 크기와 상관없다.
- 목표까지 경로가 없는 에피소드는 SPL과 NE가 무효로 나온다. NaVILA의 집계(`eval_jsons.py`)와 같게, 그 에피소드는 SPL·NE 평균에서 빼고 개수를 따로 적는다. 장면 확인에서 R2R 1,839개 중 측지 거리가 VLN-CE 파일과 1 cm 안으로 맞는 것은 97.6%였다.
