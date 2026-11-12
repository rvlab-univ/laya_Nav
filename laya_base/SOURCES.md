# laya_base: LayaNav에서 가져온 코드

`/mnt/storage/laya_Nav` 커밋 `6f94c26`(2026-11-09, 이 저장소의 `layanav` 브랜치)에서 **수정 없이** 복사했다.
학생 모델(NaVILA 경량화)은 이 코드를 바탕으로 새로 만든다. 원본과 비교할 수 있도록 이 폴더는 고치지 않는다.

## 파일

| 파일 | 원래 위치 | 학생 모델에서 쓰는 것 |
|---|---|---|
| `internnav/model/basemodel/laya_s2/laya_s2.py` | 같은 경로 | 판단 모델 본체: mmBERT + SigLIP2, 선택지 표시 토큰 점수, Laya 손실(로그 + spherical), match 헤드 |
| `internnav/model/basemodel/laya_s2/grounding.py` | 같은 경로 | v3 근거 연결 (구절 나누기, 고정 SigLIP2 패치 일치 지도) |
| `internnav/model/basemodel/laya_s2/agent.py` | 같은 경로 | 추론 래퍼: 과거 프레임 특징 캐시 |
| `internnav/model/basemodel/laya_s2/laya_nav.py` | 같은 경로 | 쓰지 않음. `__init__.py`가 불러오기 때문에 함께 복사 |
| `internnav/dataset/laya_s2_dataset.py` | 같은 경로 | `mirror_instruction`(좌우 반전 지시문), `make_image_transform`, `history_ids`, `collate` 형식 |
| `scripts/train/laya_s2/train_laya_s2.py`, `train_laya_nav.py` | 같은 경로 | 학습 루프 참고: DDP, 장면 단위 검증 분리, 로그 형식, 체크포인트 저장 |
| `scripts/train/laya_s2/diagnose_decision.py` | 같은 경로 | 진단 참고: 혼동표, 확률 보정, 지름길 지표(`swap_directions`, `mirror`) |
| `scripts/realworld/http_internvla_client.py`, `controllers.py`, `thread_utils.py` | `scripts/realworld/` | Go2 시연 참고: ROS 2 노드, `/cmd_vel_bridge`, 오도메트리 |
| `scripts/realworld/laya_kimm_client.py` | `/mnt/storage/laya_work/isaac/` | Isaac KIMM 주행 클라이언트 참고: 카메라 자르기(시야각 맞추기), 명령 기록 |
| `tests/laya_s2/*.py` | `tests/laya_s2/` | 작은 인코더로 도는 CPU 테스트 |
| `docs/v3_grounding.md`, `docs/v5_copycat.md` | `docs/laya_nav/model_improvement/` | 설계와 비용 기록 |

`internnav/__init__.py`, `internnav/model/__init__.py`, `internnav/model/basemodel/__init__.py`는 빈 파일로 새로 만들었다.
원본은 다른 정책 모델을 불러오는 코드라 여기서는 필요 없다.

## 학생 모델에서 빼는 것

- 30° 아래 화면, 바닥 칸 196개, offset 헤드, latent 헤드, 경로 헤드(`laya_nav.py`): NaVILA 출력에 바닥 칸 목표가 없다
- act(escalate) 헤드와 escalate 손실: 불확실한 판단을 8B로 넘기는 방식은 하지 않는다
- v4 위치·움직임 입력(`memory_features`): NaVILA 입력에 위치가 없다

## 테스트 (이 PC, `/mnt/storage/laya_work/env`, Python 3.9)

```bash
cd /mnt/storage/light_nav/laya_base
PYTHONPATH=$PWD HF_HOME=/mnt/storage/light_nav/assets/hf_cache \
    /mnt/storage/laya_work/env/bin/python -m pytest -q -p no:cacheprovider tests/laya_s2
```

31개 통과, 11개 실패(2026-11-09). 실패 원인은 모두 일부러 가져오지 않은 파일이다.
- InternData-N1 로더(`internvla_n1_lerobot_dataset.py`) 9개
- 경로 헤드 평가(`eval_traj.py`) 1개
- Habitat 분산 평가 1개
