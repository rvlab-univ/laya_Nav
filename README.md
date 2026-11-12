# lightnav: 경량 NaVILA

NaVILA(8B VLA)의 판단 부분을 BERT 계열 비자기회귀 판단 모델(Laya)로 바꾸는 연구 코드다.
- 위치: `rvlab-univ/laya_Nav` 저장소의 `main` 브랜치 (2026-11-12부터)
- 이전 LayaNav(InternNav / DualVLN 기반)는 `layanav` 브랜치에 그대로 있다. 두 브랜치는 기록을 공유하지 않는다.
- 계획은 [PLAN.md](PLAN.md), 현재 설계는 [design_B_laya_native.md](design_B_laya_native.md)다.
- [design_A_layanav_based.md](design_A_layanav_based.md)는 보류 중이다.

## 구성

| 경로 | 내용 |
|---|---|
| `lightnav/` | 학생 모델 패키지 (질문 정의, 데이터, 모델, 지표) |
| `scripts/` | 학습 등 실행 스크립트 |
| `tests/` | 테스트 |
| `laya_base/` | LayaNav 코드 복사본 (`layanav` 브랜치 커밋 `6f94c26`, 참고용, 수정하지 않음). 출처는 [SOURCES.md](laya_base/SOURCES.md) |
| `NaVILA/` | 서브모듈: [AnjieCheng/NaVILA](https://github.com/AnjieCheng/NaVILA) `76b98f2`. 교사 모델과 평가 루프 |
| `third_party/laya_src/` | 서브모듈: [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya) `3a38cbe` (laya 0.4.31) |

`env/`, `data/`, `assets/`, `runs/`, `logs/`는 git에 올리지 않는다. 각 장비에서 따로 만든다.

## 받기 (서버 웹 터미널)

```bash
cd ~
git clone --recurse-submodules https://github.com/rvlab-univ/laya_Nav.git light_nav
```

이후 갱신:

```bash
cd ~/light_nav && git pull && git submodule update --init
```

2026-11-12 이전에 `main`(LayaNav)을 받아 둔 클론이 있으면, 그 클론에서 아래 명령을 한 번 실행한다.
- 그 클론이 `layanav` 브랜치를 따라가게 된다. 작업 폴더의 파일은 바뀌지 않는다.
- 바꾸기 전에 `git pull`을 하면 "unrelated histories" 오류로 멈춘다.

```bash
git fetch origin && git branch -m main layanav && git branch -u origin/layanav layanav
```

## 테스트 (이 PC)

```bash
cd /mnt/storage/light_nav
HF_HOME=$PWD/assets/hf_cache env/bin/python -m pytest tests -q -p no:cacheprovider
```
