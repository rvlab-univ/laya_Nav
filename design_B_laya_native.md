# 설계 B: Laya 원본 형식으로 만드는 NaVILA 학생 모델

> 이 문서는 Claude Code에게 주는 작업 지시서다. 작업 폴더는 `/mnt/storage/light_nav/`이다.
> 2026-11-11 개정, 2026-11-12 보완. 일정, 서버 작업 방식, 교사 라벨 절차는 `PLAN.md`를 따른다. 이 문서는 `PLAN.md` 4절(학생 모델)을 대체한다.

**2026-11-12 보완에서 바뀐 것**

- 보조 과제 두 가지를 본 학습에 섞는다(4.1절, 실험 N7·N8).
  - 지시문-영상 짝 맞추기: 가짜 짝의 일부를 "랜드마크만 바꾼 지시문"으로 만든다.
  - ScanQA를 4지선다로 바꾼 것.
- 짝 맞추기는 NaVILA의 R2R·RxR로 만든다. ScaleVLN 같은 InternData-N1 데이터는 쓰지 않는다.
- EnvDrop 렌더링과 Human 데이터는 마지막 단계로 미룬다.
- NaVILA-Dataset의 ScanQA, EnvDrop, Human을 직접 확인한 내용을 1.5절에 적었다.
- 저장소가 정해졌다(3.1절). RxR 검증 분리가 아직 장면 단위가 아닌 문제를 4절에 적었다.

**2026-11-11 개정에서 바뀐 것**

- DualVLN 계열(InternNav 코드, InternData-N1, 그 평가 루프)은 쓰지 않는다. 데이터, 라벨, 평가는 모두 NaVILA 것을 쓴다.
- 본선은 **질문 하나, 선택지 10개**다. 질문 3개로 나누는 방식은 비교 실험으로 내렸다.
- 첫날 통과해야 하는 관문 2개와 되돌아갈 기한을 넣었다(5.1절).
- ModernBERT-large 원본에서 시작하는 대조군을 본 실험과 같은 시점에 돌린다.
- 정렬 단계(B1)는 건너뛴 경우와 함께 돌려 이득이 없으면 버린다.
- `design_A_layanav_based.md`는 InternData-N1의 포즈와 `internnav/` 코드를 전제로 해서 보류한다.

## 0. 한 줄 요약

NaVILA의 상위 VLA(8B)를 Laya 결정 모델로 바꾼다. Laya를 내비게이션용으로 뜯어고치지 않고, 내비게이션을 Laya의 원래 문제 형식(질문 + 선택지 + 상태)으로 옮긴다. 공개된 Laya 영어 체크포인트(ModernBERT-large)의 가중치를 물려받고, 새로 만드는 부품은 영상 투영층뿐이다. 입출력은 NaVILA와 같다: 지시문과 이미지 8장을 받아 10문장 중 하나를 낸다.

## 1. 확인한 사실

### 1.1 NaVILA 데이터 (2026-11-11, 이 PC의 `data/NaVILA-Dataset/`에서 직접 셈)

- 주석 필드는 `video_id`, `q`, `a`, `frames` 네 개뿐이다. **포즈와 저수준 행동 열은 없다.**
- `video_id`는 `<에피소드>-<판단 번호>`다. 판단 번호는 에피소드마다 0부터 빠짐없이 이어지고, 마지막 판단은 항상 정지다.
- `frames`는 `frame_0`부터 현재 프레임까지 빠짐없이 든 목록이다. 프레임 하나가 저수준 행동 한 번이다.
  - 연속한 두 판단의 프레임 수 차이는 앞 판단의 행동 크기와 항상 같다(R2R 277,775쌍, RxR 384,278쌍 모두 일치). 예: `turn right 45 degree` 다음 판단은 프레임이 3장 늘어난다.
  - 그래서 저수준 행동 열을 주석만으로 복원할 수 있다.
- 답 문장은 10개다.
  - 정지: `I think I should stop because I have finished the instruction.`
  - 나머지 9개: `The next action is move forward 25|50|75 cm.`, `The next action is turn left|right 15|30|45 degree.`
- 주석에는 균형을 맞추려는 복제가 이미 들어 있다.

| 데이터 | 행 | 고유 샘플 | 에피소드 | 복제된 것 |
|---|---|---|---|---|
| R2R | 353,894 | 288,594 | 10,819 | 정지 3배, 30°·45° 회전 2배 |
| RxR | 417,578 | 390,938 | 6,660 | 정지 5배 |

- 고유 샘플 기준 정지 비율은 R2R 3.7%, RxR 1.7%다.
- Human, EnvDrop, ScanQA는 1.5절에 따로 적었다.
- 이 PC에는 주석 전체, R2R 프레임 일부(`R2R/train_sample`, 에피소드 216개), ScanQA 영상과 주석이 있다. R2R·RxR 프레임 전체는 서버에서 받는다.

### 1.2 NaVILA 코드 (2026-11-11, `NaVILA/`에서 확인)

- 프레임 선택(`llava/mm_utils.py`의 `vlnce_frame_sampling`, `evaluation/vlnce_baselines/navila_trainer.py`의 `sample_and_pad_images`, 두 함수의 규칙이 같다)
  1. 프레임이 8장보다 적으면 앞쪽에 검은 이미지를 채워 8장으로 만든다.
  2. `np.linspace(0, len-1, num=7, endpoint=False, dtype=int)` 위치의 7장을 고른다.
  3. 마지막 프레임(현재)을 붙인다.
- 학습 프롬프트는 `llava/data/dataset.py` 2237-2242행에 있다. 지시문은 그 앞에서 대문자·공백 정리를 거친다(2231-2233행).
- 답 해석(`navila_trainer.py` 221-283행): `stop`을 가장 먼저 찾고, 다음으로 `is move forward`, `is turn left`, `is turn right`를 찾는다. 숫자는 25·50·75 cm, 15·30·45°로 맞춘다. 저수준 행동을 N/25회 또는 N/15회 실행한다.
- 교사 체크포인트는 `a8cheng/navila-llama3-8b-8f`다.

### 1.3 Laya (`NandhaKishorM/laya`, 커밋 3a38cbe, 패키지 0.4.31)

2026-11-11에 소스를 `third_party/laya_src`에 받아 설치하고 대조했다(관문 G1 통과).

- 설치: PyPI의 `laya`는 0.3.4까지만 있다. 0.4.31은 GitHub 커밋에서 설치한다. Python 3.10 이상이 필요해서 기존 `/mnt/storage/laya_work/env`(3.9)에는 못 넣는다. 학생 전용 환경 `/mnt/storage/light_nav/env`(Python 3.11, torch 2.7.1, transformers 4.51.0)를 만들고 `pip install --no-deps`로 넣었다(`vegaml`은 쓰지 않는다).
- 영어 체크포인트: `load_checkpoint("english")` → Hub의 `convaiinnovations/laya`. 설정은 encoder `answerdotai/ModernBERT-large`, `head_layers` 2, `max_len` 512, `head_max_len` 192이다. 인코더 394.8M + head 25.2M + scorer 1.1M = 421.3M.
- 인코더는 28층, 3층마다 전역 attention, 국소 attention 창 128토큰이다.
- 체크포인트 불러오기: `laya.train.load_checkpoint(model_dir)` → `(DecisionModel, tokenizer, cfg)`. Hub 이름도 받는다.
- 입력 열 형식(`laya/common.py`의 `build_head`, `build_sequence`):
  `[CLS] "<유형> question: <지시>" [SEP] [MASK] 선택지0 [MASK] 선택지1 … [SEP] 상태 [SEP]`
  - choice의 선택지 문구는 `"키: 설명"`이다. 선택지 하나는 최대 48토큰이다.
  - 질문 문장은 `head_max_len`에서 선택지를 뺀 만큼으로 잘린다.
  - 상태는 남은 길이만큼만 들어가고 넘치면 뒤쪽이 잘린다.
- `DecisionModel.forward`(`laya/common.py` 773-836행)는 `input_ids`만 받는다. `inputs_embeds` 경로가 없다.
  - 인코더 출력에 `type_emb(qtype)`를 더하고, head 2층을 거쳐, `[MASK]` 위치 벡터에 `scorer`를 적용한다.
- 학습 손실: `rlcd_loss(logits, target, mask, qtype, sigma, samples=4, w_sph=0.75, w_rps=1.0)`. `TrainConfig` 기본값은 encoder lr 2.5e-5, head lr 1e-4, sigma 0.4 → 0.1.
- 온도 보정: `laya/calibrate.py`의 `fit_temperature_map`.
- 원본 인코더에서 새 모델 만들기(N2): `laya.common.build_model({"encoder": 이름, "head_layers": 2})`.
- 모델 카드가 밝힌 약점(다시 확인하지 않음): score가 가장 약한 유형이다. act head 출력은 쓸모없다. 기본 체크포인트는 미세조정 없이는 성능이 낮다.

### 1.4 작업 환경 (`PLAN.md` 2절, 9절)

- 이 PC: RTX 5060 Ti 16 GB. 코드 개발, 테스트, 학생의 폐루프 평가.
  - 학생: `/mnt/storage/light_nav/env`(Python 3.11, torch 2.7.1, transformers 4.51.0, laya 0.4.31). Habitat은 없다.
  - 기존: `/mnt/storage/laya_work/env`(Python 3.9, Habitat 0.2.4). `laya`를 설치할 수 없다.
  - 서버의 학생 환경도 Python 3.10 이상이어야 한다. 서버 `laya` 환경의 Python 버전은 확인하지 않았다.
- 서버(H200, 웹 터미널): 데이터, 교사 라벨, 학습, 오프라인 진단. 명령은 `scripts/server/*.sh`로 만들고 `nohup`으로 돌린다.
- NaVILA 환경(transformers 4.37.2)과 학생 환경은 섞지 않는다.

### 1.5 NaVILA-Dataset의 다른 데이터 (2026-11-12, `data/NaVILA-Dataset/`와 `NaVILA/`에서 확인)

NaVILA 8B의 학습 데이터는 `r2r+rxr+envdrop+human+scanqa+video_chatgpt+sharegpt_video+sharegpt4v_sft`다(`scripts/train/sft_8frames.sh`). 이 중 이 설계에 관련된 것은 아래 세 가지다.

**ScanQA**: 바로 쓸 수 있다.
- `ScanQA/videos.tar.gz`(664 MB)는 이름과 달리 압축되지 않은 tar다.
  - mp4 800개가 들어 있다(장면당 1개, 평균 0.83 MB).
  - 학습 주석의 장면 562개, 검증 주석의 장면 71개 영상이 빠짐없이 있다.
- 주석은 `annotations/ScanQA_v1.0_{train,val,test_w_obj,test_wo_obj}_reformat.json`이다.
  - 필드: `video_id`(장면), `question_id`, `q`, `a`(정답 목록), `object_ids`, `object_names`
  - 질문 끝에 `Answer briefly.`가 붙어 있다.
- 문항 수
  - 학습 25,563문항. 정답이 1개인 문항이 96.9%다.
  - 검증 4,675문항. 대부분 정답이 2개다.
  - 학습에서 장면당 문항 수 중앙값은 30, 장면당 서로 다른 정답 수 중앙값은 24(최소 2)다.
- 질문 유형(첫 단어 기준, 학습): what 43%, where 20%, color 20%, how many 4%, 기타 13%. 정답은 평균 2.4단어다.
- NaVILA는 정답 문장을 그대로 생성하도록 학습했다(`datasets_mixture.py`의 `scanqa`).
- ScanNet 장면을 사람이 손으로 들고 찍은 스캔 영상이라 로봇 시점(높이 1.25 m, 수평 카메라)과 다르다. ScanNet 이용 약관(비상업 연구용)을 따른다.

**EnvDrop**: 렌더링이 필요하다.
- 주석은 `{video_id, instruction}` 146,304개뿐이고 영상이 없다. 영상은 VLN-CE(Habitat)로 직접 렌더링해야 한다.
- NaVILA에서 EnvDrop은 경로 영상 8장을 보고 지시문을 쓰는 과제다(`LazyEnvDropDataset`, 프롬프트 "… please describe the navigation trajectory of the robot.").

**Human**
- 형식은 R2R과 같다(557,546행).
- 프레임은 공개되지 않았다. 유튜브 영상 1,982개를 직접 받아 1 fps로 뽑아야 한다.

## 2. 범위

**한다**: 새 패키지 `lightnav/`(모델, 데이터셋, 학습, 진단, 학생 프로세스), 테스트, 폐루프 연결.

**하지 않는다**

- `laya` 패키지 수정(가져다 쓰기만 한다), `laya_base/` 수정, NaVILA 하위 보행 정책 수정.
- InternNav 코드와 InternData-N1 사용. 보조 과제도 InternData-N1(ScaleVLN 포함)로 만들지 않는다.
- 포즈·움직임 입력, 바닥 칸 목표, 경로 헤드, 내려다보기 뷰.
- 불확실한 판단을 8B로 넘기기(`PLAN.md` 1절). act head는 쓰지 않는다.

서버는 공용이므로 GPU 사용 전에 `nvidia-smi`로 확인하고, 실행 중인 학습은 건드리지 않는다.

## 3. 모델

### 3.1 파일

저장소는 `rvlab-univ/laya_Nav`의 `main`이고, 작업 폴더는 `/mnt/storage/light_nav/`다(2026-11-12 결정).

| 새 파일 | 내용 |
|---|---|
| `lightnav/model.py` | `LayaNativeConfig`, `LayaNative`, `forward_embeds` |
| `lightnav/questions.py` | 질문 정의, 답 문장 ↔ 선택지 번호, 결정 규칙 |
| `lightnav/data.py` | NaVILA 주석 읽기, 프레임 선택, 검증 분리, 정렬 과제, collate |
| `lightnav/tasks.py` | 보조 과제 샘플: 짝 맞추기, ScanQA 4지선다 (4.1절) |
| `lightnav/teacher.py` | 교사 점수 계산: 프롬프트 한 번 + 답 10개를 캐시 위에서 함께 계산, NaVILA 답 해석 |
| `scripts/make_split.py`, `splits/val_scenes.json` | 장면 단위 검증 분리와 그 결과 (4절) |
| `requirements/student.txt` | 서버 학생 환경 패키지 |
| `scripts/extract_scanqa_frames.py` | ScanQA 영상에서 8장을 한 번 뽑아 jpg로 저장 |
| `lightnav/agent.py` | 추론 래퍼(영상 특징 캐시) |
| `lightnav/server.py` | 학생 프로세스(평가 루프와 Go2가 호출) |
| `scripts/train.py` | 학습 (`--stage align / nav`) |
| `scripts/diagnose.py` | 오프라인 진단 |
| `scripts/extract_teacher_probs.py` | 교사 확률 추출(NaVILA 환경에서 실행) |
| `scripts/server/*.sh` | 서버용 실행 스크립트 |
| `tests/` | CPU 테스트 |

`laya_base/`에서 가져오는 것은 함수 복사로 한다(`internnav`를 import하지 않는다): `mirror_instruction`, `make_image_transform`, 학습 루프의 DDP·로그·체크포인트 저장 부분.

### 3.2 `forward_embeds`: Laya 순전파를 임베딩 입력으로 옮기기

`DecisionModel.forward`가 `input_ids`만 받으므로, 인코더 호출 뒤의 계산을 그대로 옮긴 함수를 만든다.

```python
def forward_embeds(dm, inputs_embeds, attention_mask, marker_pos, marker_mask, qtype):
    h = dm.encoder(inputs_embeds=inputs_embeds, attention_mask=attention_mask).last_hidden_state
    h = h + dm.type_emb(qtype)[:, None, :]
    pad = ~attention_mask.bool()
    for layer in dm.head.layers:
        h = layer(h, src_key_padding_mask=pad)
    m = torch.gather(h, 1, marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1)))
    return dm.scorer(m).squeeze(-1).float().masked_fill(~marker_mask, -1e4)
```

위 코드는 개정 전 문서의 기억을 옮긴 것이다. 설치한 패키지의 `forward`를 읽고 그대로 맞춘다.

### 3.3 입력 조립

```
[CLS] 질문 [SEP] [MASK] 선택지 ×10 [SEP] 상태(지시문) | 영상 토큰 [SEP]
```

1. `laya.train.to_internal`로 질문을 내부 형식으로 바꾸고 `laya.common.build_sequence(tok, state, q, max_len=TEXT_MAX, head_max_len=HEAD)`로 `(ids, markers)`를 얻는다. `TEXT_MAX`는 448, `HEAD`는 192로 시작한다.
2. `ids`를 임베딩으로 바꾼 뒤 **마지막 `[SEP]` 바로 앞에 영상 토큰을 끼운다.** 질문과 선택지가 앞에 있으므로 `markers` 위치는 변하지 않는다.
3. 상태는 `{"instruction": 지시문}`이다. 지시문은 NaVILA와 같은 정리(1.2절)를 거친다.

지시문을 질문이 아니라 상태에 넣는 이유는 `build_head`가 질문 문장을 `head_max_len` 안으로 자르기 때문이다. 상태에 넣으면 질문·선택지 부분이 매 판단마다 같고, Laya가 학습한 "짧은 질문 + 긴 상태" 형태와도 맞는다.

전체 열은 최대 약 990토큰이다(글자 448 + 영상 539).

### 3.4 영상 토큰

| 항목 | 값 |
|---|---|
| 프레임 선택 | NaVILA와 같은 규칙(1.2절)으로 8장의 위치를 구한다. 그중 검은 이미지에 해당하는 위치는 넣지 않고 mask로 뺀다 |
| 인코더 | `google/siglip2-base-patch16-224`, 패치 196개. 512×512 프레임을 224로 줄인다 |
| 과거 7장 | 7×7 평균 → 장당 49토큰 |
| 현재 1장 | 196토큰 그대로 |
| 투영 | `Linear(768,1024) → GELU → Linear(1024,1024)` (신규) |
| 더하는 임베딩 | 프레임 순서 8개, 패치 위치 (신규, 표준편차 0.02 초기화) |

프레임 순서 임베딩은 뒤에서부터 센다. 현재 프레임이 항상 같은 번호를 받아야 에피소드 초반(프레임 8장 미만)과 형태가 맞는다.

### 3.5 질문 (본선: 10지선다 하나)

```python
ANSWERS = [  # NaVILA 문장 그대로. 번호가 선택지 순서다
    "I think I should stop because I have finished the instruction.",
    "The next action is move forward 25 cm.",
    "The next action is move forward 50 cm.",
    "The next action is move forward 75 cm.",
    "The next action is turn left 15 degree.",
    "The next action is turn left 30 degree.",
    "The next action is turn left 45 degree.",
    "The next action is turn right 15 degree.",
    "The next action is turn right 30 degree.",
    "The next action is turn right 45 degree.",
]

QUESTION = {
    "type": "choice",
    "instructions": "What should the robot do next to follow the instruction?",
    "criteria": {
        "stop": "the instruction is finished, stop here",
        "forward 25": "move forward 25 cm",
        "forward 50": "move forward 50 cm",
        "forward 75": "move forward 75 cm",
        "left 15": "turn left 15 degree",
        "left 30": "turn left 30 degree",
        "left 45": "turn left 45 degree",
        "right 15": "turn right 15 degree",
        "right 30": "turn right 30 degree",
        "right 45": "turn right 45 degree",
    },
}
```

- 질문과 선택지 10개가 `HEAD`(192토큰) 안에 잘리지 않고 들어가는지 테스트로 확인한다. 넘치면 설명을 줄인다.
- 목표 분포는 `laya.train.target_from_expected`(정답 하나)와 `target_from_gold`(확률 분포, 교사 증류용)로 만든다.

**결정 규칙**

```
p = softmax(logits / T)            # T는 보정 전에는 1
기본: argmax(p)
비교: p[stop] > τ_stop 이면 정지, 아니면 나머지 9개 중 argmax
```

`τ_stop`은 재학습 없이 검증 데이터에서 고른다. 학생은 `ANSWERS`의 문장을 그대로 돌려주므로 NaVILA 평가 루프의 해석 코드를 고칠 필요가 없다.

### 3.6 손실

- `laya.train.rlcd_loss`를 그대로 쓴다. sigma는 0.4에서 0.1로 줄인다.
- 정지 샘플을 따로 반복하지 않는다. 주석에 이미 복제가 들어 있다(1.1절). 학습은 NaVILA와 같게 복제를 포함한 행 전체로 한다.
- 교사 증류(N5): 목표를 `(1−α)·정답 + α·교사 확률`로 둔다. α는 0.5로 시작한다.
- 보조 과제(4.1절)도 같은 손실을 쓴다. 과제마다 선택지 수가 달라서 `marker_mask`로 빈 선택지를 막는다.

## 4. 데이터

- R2R과 RxR의 주석과 프레임을 쓴다.
- **영어 지시문만 쓴다.** ModernBERT-large와 Laya 영어 체크포인트는 영어 전용이다. RxR의 비영어 지시문(`PLAN.md` 3절 기준 435개)은 뺀다.
- **검증 분리는 장면 단위다.** 2026-11-12에 `splits/val_scenes.json`으로 만들었고, 모든 실험이 이 파일을 함께 쓴다.
  - **R2R**: `video_id` 앞 숫자는 VLN-CE R2R의 `episode_id`다. `train.json.gz`의 10,819개와 번호, 지시문 문장이 모두 같다.
  - **RxR**: NaVILA는 RxR 에피소드에 자체 번호를 붙였다. 같은 번호의 RxR `instruction_id`는 다른 지시문이다. 그래서 공개된 원본 RxR 지시문 파일(`rxr_train_guide.jsonl.gz`)과 지시문 문장으로 대조한다(`rxr_episode_scenes`). 6,660개가 모두 장면 하나에 대응한다.
  - **두 데이터를 함께 나눈다**: R2R과 RxR은 같은 MP3D 학습 장면을 쓴다. 따로 뽑으면 한쪽의 검증 장면이 다른 쪽 학습에 들어가므로, 두 데이터에서 같은 장면을 뺀다(`make_val_split`).
  - **결과**: 장면 61개 중 3개(`S9hNv5qa7GM`, `mJXqzFtmKg4`, `rPc6DW4iMge`)를 뺐다. 에피소드 기준으로 R2R 744/10,819개, RxR 469/6,654개(영어만)가 검증이다.
  - 다시 만들 때는 `scripts/make_split.py`를 쓴다. 분리 파일이 git에 있으므로 서버에는 VLN-CE 파일이 필요 없다.
- 검증 지표는 복제를 뺀 고유 샘플로 계산한다. 분리를 에피소드 단위로 하므로 복제가 학습과 검증에 나뉘어 들어가지는 않는다.
- 분리 결과(검증 에피소드 목록)는 파일로 저장하고 모든 실험이 같은 파일을 읽는다.
- **좌우 반전**: 이미지를 좌우로 뒤집고, 지시문의 left와 right를 바꾸고(`mirror_instruction`), 정답의 `turn left X`와 `turn right X`를 바꾼다.

### 4.1 보조 과제 (2026-11-12 추가, N7·N8)

두 과제 모두 Laya의 choice 질문(`qtype` 0)이다. 모델 구조는 그대로 쓰고, 선택지 수만 과제마다 다르다.

**(A) 지시문-영상 짝 맞추기**

- **출처**: NaVILA R2R·RxR 주석. 각 에피소드의 마지막 판단(정지) 샘플이 경로 전체 프레임 목록을 갖고 있다.
- **입력**: 경로 전체에서 NaVILA 규칙(1.2절)으로 고른 8장 + 상태 `{"instruction": 지시문}`
- **질문**: `"Does the instruction describe this route?"`, 선택지 2개. 질문과 선택지는 합쳐 30토큰이다(2026-11-12, 실제 토크나이저).
  - `yes: the instruction describes this route`
  - `no: the instruction describes a different route`
- **짝 비율**: 진짜 짝 절반, 가짜 짝 절반
- **가짜 짝 만드는 방법**
  1. **랜드마크 교체** (가짜 짝의 절반 이상)
     - 지시문의 물체 이름 하나를 다른 물체로 바꾼다(sofa → bed).
     - 물체 구절은 `laya_base/internnav/model/basemodel/laya_s2/grounding.py`의 `split_instruction`과 `landmark`로 찾는다.
     - 바꿔 넣을 물체는 학습 지시문 전체에서 모은 랜드마크 목록에서 뽑는다.
     - 바꿀 물체가 없는 지시문은 2번 방법을 쓴다.
  2. **다른 에피소드의 지시문** (5.3절 (나)와 같음)
- **랜드마크 교체를 쓰는 이유**
  - 다른 에피소드의 지시문은 길이나 방향어만으로도 가려낼 수 있다.
  - 물체만 바꾸면 영상 속 물체를 봐야 가려낼 수 있다(LayaNav `next_candidates.md` 4번).
- **주의**
  - 바꿔 넣은 물체가 실제로 경로에 보일 수 있다(라벨 잡음).
  - 자주 나오는 구조물 단어(door, room, hallway, stairs, wall, floor 등)는 바꾸지도, 바꿔 넣지도 않는다.
  - 검증 에피소드의 지시문은 학습 쪽 가짜 짝에 쓰지 않는다.

**(A) 구현 (2026-11-13, `lightnav/tasks.py`)**

- **랜드마크 어휘**
  - R2R 랜드마크 구절(`split_instruction`)의 끝 단어 빈도에서 골랐다.
  - 네 묶음으로 나눈다: 방(bedroom, kitchen 등), "~ room" 앞말(living, dining 등), 물체 단수, 물체 복수.
  - 교체는 같은 묶음 안에서만 한다. "walk into the piano"처럼 글만 봐도 이상한 문장이 생기지 않게 하려는 것이다.
  - 지시문에 나온 그 단어는 모두 바꾼다(한 문장 안에서 앞뒤가 맞게).
  - 바꿔 넣을 단어는 지시문에 없는 것만 고른다.
  - 교체할 랜드마크가 있는 지시문: R2R 86%, RxR 98%
- **같은 경로 가려내기**
  - "다른 에피소드" 가짜 짝에 같은 경로의 다른 지시문이 뽑히면 안 된다. R2R은 경로마다 지시문이 3개다.
  - "프레임 수 + 정답 동작 열"이 같으면 같은 경로로 본다. 서버에 VLN-CE 파일이 없어도 된다.
  - R2R에서 서명 3,600개가 실제 경로 3,603개와 거의 일치한다(서로 다른 경로가 같은 서명인 경우 3개).
- **검증**: 검증 경로마다 진짜 1개, 가짜 1개를 seed로 고정한다(약 2,400개). 정확도는 진짜, 랜드마크 교체, 다른 에피소드별로 따로 기록한다(`val_match`).
- **배치**: 선택지 수가 다른 샘플을 함께 넣는다.
  - `collate`가 `marker_pos`를 10칸으로 채우고(빈 칸 -1) `marker_mask`를 만든다.
  - 손실과 `forward`가 이 마스크를 쓴다.
- **학습 중 지시문 확인**: 검증마다 같은 샘플을 다른 샘플의 지시문으로 한 번 더 돌린다. `text_same`(같은 답 비율)과 `text_kl`을 기록한다(7.2.1절).

**(B) ScanQA 4지선다**

- **입력**: 장면 영상에서 고르게 뽑은 8장
  - 마지막 장도 "현재" 칸(196토큰)에 들어간다.
  - 학습 전에 `scripts/extract_scanqa_frames.py`로 한 번 뽑아 jpg로 저장한다. 학습 중에는 영상을 디코딩하지 않는다.
- **질문과 상태**
  - 질문 문장은 `Answer briefly.`를 지우고 Laya 질문 자리(`instructions`)에 넣는다. 짧고, Laya가 원래 학습한 형태다.
  - 상태는 비운다. `build_sequence`는 빈 상태를 `{}`(2토큰)로 넣고 정상 동작한다(2026-11-12 확인).
  - 예시 문항 "What color is the chair in the kitchen?"은 질문과 선택지 4개를 합쳐 32토큰이다.
- **선택지 4개**: 정답 1개 + 오답 3개
  - 오답은 같은 장면, 같은 질문 유형(what, where, color, how many, 기타)의 다른 문항 정답에서 뽑는다.
  - 모자라면 다른 장면의 같은 유형에서 채운다.
  - 유형을 맞추지 않으면 "무슨 색?"에 위치나 개수가 오답으로 붙어, 영상을 보지 않고 답 형태만으로 풀린다.
  - 정답 목록과 겹치는 오답은 뺀다.
  - **오답끼리도 겹치지 않게 한다.**
    - 비교 전에 대소문자, 관사, 철자 변형(grey/gray, colour/color 등)을 맞춘다.
    - 예: 검증 문항 "What color is the chair in the kitchen?"(정답 dark brown, brown)에 같은 장면·같은 유형 규칙만 쓰면 오답이 black, grey, gray로 뽑혔다. 그러면 grey와 gray가 사실상 같은 선택지가 된다(2026-11-12 확인).
  - 정답과 한쪽이 다른 쪽을 포함하는 오답(정답 dark brown에 오답 brown)도 뺀다. 맞는지 틀리는지 애매하기 때문이다.
  - 정답이 여러 개면 하나를 선택지로 쓰고, 나머지는 오답 후보에서 뺀다.
  - 선택지 순서는 샘플마다 섞는다.
- **검증**: ScanQA val 4,675문항을 같은 방식으로 바꾼다. 오답과 순서는 seed로 고정한다.
- **양**
  - 학습 문항이 25,563개라, 10% 비율이면 한 epoch에 문항마다 약 4번 반복된다.
  - 과제별 검증 정확도를 보고 과적합이면 비율을 낮춘다.
- **한계**: 사람이 손으로 든 스캔 영상이다(1.5절). 물체 인식에는 도움이 될 수 있지만, 행동 판단에 도움이 되는지는 N8과 N7의 비교로만 알 수 있다.

**(C) 섞는 방식**

- 본 학습(nav 단계)의 배치에 섞는다. 정렬 단계(5.3절, N3)와는 별개다.
- 비율은 샘플 수 기준이다.
  - N7: 내비게이션 90%, 짝 맞추기 10%
  - N8: 내비게이션 80%, 짝 맞추기 10%, ScanQA 10%
- 검증은 과제별로 따로 계산한다. 내비게이션 지표(7.2절)는 내비게이션 샘플로만 계산한다.

**(D) 코드 변경 (모델 구조는 그대로)**

- `NavDataset`과 `collate`는 지금 선택지 10개와 질문 하나를 가정한다.
  - 샘플마다 과제의 질문과 선택지 수(10, 2, 4)를 받게 한다.
  - `marker_pos`를 가장 긴 것에 맞춰 채우고 `marker_mask`를 함께 넘긴다.
- `LayaNative.forward`가 `marker_mask`를 모두 1로 만드는 부분을 배치 값으로 바꾼다. `forward_embeds`는 이미 `marker_mask`를 받는다.
- `scripts/train.py`의 손실 `mask`(배치 × 10 고정)를 배치의 `marker_mask`로 바꾼다.
- 목표 분포의 길이도 과제의 선택지 수에 맞춘다. 빈 칸은 0으로 둔다.

**(E) 뒤로 미루는 것**

- **EnvDrop 렌더링**: 짝 맞추기의 효과가 확인되고, MP3D 장면에서 지시문 다양성을 더 늘리고 싶을 때 한다. NaVILA에서 EnvDrop은 지시문을 생성하는 과제인데, 학생은 문장을 생성하지 않으므로 짝 맞추기로 대신한다.
- **Human**: Go2 시연 전에 다시 판단한다(`PLAN.md` 7절).

## 5. 학습 단계와 통과 기준

### 5.1 관문 (11/13까지)

| 관문 | 내용 | 통과 기준 |
|---|---|---|
| G1 | `laya` 0.4.31 설치, 영어 체크포인트를 불러와 `cfg` 출력, 1.3절 내용 대조 | 인코더가 ModernBERT-large이고 `build_sequence`, `rlcd_loss`가 문서대로 동작 |
| G2 | `forward_embeds` 동일성 | 영상 토큰 없이 `inputs_embeds = dm.encoder.get_input_embeddings()(input_ids)`를 넣었을 때 `dm(input_ids, …)`와 logits가 허용 오차 안에서 같음 |
| G3 | 폐루프 연결(7.1절) | 정답 재생 SR 95% 이상, 학습 전 학생으로 1 에피소드 완주 |

**진행 상황 (2026-11-11)**

- G1 통과. 확인한 내용은 1.3절에 있다.
- G2 통과. 작은 무작위 모델과 실제 영어 체크포인트 모두에서 logits가 같다(`tests/test_model.py`).
- G3는 아직 하지 않았다.
- 만든 것: `lightnav/questions.py`, `data.py`, `model.py`, `metrics.py`, `scripts/train.py`, `tests/`(30개 통과).
- 아직 없는 것: `agent.py`, `server.py`, `scripts/diagnose.py`, `scripts/extract_teacher_probs.py`, `scripts/server/*.sh`, 정렬 단계(`--stage align`)의 데이터.
- 이 PC에서 실제 모델로 60 step을 돌려 확인한 것: 파라미터 516.3M, 배치 2 × 누적 4에서 step당 약 1.15초, GPU 최대 12.3 GB(gradient checkpointing 켬), 저장 후 이어서 학습됨. 60 step으로는 가장 흔한 답(전진 75 cm)만 고르는 상태라 성능에 대해서는 아무것도 말해 주지 않는다.
- 질문과 선택지 10개는 99토큰이다. `TEXT_MAX` 448에서 지시문에 348토큰이 남는다.
- 손질하지 않은 영어 체크포인트도 글만으로 방향을 읽는다. "Turn left and walk to the sofa."에 좌회전 3개가 상위 3개였고, "You have arrived. Stop here."에 정지가 1.00이었다.
- `R2R/train_sample`의 에피소드 5966은 프레임이 빠져 있다. `--only_available`이 걸러 낸다(215개 사용).

**진행 상황 (2026-11-12)**

- **서버용 코드**: `scripts/server/`(순서는 그 폴더의 README)
  - 환경 2개: 학생 `lightnav`, 교사 `navila`
  - 데이터 받기, 확인
  - 교사 라벨: 1,000개 확인, 전체 병렬 실행
  - 학습: N1, N2, N4, N5
- **교사 라벨 추출** `scripts/extract_teacher_probs.py`
  - 이 PC에서는 NaVILA 대신 가짜 모듈과 작은 llama로 끝까지 확인했다: 저장, 이어서 하기, 병합, 빠진 프레임 처리.
  - 실제 8B로는 서버의 `teacher_check.sh`가 확인한다. 처음 20개는 `model.generate` 결과와 점수 1순위가 같은지 비교한다.
- **검증 분리**: 장면 단위로 고쳤다(4절).
- **테스트**: 39개 통과, 1개 건너뜀(새로 9개: 교사 점수 계산과 추출 전체 과정 6개, 검증 분리 3개).

**G1이나 G2가 11/13까지 통과하지 못하면 `PLAN.md` 4절의 학생(mmBERT-base, `laya_base` 기반)으로 돌아간다.** G3는 어느 학생이든 필요하므로 그대로 진행한다.

### 5.2 실행 목록

**축소안 (2026-11-13, 7.2.1절 진단 뒤)**: 마감(11/26) 때문에 아래 두 실행만 본선으로 돌린다. 나머지 표는 원래 계획으로 남겨 둔다.

| 이름 | 내용 | 보려는 것 |
|---|---|---|
| A | ModernBERT-large 원본에서 시작(N2가 N1보다 나음) + 좌우 반전 0.5 + 짝 맞추기 20%, 16,000 step | 짝 맞추기로 지시문을 읽게 되는가(`text_same`) |
| B | A + 교사 분포(alpha 0.5) | 증류의 이득 |
| C | A + 지시문 대조 손실(7.2.2절, 2026-11-13 추가) | 이동 판단이 지시문을 쓰게 되는가 |

- 짝 맞추기 비율은 설계의 10%가 아니라 20%로 둔다. 이동 판단 라벨에는 지시문 신호가 없어서, A에서는 짝 맞추기가 유일한 지시문 신호이기 때문이다.
- 16,000 step은 짝 맞추기를 넣은 1.1 epoch이다. N1·N2의 검증 성능은 약 10k step에서 가장 좋았다.
- ScanQA(N8)는 넣지 않는다. 장면 영상에서 프레임을 뽑는 작업이 아직 없다.
- 첫 검증(2,000 step)에서 `text_same`이 0.9 이상이면 그 실행은 지시문을 읽지 않는 것이므로 멈추고 비율을 다시 정한다.

**원래 계획**

| 이름 | 내용 | 보려는 것 |
|---|---|---|
| N1 | Laya 체크포인트에서 시작, 10지선다, 정답 라벨, 정렬 없음 | 기준 |
| N2 | N1과 같고 `answerdotai/ModernBERT-large` 원본에서 시작 | Laya 가중치를 물려받는 이득 |
| N3 | N1 앞에 정렬 단계(5.3절)를 둠 | 정렬 단계의 이득 |
| N4 | 앞의 가장 좋은 것 + 좌우 반전 | 직전 행동 베끼기, 방향어 이해 |
| N5 | N4 + 교사 분포 | 증류 효과 |
| N6 | N4를 질문 3개로 (8절) | 질문 분해의 이득 |
| N7 | N4 + 짝 맞추기 10% (4.1절 (A)) | 짝 맞추기(랜드마크 교체)의 이득 |
| N8 | N7 + ScanQA 10% (4.1절 (B)) | ScanQA의 이득 |

- N1, N2, N3은 교사 라벨을 기다리지 않고 함께 시작한다. 학습 스텝, 데이터, 검증 분리는 모두 같게 둔다.
- **N7·N8 전에 진단을 먼저 한다.**
  - N1이 나오면 "물체만 바꾼 지시문" 진단(7.2절)으로, 물체 이름에 대한 반응이 약한지 학습 없이 잰다.
  - "물체 이름 반응이 약함"(KIMM TVD 0.041)은 이전 LayaNav v3에서 잰 값이다. 이 학생에서는 아직 재지 않았다.
  - 판단이 물체 교체에 충분히 반응하면 N7·N8의 우선순위를 낮춘다.
- N7과 N8은 같은 학습 스텝으로 돌린다.
  - N7과 N4를 비교하면 짝 맞추기의 효과가, N8과 N7을 비교하면 ScanQA의 효과가 보인다.
  - 판정은 폐루프 SR로 한다(7.3절).
- 학습률은 encoder 2.5e-5, head·투영 1e-4, vision 1e-5로 시작한다.
- N2는 head, scorer, `type_emb`가 무작위 초기화다. `laya` 패키지로 원본 인코더에서 새 `DecisionModel`을 만드는 방법은 G1에서 확인한다.
- N2가 N1과 같거나 더 좋으면 Laya 형식을 지킬 이유가 없어진다. 그때는 선택지를 열의 끝으로 옮기는 실험(10절)을 먼저 한다.
- `PLAN.md`의 학습 시간 추정(1 epoch 약 5시간)은 mmBERT-base, 430토큰 기준이라 이 모델에는 맞지 않는다. 서버에서 1,000 step을 돌려 다시 잰다.

### 5.3 정렬 단계 (N3에서만)

Laya는 영상 토큰을 본 적이 없으므로, 투영층과 프레임·위치 임베딩만 먼저 학습시킨다. Laya와 SigLIP2는 얼린다.

- **(가) 직전 행동 맞히기**: 연속한 두 판단의 현재 프레임 2장을 주고 "What did the robot just do?" → `moved forward / turned left / turned right`. 정답은 앞 판단의 답 문장에서 나온다(1.1절).
- **(나) 지시문-영상 짝 맞추기**: 프레임 8장과 지시문을 주고 "Does the instruction describe this route?" → yes/no. 샘플은 4.1절 (A)와 같은 방법으로 만든다(랜드마크 교체 포함).

과제 (가)의 검증 정확도가 우연 수준(33%)에 머물면 head 2층도 푼다. N3이 N1보다 폐루프에서 낫지 않으면 정렬 단계를 버린다.

## 6. 교사 라벨

절차와 환경은 `PLAN.md` 5절을 따른다. 이 설계에서 지켜야 할 것만 적는다.

- 답 문장 10개는 3.5절의 `ANSWERS`를 쓴다. 정지 문장은 형식이 다르다.
- 교사에게는 NaVILA 원래 방식대로 검은 이미지를 채운 8장을 준다. 검은 프레임을 빼는 것은 학생 쪽만이다.
- 복제를 뺀 고유 샘플(R2R + RxR 679,532개)에 한 번만 붙인다. 키는 `<데이터>:<video_id>`다(예: `r2r:4172-16`). R2R과 RxR의 `video_id`가 겹칠 수 있어서다.
- 저장은 `teacher_probs.npy` [N, 10]과 `keys.txt`다.
- 교사 1순위가 정답과 같은 비율을 본다. 교사가 학습한 데이터이므로 높아야 한다. 낮으면 프롬프트나 전처리가 틀린 것이다.
- 교사가 자기 학습 데이터에 답하는 것이라 확률이 정답 하나에 쏠릴 수 있다. 정답이 아닌 선택지에 실린 확률의 평균을 함께 기록한다. 이 값이 매우 작으면 N5의 이득은 기대하기 어렵다.

## 7. 평가

### 7.1 폐루프 (이 PC)

2026-11-13에 만들고 점검했다. 사용법과 점검 결과는 `scripts/eval/README.md`에 있다.

- **평가 루프** `scripts/eval_vlnce.py`
  - `navila_trainer.py`의 루프를 그대로 옮겼다: 저수준 행동마다 프레임 한 장, 8장 고르기, 답 해석, 행동 대기열.
  - 설정 yaml, VLN-CE 측정, 데이터셋 코드는 NaVILA 것을 불러 쓴다. NaVILA 코드는 고치지 않는다.
  - 8B 호출 자리에서 학생 프로세스(`lightnav/server.py`)에 HTTP로 묻는다.
    - 요청: 지시문, 8장 중 실제 프레임(원본 RGB)과 그 위치
    - 응답: `ANSWERS` 중 한 문장, 확률 10개, 걸린 시간(ms)
- **환경** `env_vlnce`(`scripts/eval/setup_vlnce_env.sh`)
  - Python 3.8, 미리 빌드된 habitat-sim 0.1.7 headless, habitat-lab v0.1.7(서브모듈)
  - numpy는 1.23.5로 고정한다. habitat-sim 0.1.7이 `np.float`를 쓴다.
- **장면**: Scene-N1의 `mp3d_ce`(90개)를 쓴다. R2R 1,839개 중 97.6%의 측지 거리가 VLN-CE 파일과 1 cm 안으로 같다.
- **정답 재생 확인(G3)**: 통과했다.
  - R2R 50개: SR 100.0, nDTW 100.0
  - RxR 50개: SR 94.0, nDTW 98.0. 실패 3개는 목표까지 경로가 없는 에피소드다.
  - **RxR 정답 행동은 30° 회전 기준이다.** 15°로 재생하면 SR이 10%로 떨어진다. G3는 30°로 확인했고, 학생 평가는 NaVILA와 같게 15°로 한다.
- **하한선** "전진 75 cm"만: R2R 100개에서 SR 0.0, OS 3.0
- **판단 시간**: 학습 전 학생이 RTX 5060 Ti에서 중앙값 55.5 ms다. 이미지 8장을 매번 모두 인코딩한 값이다.
- **지표**: SR, SPL, NE, OS, nDTW, TL, 판단 1회 시간(ms), 에피소드당 판단 횟수, GPU 메모리, 파라미터 수. 요약과 쌍 비교는 `scripts/eval_summary.py`로 한다.

### 7.2 오프라인 진단 (`scripts/diagnose.py`, 서버)

- 10지선다 정확도, 종류(정지/전진/좌/우) 정확도
- 정지 정밀도·재현율, 전진 25/50/75 cm 혼동표, 회전 15/30/45° 혼동표
- 교사 일치율(N5부터), ECE
- 지름길 검사: 지시문의 left·right 바꾸기, 좌우 반전 샘플의 일치율, 지시문을 다른 에피소드 것으로 바꿨을 때의 정확도 하락
- **물체만 바꾼 지시문**: 랜드마크 하나를 다른 물체로 바꿨을 때(4.1절 (A)의 교체 규칙) 판단이 바뀌는 비율과 정확도 하락. N1부터 잰다.
- 과제별 검증 정확도: 짝 맞추기(진짜/가짜, 가짜는 랜드마크 교체와 다른 에피소드로 나눠서), ScanQA. N7부터 잰다.
- 학습 정확도와 검증 정확도의 차이

### 7.2.1 결과: N1·N2 (2026-11-13, 서버에서 약 14k step에 멈춘 체크포인트)

**결론: 두 모델 모두 지시문을 읽지 않는다.** N2는 화면만 보고 움직이는 시각 정책이 되었고, N1은 학습 데이터도 맞추지 못한 학습 부족 상태다.

이 PC의 `R2R/train_sample`에서 검증 장면 샘플 301개와 학습 장면 샘플 1,000개로 쟀다(`scripts/diagnose.py`, 결과 `runs/diag/local.json`). "같은 답"은 원래 입력일 때와 예측이 같은 비율이다.

| 변형 | N2 정확도 / 종류 | N2 같은 답 | N1 정확도 / 종류 | N1 같은 답 |
|---|---|---|---|---|
| 원래 입력 | 0.442 / 0.618 | – | 0.369 / 0.558 | – |
| 지시문을 다른 샘플 것으로 | 0.442 / 0.615 | 0.97 (KL 0.007) | 0.359 / 0.548 | 0.91 (KL 0.026) |
| 프레임 모두 검정 + 지시문 바꾸기 (검정 대비) | 0.362 / 0.585 | 0.99 (KL 0.005) | 0.355 / 0.585 | 0.98 (KL 0.009) |
| 지시문의 left/right만 바꾸기 | 0.437 / 0.599 | 0.95 | 0.376 / 0.569 | 0.92 |
| 프레임을 다른 샘플 것으로 | 0.223 / 0.389 | 0.31 | 0.256 / 0.422 | 0.50 |
| 프레임만 좌우 반전 | 0.355 / 0.478 | 0.53 (회전 방향이 뒤집힘) | 0.379 / 0.571 | 0.91 |
| 학습 장면 1,000개 | 0.522 / 0.652 | | 0.381 / 0.562 | |
| 기준: 항상 전진 75 cm | 0.355 / 0.585 | | | |

- **학습 전 Laya 체크포인트는 지시문에 크게 반응한다.** 지시문을 바꾸면 KL 0.37이고 답의 77%가 바뀐다. 구조가 지시문을 막고 있는 것이 아니라, 학습이 지시문을 쓰지 않는 쪽으로 갔다.
- **원인: R2R의 정답 라벨에는 지시문 정보가 없다.**
  - R2R 학습 경로 3,603개는 모두 지시문 3개(10개는 4개)로 따로 렌더링되어 있다.
  - 같은 경로의 에피소드들은 프레임과 정답 동작 열이 100% 같다(3,603개 중 3,603개).
  - 정답 라벨만으로 학습하면 지시문을 읽어서 줄일 수 있는 손실이 없다.
- 정지 문턱(`tau_stop`)과 복제 보정은 정확도를 0.01 안쪽으로만 바꾼다. 정지는 주 원인이 아니다.
- 직전 동작 종류만 알면 이번 동작 종류를 61% 맞힌다(R2R 전체). 라벨이 전진과 회전을 자주 오간다(직전과 같은 종류 37%). 그래서 종류 정확도는 "항상 전진"(0.585)보다 크게 높아야 의미가 있다.

**폐루프 (R2R val_unseen 100 에피소드, 하한선과 같은 에피소드)**

| 모델 | SR | SPL | NE | OS | nDTW | 판단 시간 중앙값 |
|---|---|---|---|---|---|---|
| 전진 75 cm만 | 0.0 | 0.0 | 10.56 | 3.0 | 25.1 | – |
| N2 | 8.0 | 7.3 | 9.97 | 12.0 | 38.1 | 54.8 ms |

- 하한선 대비 SR +8.0 (95% CI +3.0 ~ +14.0)
- 88개 에피소드가 스스로 멈췄다. 멈춘 에피소드의 저수준 step 중앙값은 33, 경로 길이 중앙값은 5.6 m다. 목표에 닿기 전에 일찍 멈추는 경우가 대부분이다.
- 지시문의 끝 지점을 모르는 시각 정책이라는 진단과 맞는다.

**다음 실험에 주는 뜻**

- 좌우 반전(N4)만으로는 고쳐지지 않는다. 지시문과 프레임을 함께 뒤집으므로, 여전히 프레임만 봐도 답이 정해진다.
- 지시문에 학습 신호를 주는 것이 필요하다.
  - **교사 분포(N5)**: NaVILA는 지시문을 읽으므로, 같은 경로라도 지시문마다 분포가 다르다. 정답 라벨에 없는 신호다.
  - **짝 맞추기(N7, 4.1절 (A))**: 지시문과 경로가 맞는지를 직접 묻는다.
- 새 실행마다 "지시문 바꾸기"의 같은 답 비율을 첫 검증에서 본다. 0.9 이상이면 지시문을 쓰지 않는 것이므로 오래 돌리지 않는다.
- 서버 검증 5,000개(R2R + RxR)로 다시 재려면 `bash scripts/server/diagnose.sh N1 N2`를 쓴다.

### 7.2.2 결과: A의 첫 검증 (2026-11-13, 서버 4,000 step)

**결론: 짝 맞추기 20%를 넣어도 4,000 step까지는 지시문을 읽지 않는다.** 짝 맞추기 자체도 우연 수준이다.

| step | 검증 정확도 / 종류 | 정지 재현율 | 지시문 바꾸기 같은 답 | 짝 맞추기 (참 / 랜드마크 교체 / 다른 경로) |
|---|---|---|---|---|
| 2,000 | 0.268 / 0.505 | 0.00 | 0.88 | 0.502 |
| 4,000 | 0.275 / 0.519 | 0.00 | 0.96 | 0.501 (0.44 / 0.55 / 0.57) |
| N2, 2,000 | 0.300 / 0.521 | 0.00 | – | – |
| N2, 4,000 | 0.337 / 0.522 | 0.52 | – | – |

- 같은 시작점(ModernBERT)의 N2보다 같은 step에서 정확도와 정지 재현율이 모두 낮다. 짝 맞추기와 좌우 반전이 이동 판단 학습을 늦춘다. 둘 중 무엇 때문인지는 가리지 못했다.
- 지시문 사용은 학습이 진행될수록 줄었다(같은 답 0.88 → 0.96).
- A는 5,000 step 근처에서 멈추고, C를 짝 맞추기 없이(`MATCH=0`) 돌리기로 했다.

- **구현 문제는 아니다.** 이 PC에서 16개 경로로 과적합 실험을 했다(ModernBERT 시작, vision 고정, 300 step).
  - 각 지시문이 자기 경로와 한 번(맞음), 다른 경로와 한 번(틀림) 나온다. 글자만 보거나 화면만 봐서는 풀 수 없다.
  - 정확도는 75 step까지 0.5 근처에 머물다가 100 step에 0.88, 300 step에 0.94가 되었다. 지시문과 프레임을 함께 쓰는 경로가 동작한다.
- 남은 가능성은 둘이다. 서버에서 따로 잰다.
  - 학습 경로는 맞히지만 새 장면으로 옮겨 가지 않는다. 이 경우 단어와 영상의 대응을 미리 주는 신호가 필요하다.
  - 학습 경로도 아직 못 맞힌다. 이 경우 신호가 약하거나 학습이 덜 된 것이다.
  - 측정: `bash scripts/server/match_eval.sh A` (`scripts/match_eval.py`, 학습 경로 1,000개와 검증 경로 1,213개)
- **B가 도움이 될지 먼저 잰다.** 교사 라벨 중 같은 경로의 지시문이 둘 이상 라벨된 결정에서, NaVILA 분포가 지시문마다 얼마나 다른지 본다.
  - 측정: `bash scripts/server/teacher_signal.sh` (`scripts/teacher_signal.py`). 비교 기준은 같은 정답 라벨을 가진 다른 경로의 결정 쌍이다.
- **대안 C: 지시문 대조 손실** (`--contrast`, 2026-11-13 구현)
  - 이동 샘플의 절반(`--contrast_prob`)을 다른 경로의 지시문으로 한 번 더 돌린다. 프레임 특징은 다시 계산하지 않는다.
  - 손실: `relu(margin - (log p(라벨 | 자기 지시문) - log p(라벨 | 다른 지시문)))`, margin 0.5 nat, 가중치 1.0
  - 정답 라벨에 지시문 신호가 없다는 문제를 손실에서 직접 메운다. 로그의 `contrast_gap`이 0보다 커지면 지시문을 쓰기 시작한 것이다.

### 7.3 판정

- 최종 비교는 R2R val_unseen 전체 1,839 에피소드로 한다. 이때 SR의 표준오차는 약 1.2%p이므로 2~3%p 미만 차이는 잡음으로 본다.
- 100 에피소드 중간 점검(`PLAN.md` 11/29)은 표준오차가 약 5%p다. 순위를 가리는 데 쓰지 말고 동작 확인과 큰 차이 확인에만 쓴다.
- 두 모델 비교는 같은 에피소드에서 쌍으로 한다(부트스트랩).
- 온도 보정은 최종 모델에서만 한다(`laya.calibrate.fit_temperature_map`, 검증 데이터).

### 7.4 최종 비교표

| 모델 | SR | SPL | NE | OS | 판단 시간 | 파라미터 |
|---|---|---|---|---|---|---|
| NaVILA 8B (서버에서 재현되면 재현값, 아니면 논문 값) | | | | | | |
| N1 Laya 체크포인트 | | | | | | |
| N2 ModernBERT-large 원본 | | | | | | |
| N7 짝 맞추기 추가 | | | | | | |
| N8 짝 맞추기 + ScanQA 추가 | | | | | | |
| 최종 (N4, N5, N7, N8 중 가장 좋은 것) | | | | | | |
| N6 질문 3개 | | | | | | |

## 8. 비교 실험 N6: 질문 3개

```python
QUESTIONS_3 = {
    "arrived": {
        "type": "choice",
        "instructions": "Has the robot finished the instruction and reached the goal?",
        "criteria": {"A": "yes, the robot has arrived and should stop",
                     "B": "no, the robot should keep moving"},
    },
    "direction": {
        "type": "choice",
        "instructions": "Which way should the robot move next?",
        "criteria": {"forward": "move straight ahead", "left": "turn left", "right": "turn right"},
    },
    "amount": {
        "type": "score",
        "instructions": "How large should that move be?",
        "criteria": ["small: 25 cm forward or a 15 degree turn",
                     "medium: 50 cm forward or a 30 degree turn",
                     "large: 75 cm forward or a 45 degree turn"],
    },
}
```

- `direction`과 `amount`는 정지가 아닌 샘플에서만 손실을 건다.
- 추론: `P(arrived = A) > τ_stop`이면 정지, 아니면 방향과 크기를 각각 argmax로 고른다.
- 본선에서 내린 이유
  - 판단 한 번에 열이 3개라 인코더 비용이 약 3배다.
  - 방향과 크기를 따로 뽑으면 둘 사이의 관계가 사라진다. 교사가 "전진 75 cm 또는 좌회전 15°"에 확률을 나눠 줬을 때, 곱으로 합치면 "전진 25 cm"와 "좌회전 45°"에도 확률이 생긴다.
  - `amount`가 Laya에서 가장 약하다는 score 유형이다.
- 판단 1회 시간을 본선과 함께 잰다.

## 9. 테스트 (`tests/`)

작은 무작위 `DecisionModel`과 작은 영상 인코더로 CPU에서 돈다.

1. `forward_embeds` 동일성(G2). 실제 체크포인트로도 한 번 확인한다.
2. 영상 토큰을 끼워도 `markers` 위치가 그대로이고 마지막 토큰이 `[SEP]`임.
3. 질문과 선택지 10개가 `HEAD` 안에 잘리지 않고 들어감. 선택지 순서가 `ANSWERS`와 같음.
4. 프레임 선택: 프레임 수 1, 5, 8, 9, 100에서 NaVILA의 `vlnce_frame_sampling`과 같은 위치를 고르고, 검은 이미지 위치는 mask로 빠짐.
5. 답 문장 ↔ 번호 변환이 `R2R/annotations.json`의 모든 `a` 값에서 왕복됨. NaVILA 해석 정규식이 `ANSWERS` 10개를 올바른 행동으로 읽음.
6. 좌우 반전: 정답 번호 4~6과 7~9가 맞바뀌고 정지·전진은 그대로임.
7. 상태가 `TEXT_MAX`를 넘을 때 지시문이 잘리는 비율을 세는 함수(RxR은 지시문이 길다).
8. 정렬 단계에서 Laya와 SigLIP2의 파라미터에 기울기가 없음.
9. 정렬 과제 (가)의 라벨이 앞 판단의 답 문장과 맞음.
10. 짝 맞추기: 랜드마크 교체 지시문이 원래 지시문과 물체 하나만 다르고, 구조물 단어는 바뀌지 않음. 검증 에피소드의 지시문이 학습 가짜 짝에 쓰이지 않음.
11. ScanQA 4지선다: 오답이 정답 목록과 겹치지 않고 같은 질문 유형에서 나옴. 오답끼리 철자 변형(grey/gray)으로 겹치지 않고, 정답을 포함하거나 정답에 포함되는 오답이 없음. 순서를 섞어도 정답 번호가 맞음. `Answer briefly.`가 지워짐.
12. 선택지 수가 다른 샘플(10, 2, 4)을 한 배치에 넣었을 때 빈 선택지의 logits가 막히고 손실에 들어가지 않음.
13. 내비게이션 지표가 내비게이션 샘플로만 계산됨.
14. RxR 검증 분리가 장면 단위로 됨(RxR-CE 에피소드 파일을 줄 때).

## 10. 선택 실험 (본 작업 뒤)

- **선택지를 열의 끝으로 옮기기**: N2가 N1 이상일 때. 선택지와 현재 프레임이 가까워진다.
- **DINOv2 결합**: 현재 프레임에 `geo_proj(dino)`를 더한다. `facebook/dinov2-with-registers-base`, 196 px 입력, ImageNet 정규화, register 토큰 제거, 마지막 층 0 초기화. 전진 거리 혼동표를 본다.
- **근거 연결**: `laya_base`의 `grounding.py`를 현재 프레임에 적용한다(`PLAN.md` 비교군 D).
- **손실**: `laya.train.soft_ce_loss`만 쓴 경우.
- **EnvDrop·Human 추가**: 4.1절 (E).
- **진행도 보조 질문**: `판단 번호 / (에피소드의 판단 수 − 1)`을 5구간으로 나눈 score 질문을 학습에만 더한다. 학습 때 열이 2배가 된다.
- **DAgger**: 학생이 직접 주행한 상태에 라벨을 붙여 재학습한다. 라벨은 교사 또는 Habitat 최단 경로 행동을 3개까지 묶은 것. 이 PC에서 모은 프레임을 서버로 보내는 방법이 없어서(11절) 지금은 할 수 없다.

## 11. 정할 것과 확인하지 못한 것

**정할 것**

- 이 PC에서 서버로 데이터를 보내는 방법(DAgger에 필요). 비공개 Hugging Face 데이터셋에 올리고 서버에서 받는 방법이 가능해 보인다.
- `PLAN.md` 1절(학생 크기 0.45B, mmBERT-base)과 4절을 이 문서에 맞춰 고칠지.

**확인하지 못한 것**

- 1.3절의 Laya 내용 전부와 `rl_agent_config.json` 값. G1에서 확인한다.
- 랜드마크 추출(`split_instruction`, `landmark`)이 RxR의 긴 지시문에서도 쓸 만한 물체 이름을 뽑는지.
- ModernBERT는 층 3개마다 전역 attention이고 나머지는 국소 attention(창 128토큰)인 것으로 기억한다. 선택지는 열의 앞, 영상은 뒤에 있어서 전역 층에서만 서로를 본다. 성능이 낮으면 먼저 의심할 곳이다.
- Laya 체크포인트는 `max_len` 512로 학습됐고 이 설계의 열은 약 990토큰이다. 더 긴 열에서 문제가 없는지는 돌려 봐야 안다.
- Laya 체크포인트에서 시작하는 것이 이득이라는 보장은 없다. 영상 토큰은 Laya가 학습한 분포 밖이다. N2가 이것을 가른다.
- 파라미터 수(약 0.5B)와 속도는 추정이다. 구현 후에 잰다. 이 설계는 실행해 본 적이 없다.
