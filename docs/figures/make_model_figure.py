"""Draws the student of run D, which the L experiments (L0_r2r, L1_dagger1, L2_teacher) train further.

    python3 docs/figures/make_model_figure.py docs/figures/lightnav_model_D_ko.svg

Same palette and layout grid as docs/figures/lightnav_model_ko.svg (the N1/N2 model of 2026-11-13). Numbers are
from runs/D (run.json, lightnav_config.json, parameter counts of model.safetensors) and runs/eval/D. PNG and PDF are
made from the SVG with headless Chrome:

    google-chrome --headless=new --disable-gpu --no-sandbox --hide-scrollbars --allow-file-access-from-files \
        --force-device-scale-factor=2 --window-size=1600,1010 --screenshot=<png> file://<html showing the svg>
"""
import sys

FONT = "Noto Sans CJK KR, Liberation Sans, sans-serif"
MONO = "Liberation Mono, Noto Sans Mono CJK KR, monospace"
W, H = 1600, 1010

INK, GRAY, LINE = "#1F2328", "#57606A", "#8C959F"
TOKEN = ("#F3F4F6", "#8C959F")          # text tokens
VISION = ("#E7F0FA", "#2F6DB5")         # pretrained vision encoder (SigLIP2)
CUR = "#BCD3EE"                         # current-frame tokens
LANG = ("#E8F3EC", "#2E7D4F")           # language encoder
CHOICE = ("#F0ECF8", "#6A4C9C")         # Laya head, scorer, [M] markers
NEW = ("#FFF0DE", "#C8741E")            # parts added for images and tags
TRAIN = ("#FCF1EF", "#B5473A")          # training only

out = []
warnings = []


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def width(s, size):
    w = 0.0
    for c in s:
        if ord(c) >= 0x1100:
            w += 0.95 * size
        elif c == " ":
            w += 0.28 * size
        elif c.isupper() or c.isdigit():
            w += 0.62 * size
        else:
            w += 0.52 * size
    return w


def text(x, y, s, size=13, weight="normal", fill=INK, anchor="start", style="normal", font=FONT, maxw=None):
    if maxw is not None and width(s, size) > maxw:
        warnings.append("%r: about %d px > %d px" % (s, width(s, size), maxw))
    out.append('<text x="%s" y="%s" font-family="%s" font-size="%s" font-weight="%s" font-style="%s" fill="%s" '
               'text-anchor="%s">%s</text>' % (x, y, font, size, weight, style, fill, anchor, esc(s)))


def rect(x, y, w, h, fill="#FFFFFF", stroke=LINE, sw=1.5, rx=8, dash=None):
    d = ' stroke-dasharray="%s"' % dash if dash else ""
    out.append('<rect x="%s" y="%s" width="%s" height="%s" rx="%s" fill="%s" stroke="%s" stroke-width="%s"%s />'
               % (x, y, w, h, rx, fill, stroke, sw, d))


def arrow(points, color=LINE, marker="ah", sw=1.5):
    d = "M" + " L".join("%s,%s" % p for p in points)
    out.append('<path d="%s" fill="none" stroke="%s" stroke-width="%s" marker-end="url(#%s)" />' % (d, color, sw, marker))


def brace(x1, x2, y, label, size=12):
    """A bracket over a stretch of the input row, with its label above."""
    out.append('<path d="M%s,%s L%s,%s L%s,%s L%s,%s" fill="none" stroke="%s" stroke-width="1.2" />'
               % (x1, y + 6, x1, y, x2, y, x2, y + 6, LINE))
    text((x1 + x2) / 2, y - 6, label, size, fill=GRAY, anchor="middle", maxw=max(60, x2 - x1 + 40))


def box(x, y, w, h, title, lines, colors=("#FFFFFF", LINE), title_size=15, size=13, dash=None, gap=22,
        title_fill=INK, line_fill=GRAY, italic=(), bold=()):
    rect(x, y, w, h, colors[0], colors[1], 1.6, 8, dash)
    text(x + 16, y + 26, title, title_size, "bold", title_fill, maxw=w - 28)
    for i, s in enumerate(lines):
        text(x + 16, y + 26 + gap * (i + 1) + 2, s, size, "bold" if i in bold else "normal",
             INK if i in bold else line_fill, style="italic" if i in italic else "normal", maxw=w - 28)


def frame(x, y, s):
    rect(x, y, s, s, "#DCE3EA", "#7B8794", 1, 3)
    out.append('<polygon points="%s,%s %s,%s %s,%s %s,%s" fill="#C3CCD6"/>'
               % (x + 1, y + s - 1, x + s - 1, y + s - 1, x + s * 0.74, y + s * 0.58, x + s * 0.28, y + s * 0.58))
    rect(x + s * 0.68, y + s * 0.24, s * 0.16, s * 0.34, "#8D99A6", "none", 0, 1)


# ------------------------------------------------------------------------------------------------ canvas
out.append('<rect x="0" y="0" width="%d" height="%d" fill="#FFFFFF" />' % (W, H))
text(30, 30, "추론: 학생 모델 혼자 주행 (실행 D의 구조 · L0, L1, L2도 같은 구조)", 13, "bold", GRAY)

# instruction
box(30, 46, 300, 118, "지시문", ["“Walk past the sofa, then", "stop at the bedroom door.”"], title_size=16,
    size=15, italic=(0, 1), gap=22)
text(46, 150, "NaVILA 프롬프트와 같은 영어 문장", 12, fill=GRAY)

# frames
text(360, 62, "관측: 프레임 8장 (NaVILA와 같은 방식으로 고름)", 16, "bold")
for i in range(7):
    frame(360 + 62 * i, 76, 54)
text(573, 152, "과거 프레임 7장, 처음부터 고르게", 13, fill=GRAY, anchor="middle")
frame(808, 70, 92)
text(854, 180, "현재 프레임", 13, "bold", anchor="middle")

# Laya question format (left column)
box(30, 200, 300, 150, "Laya 질문 형식", ["ModernBERT 토크나이저, 최대 448토큰", "질문 · 선택지 10개 · 상태 {지시문}",
                                       "선택지마다 [M] 마커 하나", "양방향 주의: 순서와 무관하게 서로 봄"],
    colors=("#F6F8FA", LINE), size=12, gap=24)
arrow([(180, 164), (180, 198)])

# vision side
rect(348, 192, 590, 202, "#FFFFFF", VISION[1], 1.4, 12, "6 4")
text(362, 212, "영상 쪽 (영상 인코더는 고정)", 13, "bold", VISION[1], maxw=220)
box(362, 222, 250, 84, "SigLIP2-B/16 영상 인코더", ["고정 · 224 px · 92.9M", "학습은 저장 특징으로 (siglip2_g2)"],
    colors=VISION, size=12, gap=20)
rect(632, 222, 292, 38, "#FFFFFF", VISION[1], 1.3, 6)
text(778, 246, "현재: 14×14 → 평균 2×2 = 4 토큰", 13, anchor="middle", maxw=280)
rect(632, 268, 292, 38, "#FFFFFF", VISION[1], 1.3, 6)
text(778, 292, "과거: 평균 1×1 = 1 토큰 × 7장", 13, anchor="middle", maxw=280)
arrow([(612, 252), (630, 241)])
arrow([(612, 276), (630, 287)])
box(362, 318, 562, 64, "랜드마크 태그: 현재 10단어, 과거 5단어씩",
    ["SigLIP2 요약 임베딩 ↔ 랜드마크 133개의 글 임베딩 표, 단어별 점수 정규화 → 상위 단어"],
    colors=("#FFFFFF", VISION[1]), title_size=14, size=12, gap=22)
arrow([(573, 160), (573, 220)])
arrow([(808, 150), (612, 228)])

# projection (new parts)
box(960, 222, 160, 160, "투영 MLP", ["768 → 1024 → 1024", "+ 프레임 임베딩", "+ 위치 임베딩", "+ 태그 종류(현재·과거)",
                                    "신규 1.9M"],
    colors=NEW, size=12, gap=21)
arrow([(924, 264), (958, 264)])

# ------------------------------------------------------------------------------------------------ input row
text(30, 428, "입력 순서", 15, "bold")
y0, hh = 468, 46
x = 30
segs = {}


def seg(w, kind, label="", font=FONT, size=12, key=None):
    global x
    fill, stroke = {"text": TOKEN, "special": ("#FFFFFF", LINE), "img": (VISION[0], VISION[1]),
                    "cur": (CUR, VISION[1]), "opt": ("#FFFFFF", CHOICE[1]), "mark": (CHOICE[1], CHOICE[1]),
                    "tag": (NEW[0], VISION[1])}[kind]
    rect(x, y0, w, hh, fill, stroke, 1.3, 4)
    if label:
        text(x + w / 2, y0 + hh / 2 + 5, label, size, "normal", INK, "middle", font=font, maxw=w - 4)
    if key:
        segs[key] = (x, x + w)
    x += w + 4


seg(40, "special", "[CLS]", MONO, 10)
seg(52, "text", "질문", size=12, key="q")
seg(40, "special", "[SEP]", MONO, 10)
x0 = x
for i in range(10):
    seg(12, "mark")
    seg(16, "opt")
segs["opt"] = (x0, x - 4)
seg(40, "special", "[SEP]", MONO, 10)
seg(130, "text", "{\"instruction\": …}", MONO, 10, key="ins")
seg(100, "tag", "seen: 태그", size=11, key="tag")
x0 = x
for i in range(7):
    seg(14, "img")
for i in range(4):
    seg(20, "cur")
segs["img"] = (x0, x - 4)
seg(40, "special", "[SEP]", MONO, 10)
row_end = x - 4

brace(segs["opt"][0], segs["opt"][1], y0 - 12, "선택지 10개, 각각 [M] 마커 하나")
brace(segs["ins"][0], segs["ins"][1], y0 - 12, "상태 = 지시문")
brace(segs["tag"][0], segs["tag"][1], y0 - 12, "태그 단어")
brace(segs["img"][0], segs["img"][1], y0 - 12, "영상 토큰 11개: 과거 7 + 현재 4")
text(segs["tag"][0] - 40, y0 + hh + 18, "태그는 보통 단어 임베딩: 지시문의 \"bathroom\"과 같은 토큰", 11, fill=GRAY)
text(segs["img"][1], y0 + hh + 18, "투영 MLP의 출력", 11, fill=GRAY, anchor="end")
text(segs["q"][0], y0 + hh + 18, "N1·N2는 영상 토큰 539개 → 지시문을 안 읽음. D는 11개 + 태그", 11, fill=GRAY)
arrow([(1040, 382), (1040, 400), ((segs["img"][0] + segs["img"][1]) / 2 + 20, 400),
       ((segs["img"][0] + segs["img"][1]) / 2 + 20, y0 - 30)], VISION[1], "ahb")
arrow([((segs["tag"][0] + segs["tag"][1]) / 2, 382), ((segs["tag"][0] + segs["tag"][1]) / 2, y0 - 30)],
      VISION[1], "ahb")
arrow([(180, 350), (180, 410), (segs["q"][0] + 26, 410), (segs["q"][0] + 26, y0 - 2)])

# ------------------------------------------------------------------------------------------------ ModernBERT
ly = 546
rect(30, ly, 1090, 112, LANG[0], LANG[1], 1.8, 10)
text(50, ly + 28, "ModernBERT-large 인코더", 18, "bold")
text(296, ly + 28, "28층 · d = 1024 · RoPE · 394.8M · 사전학습 원본에서 시작 (Laya 체크포인트 아님)", 13,
     fill=GRAY, maxw=800)
for i in range(28):
    glob = i % 3 == 0
    rect(50 + 29.5 * i, ly + 42, 25, 40, LANG[1] if glob else "#FFFFFF", LANG[1], 1.3, 3)
text(54, ly + 100, "0", 11, fill=GRAY)
text(50 + 29.5 * 27 + 4, ly + 100, "27", 11, fill=GRAY)
rect(900, ly + 46, 16, 16, LANG[1], LANG[1], 1.2, 3)
text(924, ly + 59, "전역 주의 (10층)", 13)
rect(900, ly + 70, 16, 16, "#FFFFFF", LANG[1], 1.2, 3)
text(924, ly + 83, "국소 창 ±64 (18층)", 13)
arrow([(560, y0 + hh + 26), (560, ly - 2)])

# ------------------------------------------------------------------------------------------------ head, scorer
hy = 676
rect(30, hy, 1090, 44, CHOICE[0], CHOICE[1], 1.6, 8)
text(50, hy + 28, "Laya 판단 헤드", 16, "bold")
text(200, hy + 28, "Transformer 2층 · 전체 주의 · 16헤드 · 25.2M · [M] 위치 10곳에서만 읽음 · D에서 새로 학습", 13,
     fill=GRAY, maxw=900)
arrow([(560, ly + 112), (560, hy - 2)])
for i in range(10):
    xx = 210 + 30 * i
    arrow([(xx, hy + 44), (xx, hy + 66)], CHOICE[1], "ahp", 1.3)
rect(150, hy + 68, 400, 34, "#FFFFFF", CHOICE[1], 1.5, 8)
text(350, hy + 90, "채점 MLP: [M]마다 점수 1개 · 1.05M", 13, anchor="middle")

# bar chart
cy = hy + 102
text(440, cy + 26, "NaVILA 답 10개의 확률 (softmax)", 14, "bold", anchor="middle")
base = cy + 112
probs = [0.04, 0.05, 0.04, 0.12, 0.14, 0.62, 0.08, 0.04, 0.04, 0.03]
labels = ["정지", "전25", "전50", "전75", "좌15", "좌30", "좌45", "우15", "우30", "우45"]
for i, (p, lab) in enumerate(zip(probs, labels)):
    bx = 250 + 40 * i
    hgt = 70 * p / max(probs)
    top = i == 5
    rect(bx, base - hgt, 26, hgt, CHOICE[1] if top else "#CFC6E2", "none", 0, 1)
    text(bx + 13, base + 16, lab, 12, "bold" if top else "normal", INK if top else GRAY, "middle")
out.append('<line x1="244" y1="%s" x2="652" y2="%s" stroke="%s" stroke-width="1.2" />' % (base, base, LINE))
text(660, base + 16, "전 = 전진 cm, 좌·우 = 회전 각도", 11, fill=GRAY)

box(680, cy + 10, 210, 92, "가장 높은 답 → 문장", ["“The next action is", "turn left 30 degree.”"],
    colors=("#FFFFFF", CHOICE[1]), size=13, italic=(0, 1), gap=22)
box(910, cy + 10, 210, 92, "NaVILA 실행부 (그대로)", ["→ 왼쪽 15° 회전 × 2번", "Habitat VLN-CE / Go2 보행"],
    size=12, gap=22)
arrow([(890, cy + 56), (908, cy + 56)])
arrow([(656, cy + 40), (678, cy + 40)])

# parameters
rect(30, 920, 1090, 64, "#F6F8FA", "#D0D7DE", 1.2, 8)
text(48, 946, "파라미터 516.0M", 16, "bold")
text(200, 946, "= 언어 인코더 394.8M + 헤드·채점기 26.5M + 영상 인코더 92.9M(고정) + 신규 1.9M · 학습하는 것 423.1M",
     13, fill=GRAY, maxw=910)
text(48, 970, "NaVILA 8B의 약 1/15 · 판단 1회에 순전파 한 번 · RTX 5060 Ti 중앙값 46.2 ms · "
     "D의 주행 성공률 val_seen 0.21, val_unseen 0.14 (각 100개)", 13, fill=GRAY, maxw=1060)

# ------------------------------------------------------------------------------------------------ training panel
rect(1150, 20, 420, 962, TRAIN[0], TRAIN[1], 1.6, 12, "7 5")
text(1170, 52, "학습: L 실험 (서버 H200)", 15, "bold", TRAIN[1])
text(1170, 74, "주행할 때는 쓰지 않음 · 구조는 D와 같고 데이터만 다름", 12, fill=GRAY, maxw=390)

box(1170, 90, 380, 118, "공통 설정", ["runs/D/last에서 시작 · 3,000 step × 64개", "SigLIP2는 고정, 저장 특징으로 학습",
                                    "손실: Laya RLCD (σ 0.4 → 0.1)", "1,000 step마다 검증과 저장"],
    size=12, gap=20)
box(1170, 220, 380, 176, "실험 3개 (L1, L2는 L0와 비교)",
    ["L0_r2r · 기준", "R2R (NaVILA 주석)만 더 학습", "L1_dagger1", "+ DAgger 1회차 14.3만 판단 (74%가 경로 밖)",
     "L2_teacher", "+ NaVILA 8B 교사 분포 (α = 0.5)"],
    colors=("#FFFFFF", TRAIN[1]), size=12, gap=21, bold=(0, 2, 4))
box(1170, 408, 380, 96, "교사: NaVILA 8B (고정, L2만)", ["같은 프레임 8장과 지시문으로 답 10개 점수",
                                                       "→ p_교사, R2R 28.9만 샘플 전부 미리 계산"],
    colors=("#FFFFFF", TRAIN[1]), size=12, gap=22)
arrow([(1360, 504), (1360, 514)], TRAIN[1], "ahr")
box(1170, 516, 380, 158, "1 · 이동 판단 (80%)", ["목표: 정답 one-hot",
                                               "L2: (1 − α) · one-hot + α · p_교사",
                                               "+ 지시문 대조 손실 (가중치 1.0): 자기 지시문",
                                               "  에서 정답이 남의 지시문보다 0.5 nat 더 높게",
                                               "+ 좌우 반전 p = 0.5 (프레임, 지시문, 정답)"],
    colors=("#FFFFFF", TRAIN[1]), size=12, gap=22)
box(1170, 686, 380, 136, "2 · 지시문-경로 짝 맞추기 (20%)", ["“Does the instruction describe this route?”",
                                                          "선택지 yes / no (남는 [M] 8개는 가림)",
                                                          "가짜 짝: 랜드마크 교체 60% · 다른 경로",
                                                          "25%는 구분 쉬운 대체 태그로"],
    colors=("#FFFFFF", TRAIN[1]), size=12, gap=22, italic=(0,))

text(1170, 846, "범례", 14, "bold")
legend = [(VISION, None, "사전학습 영상 인코더 (SigLIP2, 고정)"),
          (LANG, None, "사전학습 언어 인코더 (ModernBERT)"),
          (CHOICE, None, "Laya 헤드 · 채점기 · [M] 마커"),
          (NEW, None, "영상·태그용 신규 층 · 태그 단어"),
          (TOKEN, None, "글자 토큰"),
          (("#FFFFFF", TRAIN[1]), "4 3", "학습 때만 사용")]
for i, (c, dash, lab) in enumerate(legend):
    yy = 858 + 20 * i
    rect(1172, yy, 20, 14, c[0], c[1], 1.3, 3, dash)
    text(1202, yy + 12, lab, 12, maxw=340)

svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" viewBox="0 0 %d %d">' % (W, H, W, H),
       '<defs>'
       '<marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
       '<path d="M0,0 L10,5 L0,10 z" fill="%s"/></marker>' % LINE +
       '<marker id="ahb" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
       '<path d="M0,0 L10,5 L0,10 z" fill="%s"/></marker>' % VISION[1] +
       '<marker id="ahp" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
       '<path d="M0,0 L10,5 L0,10 z" fill="%s"/></marker>' % CHOICE[1] +
       '<marker id="ahr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
       '<path d="M0,0 L10,5 L0,10 z" fill="%s"/></marker>' % TRAIN[1] +
       '</defs>'] + out + ['</svg>']
with open(sys.argv[1], "w", encoding="utf-8") as f:
    f.write("\n".join(svg) + "\n")
print("input row ends at x = %d (panel starts at 1150)" % row_end)
for w in warnings:
    print("may not fit:", w)
