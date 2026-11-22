"""Architecture of the proposed decision model.

    python3 docs/figures/make_arch_figure.py --variant base   # F_base / L0 / D: past frames 1 token each (11 image tokens)
    python3 docs/figures/make_arch_figure.py --variant full   # F_full: past frames 2 x 2 + pose embedding (32 image tokens)

Writes fig_arch_<variant>.png (600 dpi, one column wide) and .pdf next to this script.
Token layout follows lightnav/model.py `assemble` and laya.common.build_sequence:
[CLS] question [SEP] [MASK] option x 10 [SEP] instruction | seen: tags | image tokens [SEP].
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

p = argparse.ArgumentParser()
p.add_argument("--variant", choices=("base", "full"), default="base")
args = p.parse_args()
FULL = args.variant == "full"

for fam in ("Nimbus Roman", "Liberation Serif", "Times New Roman", "DejaVu Serif"):
    if any(f.name == fam for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = fam
        break
plt.rcParams["mathtext.fontset"] = "stix"

W, H = 3.06, 2.05                                    # inches: one column of the KRoC template
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100)
ax.set_ylim(0, 100 * H / W)
ax.axis("off")

C = dict(fixed="#e6e6e6", instr="#cfe2f3", tag="#d9ead3", img="#e4d7f0", enc="#fff2cc", out="#fce5cd",
         edge="#444444", mark="#e69138")
FS = 7.2


def box(x, y, w, h, text, fc, fs=FS, bold=False, lw=0.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.5", fc=fc, ec=C["edge"], lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", linespacing=1.05)


def arrow(x0, y0, x1, y1, text=None, tx=0.0, ty=0.0, ha="left"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=6, lw=0.6,
                                 color=C["edge"], shrinkA=0, shrinkB=0))
    if text:
        ax.text((x0 + x1) / 2 + tx, (y0 + y1) / 2 + ty, text, fontsize=FS - 1.2, ha=ha, va="center",
                style="italic")


# vertical layout (axis units; the canvas is 100 wide and 100*H/W tall)
Y_IN, H_IN = 55.5, 10.5       # inputs
Y_SIG, H_SIG = 41.5, 7.5      # SigLIP2
Y_SEQ, H_SEQ = 27.0, 7.0      # token sequence
Y_ENC, H_ENC = 15.0, 7.0      # encoder
Y_OUT = 3.8                   # outputs (bars and boxes)

# ---------------------------------------------------------------- inputs
box(1.0, Y_IN, 26.0, H_IN, "Question +\n10 options", C["fixed"])
box(30.0, Y_IN, 29.0, H_IN, "Instruction\n“Walk past the sofa,\nstop at the TV.”", C["instr"], fs=FS - 0.6)
fx, fy = 62.5, Y_IN + 3.2
for i in range(7):
    ax.add_patch(Rectangle((fx + i * 2.3, fy + 1.0), 5.0, 5.0, fc="#b7b7b7" if i % 2 else "#cccccc",
                           ec=C["edge"], lw=0.4))
ax.add_patch(Rectangle((fx + 7 * 2.3 + 1.4, fy - 0.2), 9.5, 7.8, fc="#9fc5e8", ec=C["edge"], lw=0.6))
ax.text(fx + 1.0, fy - 2.2, "7 past", fontsize=FS - 1.2, ha="left", va="center")
ax.text(fx + 7 * 2.3 + 6.1, fy - 2.2, "current", fontsize=FS - 1.2, ha="center", va="center")

# ---------------------------------------------------------------- SigLIP2
box(62.0, Y_SIG, 37.0, H_SIG, "SigLIP2-B/16\n(frozen)", C["img"], bold=True, fs=FS - 0.4)
arrow(80.5, fy - 3.6, 80.5, Y_SIG + H_SIG + 0.3)

# ---------------------------------------------------------------- token sequence
# F_full adds each past frame's pose embedding to that frame's tokens (lightnav/model.py frame_tokens)
segs = [(1.0, 11.0, "[CLS] Q\n[SEP]", C["fixed"]), (11.0, 34.0, None, C["fixed"]),
        (34.0, 56.5, "instruction", C["instr"]), (56.5, 73.0, "seen: sofa,\ntv, …", C["tag"]),
        (73.0, 95.0, "32 image tok.\n+ pose emb." if FULL else "11 image\ntokens", C["img"]),
        (95.0, 99.0, "", C["fixed"])]
for x0, x1, lab, fc in segs:
    ax.add_patch(Rectangle((x0, Y_SEQ), x1 - x0, H_SEQ, fc=fc, ec=C["edge"], lw=0.5))
    if lab:
        ax.text((x0 + x1) / 2, Y_SEQ + H_SEQ / 2, lab, ha="center", va="center", fontsize=FS - 1.2, linespacing=1.0)
mx = [12.1 + i * 2.2 for i in range(10)]
for x in mx:
    ax.add_patch(Rectangle((x, Y_SEQ + 0.9), 1.3, H_SEQ - 1.8, fc=C["mark"], ec="none"))
ax.text(22.5, Y_SEQ + H_SEQ + 2.0, "[MASK] × 10", fontsize=FS - 1.2, ha="center", va="center", color="#b45f06")

arrow(14.0, Y_IN - 0.3, 14.0, Y_SEQ + H_SEQ + 4.0)
arrow(44.5, Y_IN - 0.3, 44.5, Y_SEQ + H_SEQ + 0.3)
arrow(67.0, Y_SIG - 0.3, 65.5, Y_SEQ + H_SEQ + 0.3, "tags", tx=-1.0, ha="right")
arrow(92.0, Y_SIG - 0.3, 86.0, Y_SEQ + H_SEQ + 0.3,
      "2×2 + 7×(2×2)" if FULL else "2×2 + 7×1", tx=-1.2, ty=-0.2, ha="right")

# ---------------------------------------------------------------- encoder
box(1.0, Y_ENC, 98.0, H_ENC, "ModernBERT-large (Laya), one forward pass", C["enc"], bold=True, fs=FS - 0.3)
arrow(50.0, Y_SEQ - 0.3, 50.0, Y_ENC + H_ENC + 0.3)

# ---------------------------------------------------------------- outputs at the markers
probs = [0.04, 0.05, 0.08, 0.46, 0.06, 0.03, 0.10, 0.06, 0.03, 0.09]
for i, (x, pr) in enumerate(zip(mx, probs)):
    ax.add_patch(Rectangle((x, Y_OUT), 1.3, 1.0 + 12.0 * pr, fc="#e69138" if i == 3 else "#f6b26b", ec="none"))
ax.plot([11.4, 34.6], [Y_OUT, Y_OUT], color=C["edge"], lw=0.5)
ax.text(22.5, Y_OUT - 2.4, "10 command probabilities", fontsize=FS - 1.2, ha="center", va="center")
arrow(22.5, Y_ENC - 0.3, 22.5, Y_OUT + 7.3)
box(40.5, Y_OUT, 24.0, 8.5, "forward 75 cm", C["out"], fs=FS - 0.4)
arrow(35.5, Y_OUT + 4.25, 40.2, Y_OUT + 4.25)
box(70.5, Y_OUT, 28.5, 8.5, "Go2 locomotion", C["fixed"], fs=FS - 0.4)
arrow(64.8, Y_OUT + 4.25, 70.2, Y_OUT + 4.25)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"fig_arch_{args.variant}")
fig.savefig(out + ".png", dpi=600)
fig.savefig(out + ".pdf")
print("wrote", out + ".png", "and .pdf")
