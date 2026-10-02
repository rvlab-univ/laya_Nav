"""Check the pixel-goal coordinate convention before training Laya-S2.

Draws the dataset goal on the look-down frame twice: red = read as (x, y), blue = read as (y, x).
The marker that lands on the floor where the trajectory goes tells the order; pass it to
train_laya_s2.py as --goal_xy_order. Also reports how many goals fall outside the image per order.

    python scripts/train/laya_s2/check_goal_coords.py --vln_dataset_use r2r_125cm_0_30 --out logs/goal_check
"""

import argparse
import os
import random

from PIL import Image, ImageDraw

from internnav.dataset.laya_s2_dataset import GOAL, frame_path, load_vln_samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vln_dataset_use", default="r2r_125cm_0_30")
    ap.add_argument("--out", default="logs/goal_check")
    ap.add_argument("--n", type=int, default=12, help="images to draw")
    ap.add_argument("--n_stats", type=int, default=2000, help="samples for the out-of-image statistics")
    args = ap.parse_args()

    random.seed(0)
    goals = [s for s in load_vln_samples(args.vln_dataset_use, pixel_goal_only=True) if s["kind"] == GOAL]
    os.makedirs(args.out, exist_ok=True)

    outside = {"xy": 0, "yx": 0}
    drawn = []
    stats = random.sample(goals, min(args.n_stats, len(goals)))
    size = None
    for i, s in enumerate(stats):
        if size is None or i < args.n:
            img = Image.open(frame_path(s, s["start"], look_down=True)).convert("RGB")
            size = img.size
        W, H = size
        a, b = s["goal"]
        outside["xy"] += not (0 <= a < W and 0 <= b < H)
        outside["yx"] += not (0 <= b < W and 0 <= a < H)
        if i < args.n:
            d = ImageDraw.Draw(img)
            for (x, y), c in (((a, b), "red"), ((b, a), "blue")):
                d.ellipse([x - 8, y - 8, x + 8, y + 8], outline=c, width=4)
            d.text((10, 10), f"goal={s['goal']} size={W}x{H}  red=(x,y) blue=(y,x)", fill="yellow")
            img.save(os.path.join(args.out, f"goal_{i:02d}.jpg"))
            drawn.append(img)

    n = len(stats)
    a_vals = [s["goal"][0] for s in stats]
    b_vals = [s["goal"][1] for s in stats]
    print(f"{len(goals)} goal samples, image size {size} (W x H)")
    print(f"first value : min {min(a_vals)}  max {max(a_vals)}  mean {sum(a_vals) / n:.1f}")
    print(f"second value: min {min(b_vals)}  max {max(b_vals)}  mean {sum(b_vals) / n:.1f}")
    print(f"outside image: as (x, y) {outside['xy']}/{n}   as (y, x) {outside['yx']}/{n}")

    # all drawings in one contact sheet (4 columns, half size)
    if drawn:
        w, h = drawn[0].size[0] // 2, drawn[0].size[1] // 2
        cols = 4
        sheet = Image.new("RGB", (cols * w, ((len(drawn) + cols - 1) // cols) * h))
        for k, im in enumerate(drawn):
            sheet.paste(im.resize((w, h)), ((k % cols) * w, (k // cols) * h))
        sheet.save(os.path.join(args.out, "sheet.jpg"), quality=85)
    print(f"drawings: {args.out}/sheet.jpg (all) and goal_*.jpg  (red = (x, y), blue = (y, x))")


if __name__ == "__main__":
    main()
