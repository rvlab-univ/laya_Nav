"""Store the frozen vision encoder's features of every frame (lightnav.features), so training can skip the encoder.

Reads a NaVILA-format dataset (annotations.json and <frames>/<episode>/frame_<i>.jpg) and writes one file per
episode with the features of all its frames, as rendered and mirrored. Episodes already stored are skipped, so a
stopped run resumes; --delete_frames removes an episode's images once its features are safely on disk (for data
rendered in chunks by this PC, never for a downloaded dataset you want to keep).

    HF_HOME=assets/hf_cache env/bin/python scripts/cache_features.py --source r2r \\
        --ann data/NaVILA-Dataset/R2R/annotations.json --frames data/NaVILA-Dataset/R2R/train_sample \\
        --out data/features/siglip2_g2
"""
import argparse
import os
import shutil
import sys
import time

import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lightnav import data as D                                   # noqa: E402
from lightnav import features as F                               # noqa: E402
from lightnav.model import LayaNativeConfig, _vision_tower       # noqa: E402


class Episodes(Dataset):
    """Episode -> its frames, transformed, as rendered and mirrored: [2, T, 3, H, W]."""

    def __init__(self, episodes, frames, transform):
        self.episodes, self.frames, self.transform = episodes, frames, transform

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, i):
        prefix, n = self.episodes[i]
        both = [[], []]
        for k in range(n):
            with Image.open(os.path.join(self.frames, "%sframe_%d.jpg" % (prefix, k))) as im:
                im = im.convert("RGB")
                both[0].append(self.transform(im))
                both[1].append(self.transform(ImageOps.mirror(im)))
        return prefix, torch.stack([torch.stack(b) for b in both])


def main():
    cfg = LayaNativeConfig()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True, help="r2r, rxr, ... (the folder name inside --out)")
    p.add_argument("--ann", required=True)
    p.add_argument("--frames", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--grid", type=int, default=2, help="stored grid; 2 serves --grids 2 1")
    p.add_argument("--vision_encoder", default=cfg.vision_encoder)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--delete_frames", action="store_true")
    args = p.parse_args()

    from transformers import AutoModel
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vision = _vision_tower(AutoModel.from_pretrained(args.vision_encoder)).to(device).eval()
    F.write_meta(args.out, args.vision_encoder, vision.config.image_size, args.grid, vision.config.hidden_size)

    need = {}
    for r in D.load_rows(args.ann, args.source):
        need[r.prefix] = max(need.get(r.prefix, 0), r.n_frames)
    todo, missing = [], 0
    for prefix, n in sorted(need.items()):
        if os.path.exists(F.episode_file(args.out, args.source, prefix)):
            continue
        folder = os.path.join(args.frames, prefix)
        if os.path.isdir(folder) and {"frame_%d.jpg" % i for i in range(n)} <= set(os.listdir(folder)):
            todo.append((prefix, n))
        else:
            missing += 1
    print("%s: %d episodes, %d stored already, %d to do, %d without frames on disk"
          % (args.source, len(need), len(need) - len(todo) - missing, len(todo), missing), flush=True)
    transform = D.make_image_transform(cfg.image_size, cfg.image_mean, cfg.image_std)
    loader = DataLoader(Episodes(todo, args.frames, transform), batch_size=None, shuffle=False,
                        num_workers=args.workers)
    started, frames = time.time(), 0
    for i, (prefix, px) in enumerate(loader):
        feats = torch.stack([F.encode_pixels(vision, px[k].to(device), args.grid) for k in (0, 1)])
        F.save_episode(args.out, args.source, prefix, feats.numpy())
        frames += px.shape[1]
        if args.delete_frames:
            shutil.rmtree(os.path.join(args.frames, prefix))
        if (i + 1) % 50 == 0 or i + 1 == len(todo):
            rate = frames / (time.time() - started)
            print("%d / %d episodes, %.0f frames/s (each frame twice: as rendered and mirrored)"
                  % (i + 1, len(todo), rate), flush=True)


if __name__ == "__main__":
    main()
