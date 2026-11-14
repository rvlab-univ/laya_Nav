"""Frame features of the frozen vision encoder, computed once and read back in training (design_C_review.md, step 1).

Run D keeps SigLIP2 frozen, so what the student needs of a frame never changes: the patch features averaged down to a
small grid (the image tokens) and the pooled embedding (the landmark tags). With these on disk the vision encoder
drops out of training, and a 16 GB GPU trains the student.

Store layout:
    <root>/meta.json                      encoder, image size, grid, feature size
    <root>/<source>/<episode>.npy         float16 [2, T, grid * grid + 1, dv]
        axis 0: 0 = the frame as rendered, 1 = mirrored left-right (the mirror augmentation)
        axis 1: frame_0 .. frame_{T-1} of the episode
        axis 2: the grid tokens (row-major), then the pooled embedding

A grid of 2 gives the current frame's 2 x 2 tokens directly and a past frame's single token as their mean (equal
bins, so it is what pooling the patches gives). The colour jitter of image training has no counterpart here.
"""
import json
import os
from collections import OrderedDict
from typing import Dict, List, Sequence

import numpy as np
import torch

META = "meta.json"


def episode_file(root: str, source: str, prefix: str) -> str:
    return os.path.join(root, source, prefix.rstrip("/") + ".npy")


def write_meta(root: str, encoder: str, image_size: int, grid: int, dv: int) -> None:
    os.makedirs(root, exist_ok=True)
    meta = {"vision_encoder": encoder, "image_size": image_size, "grid": grid, "dv": dv, "mirrored": True,
            "dtype": "float16"}
    path = os.path.join(root, META)
    if os.path.exists(path):
        with open(path) as f:
            old = json.load(f)
        if old != meta:
            raise ValueError("%s was built with %s, not %s" % (root, old, meta))
        return
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)


def save_episode(root: str, source: str, prefix: str, feats: np.ndarray) -> None:
    """`feats`: [2, T, grid * grid + 1, dv]. Written under a temporary name first, so a stopped run leaves no
    half-written episode."""
    path = episode_file(root, source, prefix)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path[:-4] + ".tmp.npy"
    np.save(tmp, np.asarray(feats, dtype=np.float16))
    os.replace(tmp, path)


@torch.no_grad()
def encode_pixels(vision, pixels: torch.Tensor, grid: int, batch: int = 128) -> torch.Tensor:
    """[N, grid * grid + 1, dv] float32 (on the CPU) for transformed frames [N, 3, H, W] on the encoder's device."""
    patch = vision.config.image_size // vision.config.patch_size
    rows = []
    for s in range(0, len(pixels), batch):
        px = pixels[s:s + batch]
        with torch.autocast(px.device.type, dtype=torch.bfloat16, enabled=px.device.type == "cuda"):
            feats = vision(pixel_values=px).last_hidden_state                     # [n, patch * patch, dv]
            pooled = vision.head(feats)
        n, _, dv = feats.shape
        g = torch.nn.functional.avg_pool2d(feats.float().view(n, patch, patch, dv).permute(0, 3, 1, 2),
                                           patch // grid).flatten(2).transpose(1, 2)
        rows.append(torch.cat([g, pooled.float()[:, None]], 1).cpu())
    return torch.cat(rows)


def encode_images(vision, images: Sequence, transform, grid: int, device) -> np.ndarray:
    """[2, T, grid * grid + 1, dv] float16 for PIL images in episode order: as given, and mirrored."""
    from PIL import ImageOps

    out = [encode_pixels(vision, torch.stack([transform(ImageOps.mirror(im) if flip else im) for im in images])
                         .to(device), grid) for flip in (False, True)]
    return torch.stack(out).numpy().astype(np.float16)


class FeatureStore:
    """The features of one source ("r2r", "rxr", ...). Passed where a frame folder would go: `lightnav.data.load_frames`
    reads from it instead of decoding images."""

    def __init__(self, root: str, source: str, keep: int = 256):
        with open(os.path.join(root, META)) as f:
            self.meta = json.load(f)
        self.root, self.source, self.grid, self.keep = root, source, self.meta["grid"], keep
        self._open: "OrderedDict[str, np.ndarray]" = OrderedDict()

    def __getstate__(self):                     # DataLoader workers get their own (empty) set of open files
        return {**self.__dict__, "_open": OrderedDict()}

    def has(self, prefix: str, n_frames: int = 0) -> bool:
        path = episode_file(self.root, self.source, prefix)
        if not os.path.exists(path):
            return False
        return n_frames <= 0 or self.episode(prefix).shape[1] >= n_frames

    def episode(self, prefix: str) -> np.ndarray:
        arr = self._open.get(prefix)
        if arr is None:
            arr = np.load(episode_file(self.root, self.source, prefix), mmap_mode="r")
            self._open[prefix] = arr
            if len(self._open) > self.keep:
                self._open.popitem(last=False)
        else:
            self._open.move_to_end(prefix)
        return arr

    def frames(self, prefix: str, index: List[int], flip: bool = False) -> Dict[str, torch.Tensor]:
        """{"grid": [k, grid * grid, dv], "pooled": [k, dv]} float32 for the frames `index` of an episode."""
        x = torch.from_numpy(np.ascontiguousarray(self.episode(prefix)[int(flip), index])).float()
        return {"grid": x[:, :-1], "pooled": x[:, -1]}
