"""The student as a decision function: NaVILA's instruction and frames in, one of NaVILA's 10 sentences out."""
import time
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

from . import data as D
from .pose import relative
from .questions import ANSWERS, QUESTION, decide


class StudentAgent:
    """`ckpt` is a folder written by `lightnav.model.save` (e.g. runs/N1/last), or "laya" for the untrained student
    (Laya checkpoint and SigLIP2 with fresh projection layers), which only checks the plumbing."""

    def __init__(self, ckpt: str, device: str = "cuda", tau_stop: Optional[float] = None):
        from laya.train import to_internal

        from .model import LayaNativeConfig, build, load

        if ckpt == "laya":
            self.model, self.tok = build(LayaNativeConfig())
        else:
            self.model, self.tok = load(ckpt)
        self.device = torch.device(device)
        self.model.to(self.device).eval()
        self.cfg = self.model.cfg
        self.question = to_internal("action", QUESTION)
        self.transform = D.make_image_transform(self.cfg.image_size, self.cfg.image_mean, self.cfg.image_std)
        self.tau_stop = tau_stop
        self.params = sum(p.numel() for p in self.model.parameters())
        self.no_poses = 0               # decisions a pose student made without poses

    @torch.no_grad()
    def decide(self, instruction: str, frames: Sequence[np.ndarray], slots: Sequence[int],
               poses: Optional[np.ndarray] = None) -> Tuple[str, List[float], float]:
        """frames: the real frames among NaVILA's 8 (H x W x 3 uint8), slots: their positions (the current one is 7),
        poses: their poses [k, 3] (lightnav.pose.dead_reckoning of the actions run), read only by a model with
        `cfg.pose`; without them its frames count as pose unknown.
        Returns (sentence, 10 probabilities, milliseconds for the model including image preprocessing)."""
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        t = time.perf_counter()
        pixels = torch.stack([self.transform(Image.fromarray(np.asarray(f, dtype=np.uint8)[..., :3])) for f in frames])
        ids, markers = D.encode_text(self.tok, self.question, instruction, self.cfg.text_max_len, self.cfg.head_max_len)
        item = {"ids": ids, "markers": markers, "pixels": pixels, "slots": list(slots)}
        if self.cfg.pose and poses is not None:
            p = np.asarray(poses, dtype=np.float64)
            item["poses"] = relative(p, p[-1])
        elif self.cfg.pose:
            self.no_poses += 1
            if self.no_poses == 1:
                print("this student reads poses and got none: its frames count as pose unknown", flush=True)
        batch = D.collate([item], self.tok.pad_token_id)
        batch = {k: v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            logits = self.model(batch)
        probs = torch.softmax(logits.float(), -1)[0].cpu().numpy()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        ms = (time.perf_counter() - t) * 1000
        return ANSWERS[decide(probs, self.tau_stop)], probs.tolist(), ms
