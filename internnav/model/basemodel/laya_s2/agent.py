"""Laya-S2 (System 2) + DualVLN System 1, as used by the Habitat evaluator (mode='laya_s2')."""

import os

import torch
import torch.nn as nn
from PIL import Image

from internnav.dataset.laya_s2_dataset import make_image_transform
from internnav.model.basemodel.internvla_n1.system1_standalone import DualVLNSystem1

from .laya_s2 import LayaS2


class LayaS2Agent(nn.Module):
    """Laya-S2 (System 2) + DualVLN System 1."""

    def __init__(self, student: LayaS2, s1: nn.Module, tokenizer, device):
        super().__init__()
        self.student = student
        self.s1 = s1
        self.tok = tokenizer
        self.device = device
        self.transform = make_image_transform(student.cfg)

    @torch.no_grad()
    def decide(self, instruction: str, history: list, cur_image: Image.Image, down_image: Image.Image) -> dict:
        cfg = self.student.cfg
        S, H = cfg.image_size, cfg.num_history
        hist = torch.zeros(1, H, 3, S, S)
        hist_mask = torch.zeros(1, H, dtype=torch.bool)
        for j, img in enumerate(history[:H]):
            hist[0, j] = self.transform(img)
            hist_mask[0, j] = True
        ids = torch.tensor([self.tok(instruction, truncation=True, max_length=cfg.max_text_len)["input_ids"]])
        dev = self.device
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.device(dev).type == "cuda"):
            out = self.student(
                ids.to(dev),
                torch.ones_like(ids, dtype=torch.bool).to(dev),
                hist.to(dev),
                hist_mask.to(dev),
                self.transform(cur_image)[None].to(dev),
                self.transform(down_image)[None].to(dev),
            )
        return self.student.decide(out)[0]


def load_laya_s2(model_args, device) -> LayaS2Agent:
    from transformers import AutoTokenizer

    student = LayaS2.from_pretrained(model_args.model_path).to(device).eval()
    tok = AutoTokenizer.from_pretrained(os.path.join(model_args.model_path, "tokenizer"))
    s1 = DualVLNSystem1.from_pretrained(model_args.system1_path, device=device)
    assert "nextdit" in s1.get_system1_type(), "Laya-S2 latents are cond_projector outputs (nextdit System 1 only)"
    return LayaS2Agent(student, s1, tok, device)
