"""Agents for the Habitat evaluator.

- ``LayaS2Agent`` (mode='laya_s2'): Laya-S2 as System 2 + the DualVLN System 1 (DiT).
- ``LayaNavAgent`` (mode='laya_nav'): the single LayaNav model for both roles.

Both expose the same three calls, so the evaluation loop does not depend on the System 1:
``decide`` (System 2 rate), ``start_goal`` (after a pixel-goal decision), ``plan`` (System 1 rate).
"""

import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from internnav.dataset.laya_s2_dataset import make_image_transform

from .laya_nav import LayaNav
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
        self._out = None

    def _autocast(self):
        return torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.device(self.device).type == "cuda")

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
        with self._autocast():
            self._out = self.student(
                ids.to(dev),
                torch.ones_like(ids, dtype=torch.bool).to(dev),
                hist.to(dev),
                hist_mask.to(dev),
                self.transform(cur_image)[None].to(dev),
                self.transform(down_image)[None].to(dev),
            )
        return self.student.decide(self._out)[0]

    @torch.no_grad()
    def start_goal(self, decision: dict, down_image: Image.Image, down_depth: torch.Tensor):
        """Fix the System 1 condition for this pixel goal (latent + look-down frame at decision time)."""
        self._latent = decision["latent"][None].to(torch.bfloat16)
        self._goal_image = torch.tensor(np.array(down_image.resize((224, 224)))).to(torch.bfloat16) / 255
        self._goal_depth = down_depth.unsqueeze(-1).to(torch.bfloat16)

    @torch.no_grad()
    def plan(self, down_image: Image.Image, down_depth: torch.Tensor) -> torch.Tensor:
        """Trajectory samples [N, 32, 3] (DualVLN format) from the current look-down frame."""
        image = torch.tensor(np.array(down_image.resize((224, 224)))).to(torch.bfloat16) / 255
        images = torch.stack([self._goal_image, image]).unsqueeze(0).to(self.device)
        depths = torch.stack([self._goal_depth, down_depth.unsqueeze(-1).to(torch.bfloat16)]).unsqueeze(0)
        return self.s1.generate_traj(self._latent, images, depths.to(self.device), latents_projected=True)


class LayaNavAgent(LayaS2Agent):
    """LayaNav: one model, the trajectory head takes the place of the DualVLN System 1."""

    def __init__(self, model: LayaNav, tokenizer, device):
        super().__init__(model, None, tokenizer, device)

    @torch.no_grad()
    def start_goal(self, decision: dict, down_image: Image.Image, down_depth: torch.Tensor):
        with self._autocast():
            self._memory = self.student.plan_memory(self._out)
        self._goal_feat = self._out["down_feat"]

    @torch.no_grad()
    def plan(self, down_image: Image.Image, down_depth: torch.Tensor) -> torch.Tensor:
        with self._autocast():
            cur = self.student.encode_frame(self.transform(down_image)[None].to(self.device))
            return self.student.plan(self._memory, self._goal_feat, cur)


def _tokenizer(model_path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(os.path.join(model_path, "tokenizer"))


def load_laya_s2(model_args, device) -> LayaS2Agent:
    from internnav.model.basemodel.internvla_n1.system1_standalone import (
        DualVLNSystem1,
    )

    student = LayaS2.from_pretrained(model_args.model_path).to(device).eval()
    s1 = DualVLNSystem1.from_pretrained(model_args.system1_path, device=device)
    assert "nextdit" in s1.get_system1_type(), "Laya-S2 latents are cond_projector outputs (nextdit System 1 only)"
    return LayaS2Agent(student, s1, _tokenizer(model_args.model_path), device)


def load_laya_nav(model_args, device) -> LayaNavAgent:
    model = LayaNav.from_pretrained(model_args.model_path).to(device).eval()
    return LayaNavAgent(model, _tokenizer(model_args.model_path), device)


def load_agent(model_args, device):
    return load_laya_nav(model_args, device) if model_args.mode == "laya_nav" else load_laya_s2(model_args, device)
