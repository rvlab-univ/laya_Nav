"""LayaNav: Laya-S2 and the System 1 trajectory generator merged into one model.

The DualVLN System 1 re-encodes images with its own backbone (Depth-Anything ViT-S + memory encoder
+ QFormer) and samples 32 trajectories with a 10-step flow-matching DiT (x2 for CFG), only to average
them. LayaNav drops both: the trajectory is regressed in a single pass by a small transformer decoder
that reads features the decision model already computed.

One set of weights, two calls:
- ``forward`` (System 2 role, every re-decision): Laya-S2 decision -> action, or pixel goal + plan memory.
- ``plan`` (System 1 role, every re-plan): encode only the current look-down frame with the shared vision
  encoder, then decode the 32-step trajectory against
  [plan memory, look-down frame at decision time, current look-down frame].

The trajectory format is the DualVLN one ((dx, dy, dyaw) per 0.1 m step, dx / dy scaled by 4), so
``traj_to_actions`` and the evaluation loop are unchanged.
"""

import json
import os
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .laya_s2 import CONFIG_NAME, FourierXY, LayaS2, LayaS2Config

TRAJ_SEG_MEMORY, TRAJ_SEG_GOAL_FRAME, TRAJ_SEG_CUR_FRAME = range(3)
GOAL_MARK_SIGMA = 0.75  # patches, same spread as the decision target around the goal


@dataclass
class LayaNavConfig(LayaS2Config):
    model_type: str = "laya_nav"
    traj_dim: int = 384
    traj_layers: int = 4
    traj_steps: int = 32  # DualVLN predict_step_num
    # off by default so that older checkpoints load unchanged; train_laya_nav.py turns both on for new heads
    traj_fuse_layers: int = 0  # self-attention over [memory, goal frame, current frame] (System 1 memory_encoder role)
    traj_goal_mark: bool = False  # mark the goal on the goal-frame patches and add a goal position token


class LayaNav(LayaS2):
    config_class = LayaNavConfig

    def __init__(self, cfg: LayaNavConfig, text_encoder: nn.Module, vision_encoder: nn.Module):
        super().__init__(cfg, text_encoder, vision_encoder)
        d, t = self.d, cfg.traj_dim
        nhead = max(1, t // 64)
        self.traj_mem_proj = nn.Linear(d, t)  # goal token + frame features
        self.traj_latent_proj = nn.Linear(cfg.latent_dim, t)
        self.traj_seg = nn.Parameter(torch.zeros(3, t))
        self.traj_patch_pos = nn.Parameter(torch.zeros(self.n_patch, t))
        self.traj_queries = nn.Parameter(torch.zeros(cfg.traj_steps, t))
        for p in (self.traj_seg, self.traj_patch_pos, self.traj_queries):
            nn.init.normal_(p, std=0.02)
        # the decoder queries read the frames only through cross-attention, so patches of the two frames never meet;
        # the fusion layers let them attend to each other (where is the goal patch now, how far have we moved)
        self.traj_fuse = None
        if cfg.traj_fuse_layers > 0:
            enc = nn.TransformerEncoderLayer(t, nhead, 4 * t, cfg.dropout, batch_first=True, norm_first=True)
            self.traj_fuse = nn.TransformerEncoder(enc, cfg.traj_fuse_layers, enable_nested_tensor=False)
        if cfg.traj_goal_mark:
            self.traj_goal_xy = FourierXY(t)
            self.traj_goal_mark = nn.Parameter(torch.zeros(t))
            nn.init.normal_(self.traj_goal_mark, std=0.02)
        layer = nn.TransformerDecoderLayer(t, nhead, 4 * t, cfg.dropout, batch_first=True, norm_first=True)
        self.traj_decoder = nn.TransformerDecoder(layer, cfg.traj_layers)
        self.traj_out = nn.Sequential(nn.LayerNorm(t), nn.Linear(t, 3))

    # ------------------------------------------------------------------ io
    @classmethod
    def from_laya_s2(cls, ckpt_dir: str, **traj_cfg) -> "LayaNav":
        """Start from a trained Laya-S2 checkpoint; the trajectory head is freshly initialized."""
        s2 = LayaS2.from_pretrained(ckpt_dir)
        cfg = LayaNavConfig(**{**vars(s2.cfg), **traj_cfg})
        model = cls(cfg, s2.text, s2.vision)
        missing, unexpected = model.load_state_dict(s2.state_dict(), strict=False)
        assert not unexpected and all(k.startswith("traj_") for k in missing), (missing, unexpected)
        return model

    @classmethod
    def load_any(cls, ckpt_dir: str, **traj_cfg) -> "LayaNav":
        """LayaNav checkpoint as is, or a Laya-S2 checkpoint with a new trajectory head (``traj_cfg``)."""
        return cls.from_pretrained(ckpt_dir) if _is_nav(ckpt_dir) else cls.from_laya_s2(ckpt_dir, **traj_cfg)

    def traj_parameters(self):
        return [p for n, p in self.named_parameters() if n.startswith("traj_")]

    def forward(self, *args, traj_pixels=None, traj_mask=None, **kwargs):
        """Laya-S2 decision; with ``traj_pixels`` also trajectories from the starting frames given there.

        ``traj_pixels`` is [B, K, 3, S, S] (K starting frames per sample, ``traj_mask`` [B, K]) or [B, 3, S, S]
        (one per sample, ``traj_mask`` [B]). ``out["traj"]`` holds one trajectory per valid (sample, start) pair,
        indexed by ``out["traj_idx"]`` (sample) and ``out["traj_slot"]`` (start).

        Training runs both in one forward (needed for DDP gradient sync); at inference ``forward`` and
        ``plan`` are called separately, at the System 2 and System 1 rates.
        """
        out = super().forward(*args, **kwargs)
        if traj_pixels is not None:
            if traj_pixels.dim() == 4:
                traj_pixels = traj_pixels[:, None]
                traj_mask = None if traj_mask is None else traj_mask[:, None]
            if traj_mask is None:
                traj_mask = torch.ones(traj_pixels.shape[:2], dtype=torch.bool, device=traj_pixels.device)
            idx, slot = traj_mask.bool().nonzero(as_tuple=True)
            out["traj_idx"], out["traj_slot"] = idx, slot
            if len(idx):
                memory = self.plan_memory({k: out[k][idx] for k in ("goal_token", "latent", "goal_xy")})
                cur = self.encode_frame(traj_pixels[idx, slot])
                out["traj"] = self.plan(memory, out["down_feat"][idx], cur, goal_xy=out["goal_xy"][idx])
        return out

    # ------------------------------------------------------------------ planning (System 1 role)
    def encode_frame(self, pixels: torch.Tensor) -> torch.Tensor:
        """[B, 3, S, S] -> [B, P, d] projected patch features (same space as ``down_feat``)."""
        return self.vis_proj(self.encode_images(pixels))

    def plan_memory(self, out: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Decision outputs -> [B, M, traj_dim] memory kept until the next decision."""
        goal = self.traj_mem_proj(out["goal_token"])[:, None]
        parts = [goal, self.traj_latent_proj(out["latent"].to(goal.dtype))]
        if self.cfg.traj_goal_mark:
            parts.append(self.traj_goal_xy(out["goal_xy"]).to(goal.dtype)[:, None])
        return torch.cat(parts, 1) + self.traj_seg[TRAJ_SEG_MEMORY]

    def _goal_bump(self, goal_xy: torch.Tensor) -> torch.Tensor:
        """Normalized goal [B, 2] -> [B, P] weights, 1 at the goal and falling off over neighbouring patches."""
        g, dev = self.grid, goal_xy.device
        ys, xs = torch.meshgrid(torch.arange(g, device=dev), torch.arange(g, device=dev), indexing="ij")
        centers = torch.stack([xs, ys], -1).reshape(-1, 2).float() + 0.5
        d2 = ((centers[None] - goal_xy.float()[:, None] * g) ** 2).sum(-1)
        return torch.exp(-d2 / (2 * GOAL_MARK_SIGMA**2))

    def plan(
        self,
        memory: torch.Tensor,
        goal_feat: torch.Tensor,
        cur_feat: torch.Tensor,
        goal_xy: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """memory [B, M, t], goal_feat / cur_feat [B, P, d], goal_xy [B, 2] -> trajectory [B, traj_steps, 3]."""
        B = memory.shape[0]
        frames = [
            self.traj_mem_proj(f) + self.traj_patch_pos + self.traj_seg[s]
            for f, s in ((goal_feat, TRAJ_SEG_GOAL_FRAME), (cur_feat, TRAJ_SEG_CUR_FRAME))
        ]
        if self.cfg.traj_goal_mark:
            assert goal_xy is not None, "traj_goal_mark needs the goal position"
            frames[0] = frames[0] + self._goal_bump(goal_xy)[..., None].to(frames[0].dtype) * self.traj_goal_mark
        mem = torch.cat([memory.to(frames[0].dtype)] + frames, 1)
        if self.traj_fuse is not None:
            mem = self.traj_fuse(mem)
        q = self.traj_queries.to(mem.dtype).expand(B, -1, -1)
        return self.traj_out(self.traj_decoder(q, mem)).float()


def _is_nav(ckpt_dir: str) -> bool:
    with open(os.path.join(ckpt_dir, CONFIG_NAME)) as f:
        return json.load(f).get("model_type") == "laya_nav"


@dataclass
class TrajLossWeights:
    traj: float = 1.0  # Huber on the per-step (dx, dy, dyaw) targets
    traj_cum: float = 1.0  # L2 on the accumulated (x, y) path in metres


def traj_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, w: TrajLossWeights):
    """pred / target [B, T, 3] in the DualVLN format (dx, dy scaled by 4); mask [B] samples with a target."""
    zero = pred.sum() * 0  # keeps the graph (and DDP) happy when no sample has a trajectory
    if not mask.any():
        return zero, dict(l_traj=zero.detach(), ade=zero.detach(), fde=zero.detach(), n_traj=zero.detach())
    p, t = pred[mask].float(), target[mask].float()
    l_step = F.smooth_l1_loss(p, t)
    xy_p, xy_t = torch.cumsum(p[..., :2] / 4, 1), torch.cumsum(t[..., :2] / 4, 1)
    dist = (xy_p - xy_t).norm(dim=-1)  # [N, T] metres
    l_cum = dist.mean()
    loss = w.traj * l_step + w.traj_cum * l_cum
    # ade / fde: mean / final displacement of the accumulated path, metres
    stats = dict(l_traj=l_step.detach(), ade=l_cum.detach(), fde=dist[:, -1].mean().detach(), n_traj=mask.sum().float())
    return loss, stats
