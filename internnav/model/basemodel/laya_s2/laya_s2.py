"""Laya-style lightweight System 2 for InternVLA-N1 (DualVLN).

Replaces the autoregressive Qwen2.5-VL-7B System 2 with a non-autoregressive decision model in the
style of Laya's ``DecisionModel`` (https://huggingface.co/convaiinnovations/laya): a bidirectional
encoder (mmBERT) reads the instruction together with projected visual tokens, and every candidate
answer is scored at its own marker token. Here the candidates are the discrete actions
(STOP / TURN_LEFT / TURN_RIGHT) and the patches of the look-down image (pixel goal), so one softmax
covers both "what to do" and "where to go".

Extra heads:
- offset head: sub-patch position of the pixel goal.
- act head (Laya): pooled state + a summary of its own answer distribution -> act / escalate to teacher.
- latent head: goal-conditioned queries -> System 1 condition, distilled from
  ``cond_projector(teacher hidden states)`` so the existing System 1 can be reused as is.

Input sequence (built per sample, then right-padded):
    [CLS] instruction [SEP] | action markers | history frames (pooled) | current frame (pooled) | look-down patches
"""

import json
import math
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

SEG_TEXT, SEG_ACTION, SEG_HIST, SEG_CUR, SEG_DOWN = range(5)
CONFIG_NAME = "laya_s2_config.json"
UNUSED_PREFIXES = ("vision.head.",)  # SigLIP pooling head, removed (see LayaS2.__init__)
ADDON_PREFIXES = ("ground_proj.", "match_head.")  # may be added to a checkpoint that was trained without them
WEIGHTS_NAME = "laya_s2.pt"


@dataclass
class LayaS2Config:
    text_encoder: str = "jhu-clsp/mmBERT-base"
    vision_encoder: str = "google/siglip2-base-patch16-224"
    image_size: int = 224
    image_mean: List[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])
    image_std: List[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])
    num_history: int = 8
    hist_pool: int = 4  # history frame -> hist_pool x hist_pool tokens
    cur_pool: int = 7  # current (horizontal) frame -> cur_pool x cur_pool tokens
    head_layers: int = 2
    latent_layers: int = 2
    n_query: int = 4  # must match the teacher's TRAJ token count
    latent_dim: int = 768  # must match System 1's LatentEmbSize
    actions: List[int] = field(default_factory=lambda: [0, 2, 3])  # habitat ids: STOP, TURN_LEFT, TURN_RIGHT
    max_text_len: int = 160
    dropout: float = 0.1
    # grounding prior (grounding.py): per-patch match of the instruction's phrases from a frozen SigLIP2, added to
    # the current-frame and look-down tokens. Off by default so that older checkpoints load unchanged.
    grounding: bool = False
    grounding_model: str = "google/siglip2-base-patch16-224"
    grounding_chunks: int = 6
    match_head: bool = False  # instruction / view matching score (auxiliary loss against shuffled instructions)

    def save(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, CONFIG_NAME), "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "LayaS2Config":
        if os.path.isdir(path):
            path = os.path.join(path, CONFIG_NAME)
        with open(path) as f:
            return cls(**json.load(f))


def _vision_tower(model: nn.Module) -> nn.Module:
    # SigLIP/CLIP checkpoints load as dual-tower models; keep the image tower only.
    return getattr(model, "vision_model", model)


def build_encoders(cfg: LayaS2Config):
    from transformers import AutoModel

    text = AutoModel.from_pretrained(cfg.text_encoder, attn_implementation="sdpa")
    vision = _vision_tower(AutoModel.from_pretrained(cfg.vision_encoder))
    return text, vision


class FourierXY(nn.Module):
    """Normalized (x, y) in [0, 1] -> d-dim embedding."""

    def __init__(self, d: int, n_freq: int = 16):
        super().__init__()
        self.register_buffer("freqs", 2.0 ** torch.arange(n_freq).float() * math.pi, persistent=False)
        self.proj = nn.Linear(4 * n_freq, d)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        a = xy[..., None].float() * self.freqs  # [B, 2, F]
        return self.proj(torch.cat([a.sin(), a.cos()], -1).flatten(-2))


class LayaS2(nn.Module):
    config_class = LayaS2Config

    def __init__(self, cfg: LayaS2Config, text_encoder: nn.Module, vision_encoder: nn.Module):
        super().__init__()
        self.cfg = cfg
        self.text = text_encoder
        self.vision = _vision_tower(vision_encoder)
        # SigLIP's attention-pooling head summarizes the image into one vector (pooler_output); only the patch
        # tokens are used here, so drop it instead of computing and discarding it on every image
        if getattr(self.vision, "use_head", False):
            self.vision.use_head = False
            self.vision.config.vision_use_head = False
            del self.vision.head

        d = self.text.config.hidden_size
        dv = self.vision.config.hidden_size
        self.d = d
        self.grid = cfg.image_size // self.vision.config.patch_size
        self.n_patch = self.grid * self.grid
        self.n_action = len(cfg.actions)
        nhead = max(1, d // 64)

        self.vis_proj = nn.Sequential(nn.Linear(dv, d), nn.GELU(), nn.Linear(d, d))
        self.seg_emb = nn.Embedding(5, d)
        self.hist_slot_emb = nn.Embedding(cfg.num_history, d)
        self.pos_hist = nn.Parameter(torch.zeros(cfg.hist_pool**2, d))
        self.pos_cur = nn.Parameter(torch.zeros(cfg.cur_pool**2, d))
        self.pos_down = nn.Parameter(torch.zeros(self.n_patch, d))
        self.action_marker = nn.Parameter(torch.zeros(d))
        self.action_emb = nn.Embedding(self.n_action, d)
        for p in (self.pos_hist, self.pos_cur, self.pos_down):
            nn.init.normal_(p, std=0.02)

        layer = nn.TransformerEncoderLayer(d, nhead, 4 * d, cfg.dropout, batch_first=True, norm_first=True)
        self.head = (
            nn.TransformerEncoder(layer, cfg.head_layers, enable_nested_tensor=False) if cfg.head_layers > 0 else None
        )
        self.scorer = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.offset_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 2))
        self.act_head = nn.Sequential(nn.Linear(d + 4, 256), nn.GELU(), nn.Linear(256, 2))

        if cfg.grounding:
            # zero output layer: a model that gets grounding added starts out computing exactly what it did before
            self.ground_proj = nn.Sequential(nn.Linear(cfg.grounding_chunks, d), nn.GELU(), nn.Linear(d, d))
            nn.init.zeros_(self.ground_proj[-1].weight)
            nn.init.zeros_(self.ground_proj[-1].bias)
        if cfg.match_head:
            self.match_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
        self._grounder = []  # frozen SigLIP2, loaded on first use; kept out of the module tree and the checkpoint

        self.latent_queries = nn.Parameter(torch.randn(cfg.n_query, d) * 0.02)
        self.goal_xy_emb = FourierXY(d)
        dec = nn.TransformerDecoderLayer(d, nhead, 4 * d, cfg.dropout, batch_first=True, norm_first=True)
        self.latent_decoder = nn.TransformerDecoder(dec, cfg.latent_layers)
        self.latent_out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, cfg.latent_dim))

    # ------------------------------------------------------------------ io
    @classmethod
    def from_config(cls, cfg: LayaS2Config) -> "LayaS2":
        return cls(cfg, *build_encoders(cfg))

    @classmethod
    def from_pretrained(cls, ckpt_dir: str, map_location="cpu", **overrides) -> "LayaS2":
        """``overrides`` change config fields, e.g. ``grounding=True`` to add the grounding prior to a checkpoint
        trained without it (the new modules start at zero output, so the model behaves as before)."""
        from transformers import AutoConfig, AutoModel

        cfg = cls.config_class.load(ckpt_dir)
        for k, v in overrides.items():
            assert hasattr(cfg, k), f"unknown config field {k}"
            setattr(cfg, k, v)
        # architecture only; all weights come from the saved state dict
        text = AutoModel.from_config(
            AutoConfig.from_pretrained(os.path.join(ckpt_dir, "text_encoder")), attn_implementation="sdpa"
        )
        vision = AutoModel.from_config(AutoConfig.from_pretrained(os.path.join(ckpt_dir, "vision_encoder")))
        model = cls(cfg, text, vision)
        model.load_weights(os.path.join(ckpt_dir, WEIGHTS_NAME), map_location=map_location)
        return model

    def load_weights(self, path: str, map_location="cpu"):
        """Load a saved state dict. Tensors of the removed SigLIP pooling head (older checkpoints) are ignored, and
        add-on modules (grounding, matching) may be missing when they were just enabled on an older checkpoint."""
        sd = torch.load(path, map_location=map_location)
        missing, unexpected = self.load_state_dict(
            {k: v for k, v in sd.items() if not k.startswith(UNUSED_PREFIXES)}, strict=False
        )
        bad = [k for k in missing if not k.startswith(ADDON_PREFIXES)]
        if bad or unexpected:
            raise RuntimeError(f"checkpoint mismatch: missing {bad[:5]}, unexpected {unexpected[:5]}")

    def save_pretrained(self, out_dir: str):
        self.cfg.save(out_dir)
        self.text.config.save_pretrained(os.path.join(out_dir, "text_encoder"))
        self.vision.config.save_pretrained(os.path.join(out_dir, "vision_encoder"))
        torch.save(self.state_dict(), os.path.join(out_dir, WEIGHTS_NAME))

    # ------------------------------------------------------------------ helpers
    def goal_to_patch(self, goal_xy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Normalized goal [B, 2] -> (patch index [B], sub-patch offset in [0, 1] [B, 2])."""
        g = goal_xy.clamp(0, 1 - 1e-6) * self.grid
        cr = g.floor()
        idx = (cr[:, 1] * self.grid + cr[:, 0]).long()
        return idx, g - cr

    def patch_to_goal(self, idx: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        col = (idx % self.grid).float()
        row = torch.div(idx, self.grid, rounding_mode="floor").float()
        return torch.stack([col + offset[:, 0], row + offset[:, 1]], -1) / self.grid

    def encode_images(self, pixels: torch.Tensor) -> torch.Tensor:
        h = self.vision(pixel_values=pixels).last_hidden_state
        return h[:, -self.n_patch :]  # drop CLS / register tokens if the backbone has them

    @property
    def grounder(self):
        if not self._grounder:
            from .grounding import SiglipGrounder

            self._grounder.append(SiglipGrounder(self.cfg.grounding_model))
        g = self._grounder[0]
        dev = self.vis_proj[0].weight.device
        if next(g.parameters()).device != dev:
            g.to(dev)
        return g

    def set_grounder(self, grounder: nn.Module):
        """Use this grounder instead of loading SigLIP2 (tests, or sharing one grounder between models)."""
        self._grounder[:] = [grounder]

    def grounding_features(self, chunks, cur_pixels, down_pixels):
        """Phrase-match maps -> additive embeddings for the current-frame (pooled) and look-down tokens."""
        from .grounding import grounding_maps

        m = self.cfg.grounding_chunks
        maps = grounding_maps(self.grounder, chunks, m, torch.stack([cur_pixels, down_pixels], 1))  # [B, 2, P, m]
        g = self.grid
        cur = maps[:, 0].transpose(1, 2).reshape(-1, m, g, g)
        cur = F.adaptive_avg_pool2d(cur, self.cfg.cur_pool).flatten(2).transpose(1, 2)  # [B, cur_pool^2, m]
        return self.ground_proj(cur), self.ground_proj(maps[:, 1])

    def _pool(self, tokens: torch.Tensor, size: int) -> torch.Tensor:
        n, _, c = tokens.shape
        if n == 0:
            return tokens.new_zeros(0, size * size, c)
        x = tokens.transpose(1, 2).reshape(n, c, self.grid, self.grid)
        return F.adaptive_avg_pool2d(x.float(), size).to(tokens.dtype).flatten(2).transpose(1, 2)

    # ------------------------------------------------------------------ forward
    def forward(
        self,
        input_ids: torch.Tensor,  # [B, Lt] instruction, right-padded
        text_mask: torch.Tensor,  # [B, Lt]
        hist_pixels: torch.Tensor,  # [B, H, 3, S, S]
        hist_mask: torch.Tensor,  # [B, H] valid history frames
        cur_pixels: torch.Tensor,  # [B, 3, S, S]
        down_pixels: torch.Tensor,  # [B, 3, S, S]
        goal_xy: Optional[torch.Tensor] = None,  # [B, 2] teacher-forced goal for the latent head
        chunks: Optional[List[List[str]]] = None,  # instruction phrases (grounding.split_instruction), if grounding
    ) -> Dict[str, torch.Tensor]:
        B, H = hist_mask.shape
        dev = input_ids.device
        hist_mask = hist_mask.bool()

        # one vision pass over every real frame
        n_hist = int(hist_mask.sum())
        frames = torch.cat([hist_pixels[hist_mask], cur_pixels, down_pixels], 0)
        vis = self.vis_proj(self.encode_images(frames))
        hist_tok = self._pool(vis[:n_hist], self.cfg.hist_pool) + self.pos_hist + self.seg_emb.weight[SEG_HIST]
        cur_tok = self._pool(vis[n_hist : n_hist + B], self.cfg.cur_pool) + self.pos_cur + self.seg_emb.weight[SEG_CUR]
        down_tok = vis[n_hist + B :] + self.pos_down + self.seg_emb.weight[SEG_DOWN]
        hist_tok = hist_tok + self.hist_slot_emb(hist_mask.nonzero()[:, 1])[:, None]
        if self.cfg.grounding:
            assert chunks is not None and len(chunks) == B, "grounding needs the instruction phrases (chunks)"
            g_cur, g_down = self.grounding_features(chunks, cur_pixels, down_pixels)
            cur_tok = cur_tok + g_cur.to(cur_tok.dtype)
            down_tok = down_tok + g_down.to(down_tok.dtype)

        text_emb = self.text.get_input_embeddings()(input_ids) + self.seg_emb.weight[SEG_TEXT]
        act_tok = self.action_marker + self.action_emb.weight + self.seg_emb.weight[SEG_ACTION]
        dtype = text_emb.dtype

        seqs, a_start, d_start = [], [], []
        hist_split = torch.split(hist_tok, hist_mask.sum(1).tolist())
        for b in range(B):
            t = text_emb[b, : int(text_mask[b].sum())]
            parts = [t, act_tok.to(dtype), hist_split[b].flatten(0, 1).to(dtype), cur_tok[b].to(dtype)]
            a_start.append(t.shape[0])
            d_start.append(sum(p.shape[0] for p in parts))
            seqs.append(torch.cat(parts + [down_tok[b].to(dtype)], 0))
        lens = torch.tensor([s.shape[0] for s in seqs], device=dev)
        emb = nn.utils.rnn.pad_sequence(seqs, batch_first=True)
        attn = torch.arange(emb.shape[1], device=dev)[None] < lens[:, None]

        h = self.text(inputs_embeds=emb, attention_mask=attn.long()).last_hidden_state
        pad = ~attn
        if self.head is not None:
            h = self.head(h, src_key_padding_mask=pad)

        a_idx = torch.tensor(a_start, device=dev)[:, None] + torch.arange(self.n_action, device=dev)
        d_idx = torch.tensor(d_start, device=dev)[:, None] + torch.arange(self.n_patch, device=dev)
        h_act = torch.gather(h, 1, a_idx[..., None].expand(-1, -1, self.d))
        h_down = torch.gather(h, 1, d_idx[..., None].expand(-1, -1, self.d))

        logits = torch.cat([self.scorer(h_act), self.scorer(h_down)], 1).squeeze(-1).float()  # [B, A + P]
        offsets = torch.sigmoid(self.offset_head(h_down).float())  # [B, P, 2]

        # act head sees the pooled sequence + detached summary of its own answer distribution (as in Laya)
        p = torch.softmax(logits.detach(), -1)
        ent = -(p * torch.log(p.clamp_min(1e-9))).sum(-1) / math.log(p.shape[-1])
        top2 = p.topk(2, -1).values
        goal_mass = p[:, self.n_action :].sum(-1)
        feats = torch.stack([top2[:, 0], top2[:, 0] - top2[:, 1], ent, goal_mass], -1)
        act_logits = self.act_head(torch.cat([h[:, 0].float(), feats], -1).to(h.dtype)).float()

        # goal-conditioned latent for System 1 (teacher-forced in training, own prediction at inference)
        if goal_xy is None:
            patch_idx = logits[:, self.n_action :].argmax(-1)
            goal_xy = self.patch_to_goal(patch_idx, offsets[torch.arange(B, device=dev), patch_idx])
        else:
            patch_idx, _ = self.goal_to_patch(goal_xy)
        goal_tok = h_down[torch.arange(B, device=dev), patch_idx] + self.goal_xy_emb(goal_xy).to(h.dtype)
        tgt = torch.cat([goal_tok[:, None], self.latent_queries.to(h.dtype).expand(B, -1, -1)], 1)
        lat = self.latent_decoder(tgt, h, memory_key_padding_mask=pad)[:, 1:]
        latent = self.latent_out(lat)

        extra = {}
        if self.cfg.match_head:
            extra["match_logit"] = self.match_head(h[:, 0]).squeeze(-1).float()
        return dict(
            **extra,
            logits=logits,
            offsets=offsets,
            act_logits=act_logits,
            latent=latent,
            goal_xy=goal_xy,
            goal_token=goal_tok,  # fused hidden state of the chosen goal patch (+ goal position)
            down_feat=vis[n_hist + B :],  # look-down frame features before fusion, reusable by a planner
        )

    @torch.no_grad()
    def decide(self, out: Dict[str, torch.Tensor]) -> List[Dict]:
        """Turn forward outputs into per-sample decisions."""
        res = []
        p = torch.softmax(out["logits"], -1)
        esc = torch.softmax(out["act_logits"], -1)[:, 1]
        for b, k in enumerate(p.argmax(-1).tolist()):
            d = dict(confidence=float(p[b, k]), escalate_prob=float(esc[b]))
            if k < self.n_action:
                d.update(kind="action", action=self.cfg.actions[k])
            else:
                d.update(kind="goal", goal_xy=out["goal_xy"][b].tolist(), latent=out["latent"][b])
            res.append(d)
        return res


# ---------------------------------------------------------------------- training helpers
def instruction_chunks(model: LayaS2, instructions: List[str]):
    """Instruction phrases for the grounding prior, or None when the model has no grounding."""
    if not model.cfg.grounding:
        return None
    from .grounding import split_instruction

    return [split_instruction(t, model.cfg.grounding_chunks) for t in instructions]


def mismatched(instructions: List[str]):
    """Index of another sample's instruction for each sample (next different one), or -1 if there is none."""
    n, out = len(instructions), []
    for i in range(n):
        j = next((k % n for k in range(i + 1, i + n) if instructions[k % n] != instructions[i]), -1)
        out.append(j)
    return out


def match_loss(pos_logit: torch.Tensor, neg_logit: torch.Tensor, valid: torch.Tensor):
    """Binary cross-entropy: true (instruction, view) pairs vs pairs with another sample's instruction."""
    zero = pos_logit.sum() * 0
    if not valid.any():
        return zero, dict(l_match=zero.detach(), acc_match=zero.detach())
    p, n = pos_logit[valid], neg_logit[valid]
    loss = 0.5 * (F.binary_cross_entropy_with_logits(p, torch.ones_like(p))
                  + F.binary_cross_entropy_with_logits(n, torch.zeros_like(n)))
    acc = 0.5 * ((p > 0).float().mean() + (n < 0).float().mean())
    return loss, dict(l_match=loss.detach(), acc_match=acc.detach())


# ---------------------------------------------------------------------- loss
@dataclass
class LossWeights:
    decision: float = 1.0
    spherical: float = 0.5  # Laya: log score + spherical score, both strictly proper
    offset: float = 1.0
    latent: float = 1.0
    escalate: float = 0.1
    match: float = 0.0  # instruction / view matching against shuffled instructions (needs match_head)
    goal_sigma: float = 0.75  # soft target over neighbouring patches, in patch units
    correct_radius: float = 1.5  # predicted goal within this many patches counts as correct


def compute_loss(model: LayaS2, out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], w: LossWeights):
    logits = out["logits"]
    B, K = logits.shape
    A, g = model.n_action, model.grid
    dev = logits.device
    is_goal = batch["is_goal"].bool()

    # target distribution: one-hot action, or a gaussian bump over look-down patches around the goal
    target = torch.zeros(B, K, device=dev)
    act_rows = (~is_goal).nonzero()[:, 0]
    target[act_rows, batch["action_idx"][act_rows]] = 1.0
    goal_idx, goal_off = model.goal_to_patch(batch["goal_xy"])
    if is_goal.any():
        ys, xs = torch.meshgrid(torch.arange(g, device=dev), torch.arange(g, device=dev), indexing="ij")
        centers = torch.stack([xs, ys], -1).reshape(-1, 2).float() + 0.5
        d2 = ((centers[None] - batch["goal_xy"][:, None] * g) ** 2).sum(-1)
        bump = torch.softmax(-d2 / (2 * w.goal_sigma**2), -1)
        target[is_goal, A:] = bump[is_goal]

    logp = torch.log_softmax(logits, -1)
    q = logp.exp()
    log_score = -(target * logp).sum(-1)
    spherical = -(target * q).sum(-1) / q.norm(dim=-1).clamp_min(1e-9) / target.norm(dim=-1).clamp_min(1e-9)
    l_dec = (log_score + w.spherical * spherical).mean()

    zero = logits.new_zeros(())
    rows = torch.arange(B, device=dev)
    pred = logits.argmax(-1)
    if is_goal.any():
        pred_off = out["offsets"][rows, goal_idx]
        l_off = F.l1_loss(pred_off[is_goal], goal_off[is_goal])
    else:
        l_off = zero

    lat_mask = batch["latent_mask"].bool() & is_goal
    if lat_mask.any():
        p_lat, t_lat = out["latent"][lat_mask].float(), batch["latent"][lat_mask].float()
        l_lat = F.mse_loss(p_lat, t_lat) + (1 - F.cosine_similarity(p_lat, t_lat, dim=-1)).mean()
        lat_cos = F.cosine_similarity(p_lat, t_lat, dim=-1).mean().detach()
    else:
        l_lat, lat_cos = zero, zero

    # escalate target: was the model's own top answer wrong?
    with torch.no_grad():
        pred_patch = (pred - A).clamp(min=0)
        pred_xy = model.patch_to_goal(pred_patch, out["offsets"][rows, pred_patch])
        goal_ok = (pred >= A) & ((pred_xy - batch["goal_xy"]).norm(dim=-1) * g <= w.correct_radius)
        act_ok = pred == batch["action_idx"]
        correct = torch.where(is_goal, goal_ok, act_ok)
    l_esc = F.cross_entropy(out["act_logits"], (~correct).long())

    loss = w.decision * l_dec + w.offset * l_off + w.latent * l_lat + w.escalate * l_esc
    stats = dict(
        loss=loss.detach(),
        l_dec=l_dec.detach(),
        l_off=l_off.detach(),
        l_lat=l_lat.detach(),
        l_esc=l_esc.detach(),
        acc=correct.float().mean(),
        acc_goal=correct[is_goal].float().mean() if is_goal.any() else zero,
        acc_action=correct[~is_goal].float().mean() if (~is_goal).any() else zero,
        latent_cos=lat_cos,
        # sample counts, so that evaluation can weight the subset metrics correctly
        n=logits.new_tensor(float(B)),
        n_goal=is_goal.sum().float(),
        n_action=(~is_goal).sum().float(),
        n_lat=lat_mask.sum().float(),
    )
    return loss, stats
