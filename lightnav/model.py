"""LayaNative: a Laya `DecisionModel` that also reads image tokens.

The text half of the sequence is built by `laya.common.build_sequence` and left untouched. Image tokens are
inserted just before the closing [SEP], so the option markers keep their positions:

    [CLS] question [SEP] [MASK] option x10 [SEP] state (instruction) | image tokens [SEP]

`laya` itself is not modified. `forward_embeds` repeats `DecisionModel.forward` (laya/common.py) with
`inputs_embeds` in place of `input_ids`, which the original does not accept.
"""
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .questions import NUM_ACTIONS


@dataclass
class LayaNativeConfig:
    laya: Optional[str] = "english"     # Laya checkpoint to start from: alias, Hub id or directory
    encoder: Optional[str] = None       # used when `laya` is None: a plain encoder, e.g. answerdotai/ModernBERT-large
    head_layers: int = 2                # only for the plain-encoder start; a Laya checkpoint brings its own
    vision_encoder: str = "google/siglip2-base-patch16-224"
    image_size: int = 224
    image_mean: List[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])
    image_std: List[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])
    num_frames: int = 8                 # NaVILA: 7 past frames + the current one
    hist_grid: int = 7                  # a past frame becomes hist_grid x hist_grid tokens
    text_max_len: int = 448             # question + options + instruction, in tokens
    head_max_len: int = 192             # question + options


def forward_embeds(dm, inputs_embeds, attention_mask, marker_pos, marker_mask, qtype):
    """Option logits of a Laya `DecisionModel` for a sequence given as embeddings.

    The same computation as `DecisionModel.forward` on its default (sequential layout) path, minus the
    act head, which this project does not use.
    """
    h = dm.encoder(inputs_embeds=inputs_embeds, attention_mask=attention_mask).last_hidden_state
    h = h + dm.type_emb(qtype)[:, None, :]
    if dm.head is not None:
        pad = ~attention_mask.bool()
        for layer in dm.head.layers:
            if dm.head_checkpointing and dm.training and torch.is_grad_enabled():
                h = checkpoint(layer, h, src_key_padding_mask=pad, use_reentrant=False)
            else:
                h = layer(h, src_key_padding_mask=pad)
    idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
    logits = dm.scorer(torch.gather(h, 1, idx)).squeeze(-1).float()
    return logits.masked_fill(~marker_mask, -1e4)


def _vision_tower(model):
    return getattr(model, "vision_model", model)


class LayaNative(nn.Module):
    def __init__(self, cfg: LayaNativeConfig, dm, vision):
        super().__init__()
        self.cfg, self.dm, self.vision = cfg, dm, _vision_tower(vision)
        # ModernBERT may compile its embedding and MLP blocks on first use; keep every path plain.
        if hasattr(dm.encoder.config, "reference_compile"):
            dm.encoder.config.reference_compile = False
        d, dv = dm.encoder.config.hidden_size, self.vision.config.hidden_size
        self.grid = self.vision.config.image_size // self.vision.config.patch_size
        if self.grid % cfg.hist_grid:
            raise ValueError("hist_grid %d does not divide the %d x %d patch grid" % (cfg.hist_grid, self.grid, self.grid))
        self.vis_proj = nn.Sequential(nn.Linear(dv, d), nn.GELU(), nn.Linear(d, d))
        # Slot num_frames - 1 is always the current frame, so an early step (fewer than 8 frames) and a
        # late one give the current view the same embedding.
        self.frame_emb = nn.Embedding(cfg.num_frames, d)
        self.pos_cur = nn.Parameter(torch.zeros(self.grid * self.grid, d))
        self.pos_hist = nn.Parameter(torch.zeros(cfg.hist_grid * cfg.hist_grid, d))
        for p in (self.frame_emb.weight, self.pos_cur, self.pos_hist):
            nn.init.normal_(p, std=0.02)

    # ------------------------------------------------------------------------------- vision

    def encode_frames(self, pixels: torch.Tensor) -> torch.Tensor:
        """Patch features [N, grid * grid, dv] of frames [N, 3, H, W]."""
        return self.vision(pixel_values=pixels).last_hidden_state

    def pool_history(self, feats: torch.Tensor) -> torch.Tensor:
        """Past-frame features averaged down to [N, hist_grid * hist_grid, dv]."""
        n, _, dv = feats.shape
        x = feats.view(n, self.grid, self.grid, dv).permute(0, 3, 1, 2)
        x = F.avg_pool2d(x, self.grid // self.cfg.hist_grid)
        return x.flatten(2).transpose(1, 2)

    def frame_tokens(self, feats: torch.Tensor, slots: torch.Tensor) -> List[torch.Tensor]:
        """Image tokens of each frame: the current frame keeps every patch, past frames are pooled.

        `feats` is [N, grid * grid, dv] for every frame, or a list holding, per frame, either that or an
        already pooled [hist_grid * hist_grid, dv] (what the agent caches for past frames).
        """
        cur = self.cfg.num_frames - 1
        out = []
        for f, s in zip(feats, slots.tolist()):
            if s == cur:
                tok = self.vis_proj(f) + self.pos_cur
            else:
                if f.shape[0] != self.pos_hist.shape[0]:
                    f = self.pool_history(f[None])[0]
                tok = self.vis_proj(f) + self.pos_hist
            out.append(tok + self.frame_emb.weight[s])
        return out

    # ----------------------------------------------------------------------------- sequence

    def assemble(self, input_ids, text_len, tokens: List[torch.Tensor], frame_owner):
        """`(inputs_embeds, attention_mask)` with each sample's image tokens placed before its last [SEP]."""
        emb = self.dm.encoder.get_input_embeddings()(input_ids)
        owner = frame_owner.tolist()
        rows = []
        for b in range(input_ids.size(0)):
            n = int(text_len[b])
            vis = [t.to(emb.dtype) for t, o in zip(tokens, owner) if o == b]
            rows.append(torch.cat([emb[b, :n - 1]] + vis + [emb[b, n - 1:n]]))
        length = max(r.size(0) for r in rows)
        embeds = emb.new_zeros(len(rows), length, emb.size(-1))
        mask = torch.zeros(len(rows), length, dtype=torch.long, device=emb.device)
        for b, r in enumerate(rows):
            embeds[b, :r.size(0)] = r
            mask[b, :r.size(0)] = 1
        return embeds, mask

    def forward(self, batch: Dict, feats=None) -> torch.Tensor:
        """Logits [B, 10] over `questions.ANSWERS`. `feats` skips the vision encoder (see `frame_tokens`)."""
        if feats is None:
            feats = self.encode_frames(batch["pixels"])
        tokens = self.frame_tokens(feats, batch["frame_slot"])
        embeds, mask = self.assemble(batch["input_ids"], batch["text_len"], tokens, batch["frame_owner"])
        marker_pos = batch["marker_pos"]
        marker_mask = torch.ones_like(marker_pos, dtype=torch.bool)
        qtype = torch.zeros(marker_pos.size(0), dtype=torch.long, device=marker_pos.device)   # 0 = choice
        return forward_embeds(self.dm, embeds, mask, marker_pos, marker_mask, qtype)

    # ----------------------------------------------------------------------------- training

    def new_parameters(self):
        """The parts Laya and SigLIP2 do not bring: projection, frame and position embeddings."""
        return [*self.vis_proj.parameters(), self.frame_emb.weight, self.pos_cur, self.pos_hist]

    def set_stage(self, stage: str) -> None:
        """"align": only the new parts train, Laya and the vision encoder are frozen. "nav": everything trains."""
        if stage not in ("align", "nav"):
            raise ValueError("stage must be 'align' or 'nav', got %r" % (stage,))
        for p in self.parameters():
            p.requires_grad_(stage == "nav")
        for p in self.new_parameters():
            p.requires_grad_(True)

    def param_groups(self, encoder_lr: float, head_lr: float, vision_lr: float):
        new = {id(p) for p in self.new_parameters()}
        enc = [p for p in self.dm.encoder.parameters() if p.requires_grad]
        vis = [p for p in self.vision.parameters() if p.requires_grad]
        skip = new | {id(p) for p in enc} | {id(p) for p in vis}
        head = [p for p in self.dm.parameters() if p.requires_grad and id(p) not in skip]
        groups = [{"params": enc, "lr": encoder_lr}, {"params": vis, "lr": vision_lr},
                  {"params": head + [p for p in self.new_parameters() if p.requires_grad], "lr": head_lr}]
        return [g for g in groups if g["params"]]


# ----------------------------------------------------------------------------------- build / io

def build(cfg: LayaNativeConfig):
    """`(model, tokenizer)` from pretrained parts: a Laya checkpoint (or a plain encoder) and the vision encoder."""
    from transformers import AutoModel, AutoTokenizer

    if cfg.laya:
        from laya.train import load_checkpoint
        dm, tok, _ = load_checkpoint(cfg.laya)
    else:
        from laya.common import build_model
        dm = build_model({"encoder": cfg.encoder, "head_layers": cfg.head_layers})
        tok = AutoTokenizer.from_pretrained(cfg.encoder)
    return LayaNative(cfg, dm, AutoModel.from_pretrained(cfg.vision_encoder)), tok


def save(model: LayaNative, tok, path: str) -> None:
    from safetensors.torch import save_file

    os.makedirs(path, exist_ok=True)
    save_file({k: v.detach().float().cpu().contiguous() for k, v in model.state_dict().items()},
              os.path.join(path, "model.safetensors"))
    model.dm.encoder.config.save_pretrained(os.path.join(path, "encoder"))
    model.vision.config.save_pretrained(os.path.join(path, "vision"))
    tok.save_pretrained(os.path.join(path, "tokenizer"))
    meta = {"config": asdict(model.cfg), "head_layers": len(model.dm.head.layers) if model.dm.head is not None else 0,
            "n_act": model.dm.act_head[-1].out_features, "num_actions": NUM_ACTIONS}
    with open(os.path.join(path, "lightnav_config.json"), "w") as f:
        json.dump(meta, f, indent=2)


def load(path: str):
    """`(model, tokenizer)` from a directory written by `save`. Nothing is downloaded."""
    from laya.common import DecisionModel
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    with open(os.path.join(path, "lightnav_config.json")) as f:
        meta = json.load(f)
    enc = AutoModel.from_config(AutoConfig.from_pretrained(os.path.join(path, "encoder")), attn_implementation="sdpa")
    vision = AutoModel.from_config(AutoConfig.from_pretrained(os.path.join(path, "vision")))
    model = LayaNative(LayaNativeConfig(**meta["config"]), DecisionModel(enc, meta["head_layers"], meta["n_act"]), vision)
    model.load_state_dict(load_file(os.path.join(path, "model.safetensors")), strict=True)
    return model, AutoTokenizer.from_pretrained(os.path.join(path, "tokenizer"))
