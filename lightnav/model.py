"""LayaNative: a Laya `DecisionModel` that also reads image tokens.

The text half of the sequence is built by `laya.common.build_sequence` and left untouched. Image tokens are
inserted just before the closing [SEP], so the option markers keep their positions:

    [CLS] question [SEP] [MASK] option x10 [SEP] state (instruction) | image tokens [SEP]

`laya` itself is not modified. `forward_embeds` repeats `DecisionModel.forward` (laya/common.py) with
`inputs_embeds` in place of `input_ids`, which the original does not accept.

Landmark tags (optional, `tags_current` / `tags_history`, design B 7.2.3): SigLIP2 names what the frames show
with its own image-text alignment: the pooled image embedding against a fixed table of landmark phrases made once
by SigLIP2's text tower, each phrase's score normalised by its mean and spread over training frames (otherwise
"foyer" or "basement" wins everywhere). The best words of the current frame and of each past frame go right after
the instruction as ordinary word embeddings of the encoder, so "bathroom" in the instruction and "bathroom" seen
are the same token. At inference only the table and the statistics are needed (no text tower):

    ... state (instruction) | seen: w1, w2, ... | image tokens [SEP]
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
    cur_grid: int = 0                   # the current frame becomes cur_grid x cur_grid tokens; 0 = every patch
    text_max_len: int = 448             # question + options + instruction, in tokens
    head_max_len: int = 192             # question + options
    tags_current: int = 0               # landmark words named for the current frame; 0 and 0 = no tags
    tags_history: int = 0               # landmark words named for each past frame
    tag_vocab: List[str] = field(default_factory=list)   # the phrases; filled by `build` from lightnav.tasks


def tag_vocabulary() -> List[str]:
    """Landmark phrases for the tags: lightnav.tasks' rooms, room kinds ("living room") and objects."""
    from .tasks import tag_phrases

    return tag_phrases()


def tag_prompt(phrase: str) -> str:
    from .tasks import OBJECTS_PLURAL

    return ("a photo of %s." if phrase in OBJECTS_PLURAL else "a photo of a %s.") % phrase


@torch.no_grad()
def siglip_text_table(siglip, name: str, phrases: List[str]) -> torch.Tensor:
    """[len(phrases), dim] unit vectors from SigLIP's text tower (padding to 64 tokens, as SigLIP was trained)."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    t = tok([tag_prompt(p) for p in phrases], padding="max_length", max_length=64, truncation=True,
            return_tensors="pt")
    emb = siglip.get_text_features(input_ids=t["input_ids"]).float()
    return F.normalize(emb, dim=-1)


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
        if cfg.cur_grid and self.grid % cfg.cur_grid:
            raise ValueError("cur_grid %d does not divide the %d x %d patch grid" % (cfg.cur_grid, self.grid, self.grid))
        cur_n = (cfg.cur_grid or self.grid) ** 2
        self.vis_proj = nn.Sequential(nn.Linear(dv, d), nn.GELU(), nn.Linear(d, d))
        # Slot num_frames - 1 is always the current frame, so an early step (fewer than 8 frames) and a
        # late one give the current view the same embedding.
        self.frame_emb = nn.Embedding(cfg.num_frames, d)
        self.pos_cur = nn.Parameter(torch.zeros(cur_n, d))
        self.pos_hist = nn.Parameter(torch.zeros(cfg.hist_grid * cfg.hist_grid, d))
        for p in (self.frame_emb.weight, self.pos_cur, self.pos_hist):
            nn.init.normal_(p, std=0.02)
        self.tagging = bool(cfg.tags_current or cfg.tags_history)
        if self.tagging:
            if not cfg.tag_vocab:
                raise ValueError("tags need cfg.tag_vocab")
            if getattr(self.vision, "head", None) is None:
                raise ValueError("tags need the vision tower's pooling head")
            v = len(cfg.tag_vocab)
            self.register_buffer("tag_text", torch.zeros(v, dv))         # saved with the model
            self.register_buffer("tag_mu", torch.zeros(v))               # score statistics over training frames
            self.register_buffer("tag_sd", torch.ones(v))
            for name in ("tag_ids", "tag_prefix", "tag_sep"):            # from the tokenizer (set_tag_tokens)
                self.register_buffer(name, torch.full((1, 1), -1, dtype=torch.long), persistent=False)
            self.tag_type = nn.Parameter(torch.zeros(2, d))       # 0: tag of a past frame, 1: of the current one
            nn.init.normal_(self.tag_type, std=0.02)

    # ------------------------------------------------------------------------------- vision

    def encode_frames(self, pixels: torch.Tensor) -> torch.Tensor:
        """Patch features [N, grid * grid, dv] of frames [N, 3, H, W]."""
        return self.vision(pixel_values=pixels).last_hidden_state

    def pool_history(self, feats: torch.Tensor, grid: Optional[int] = None) -> torch.Tensor:
        """Frame features averaged down to [N, grid * grid, dv] (default: hist_grid, for past frames)."""
        grid = grid or self.cfg.hist_grid
        n, _, dv = feats.shape
        x = feats.view(n, self.grid, self.grid, dv).permute(0, 3, 1, 2)
        x = F.avg_pool2d(x, self.grid // grid)
        return x.flatten(2).transpose(1, 2)

    def set_tag_tokens(self, tok) -> None:
        """Encoder token ids of every tag phrase (as it reads after a space), of " seen:" and of ","."""
        ids = [tok(" " + p, add_special_tokens=False)["input_ids"] for p in self.cfg.tag_vocab]
        t = torch.full((len(ids), max(len(x) for x in ids)), -1, dtype=torch.long)
        for i, x in enumerate(ids):
            t[i, :len(x)] = torch.tensor(x, dtype=torch.long)
        dev = self.tag_text.device
        self.tag_ids = t.to(dev)
        self.tag_prefix = torch.tensor(tok(" seen:", add_special_tokens=False)["input_ids"], device=dev)
        self.tag_sep = torch.tensor(tok(",", add_special_tokens=False)["input_ids"], device=dev)

    @torch.no_grad()
    def tag_scores(self, feats: torch.Tensor, normalise: bool = True) -> torch.Tensor:
        """[N, vocab] score of each tag phrase for each frame: cosine similarity of the pooled SigLIP embedding,
        normalised by the phrase's statistics over training frames (`set_tag_stats`)."""
        pooled = self.vision.head(feats.to(next(self.vision.head.parameters()).dtype))
        cos = F.normalize(pooled.float(), dim=-1) @ self.tag_text.T
        return (cos - self.tag_mu) / self.tag_sd if normalise else cos

    @torch.no_grad()
    def set_tag_stats(self, cos: torch.Tensor) -> None:
        """Statistics from raw scores [N frames, vocab] (`tag_scores(..., normalise=False)` of training frames)."""
        self.tag_mu.copy_(cos.mean(0))
        self.tag_sd.copy_(cos.std(0).clamp(min=1e-4))

    def row_tags(self, feats, slots, owner, n_rows: int, override=None) -> List[List]:
        """Per row, the tag words `[(vocab index, from the current frame)]`: the current frame's best
        `tags_current`, then each past frame's best `tags_history` in slot order, without repeats. A row of
        `override` ([B, K] vocab indices, -1 padding) replaces them (the matching task's separable warm-up)."""
        cur = self.cfg.num_frames - 1
        full_n = self.grid * self.grid
        if torch.is_tensor(feats):
            full = [i for i in range(len(feats))] if feats.shape[1] == full_n else []
            scores = self.tag_scores(feats) if full else None
        else:                                   # the agent's cache: pooled past frames have no tags
            full = [i for i, f in enumerate(feats) if f.shape[0] == full_n]
            scores = self.tag_scores(torch.stack([feats[i] for i in full])) if full else None
        k = max(self.cfg.tags_current, self.cfg.tags_history)
        best = scores.topk(k, dim=-1).indices.tolist() if scores is not None else []
        best = dict(zip(full, best))
        slots, owner = slots.tolist(), owner.tolist()
        frames = [[] for _ in range(n_rows)]
        for i, (s, o) in enumerate(zip(slots, owner)):
            frames[o].append((s, i))
        out = []
        for b in range(n_rows):
            if override is not None and (override[b] >= 0).any():
                out.append([(int(v), False) for v in override[b].tolist() if v >= 0])
                continue
            words, seen = [], set()
            for s, i in sorted(frames[b], key=lambda x: (x[0] != cur, x[0])):    # current frame first
                if i not in best:
                    continue
                n = self.cfg.tags_current if s == cur else self.cfg.tags_history
                for v in best[i][:n]:
                    if v not in seen:
                        seen.add(v)
                        words.append((v, s == cur))
            out.append(words)
        return out

    def tag_segment(self, words) -> Optional[torch.Tensor]:
        """Embeddings of " seen: w1, w2, ..." for one row ([n tokens, d]); None without words."""
        if not words:
            return None
        ids, types = list(self.tag_prefix.tolist()), [-1] * len(self.tag_prefix)
        sep = self.tag_sep.tolist()
        for j, (v, current) in enumerate(words):
            w = [t for t in self.tag_ids[v].tolist() if t >= 0]
            ids += w
            types += [int(current)] * len(w)
            if j + 1 < len(words):
                ids += sep
                types += [-1] * len(sep)
        ids = torch.tensor(ids, device=self.tag_type.device)
        types = torch.tensor(types, device=self.tag_type.device)
        emb = self.dm.encoder.get_input_embeddings()(ids)
        return emb + self.tag_type[types.clamp(min=0)] * (types >= 0)[:, None]

    def frame_tokens(self, feats: torch.Tensor, slots: torch.Tensor) -> List[torch.Tensor]:
        """Image tokens of each frame: the current frame keeps every patch, past frames are pooled.

        `feats` is [N, grid * grid, dv] for every frame, or a list holding, per frame, either that or an
        already pooled [hist_grid * hist_grid, dv] (what the agent caches for past frames).
        """
        cur = self.cfg.num_frames - 1
        out = []
        for f, s in zip(feats, slots.tolist()):
            if s == cur:
                if self.cfg.cur_grid and f.shape[0] != self.pos_cur.shape[0]:
                    f = self.pool_history(f[None], self.cfg.cur_grid)[0]
                tok = self.vis_proj(f) + self.pos_cur
            else:
                if f.shape[0] != self.pos_hist.shape[0]:
                    f = self.pool_history(f[None])[0]
                tok = self.vis_proj(f) + self.pos_hist
            out.append(tok + self.frame_emb.weight[s])
        return out

    # ----------------------------------------------------------------------------- sequence

    def assemble(self, input_ids, text_len, tokens: List[torch.Tensor], frame_owner, tags=None):
        """`(inputs_embeds, attention_mask)` with each sample's image tokens placed before its last [SEP], and its
        tag segment (`tags[b]`, optional) between the text and the image tokens."""
        emb = self.dm.encoder.get_input_embeddings()(input_ids)
        owner = frame_owner.tolist()
        rows = []
        for b in range(input_ids.size(0)):
            n = int(text_len[b])
            vis = [t.to(emb.dtype) for t, o in zip(tokens, owner) if o == b]
            seg = [tags[b].to(emb.dtype)] if tags is not None and tags[b] is not None else []
            rows.append(torch.cat([emb[b, :n - 1]] + seg + vis + [emb[b, n - 1:n]]))
        length = max(r.size(0) for r in rows)
        embeds = emb.new_zeros(len(rows), length, emb.size(-1))
        mask = torch.zeros(len(rows), length, dtype=torch.long, device=emb.device)
        for b, r in enumerate(rows):
            embeds[b, :r.size(0)] = r
            mask[b, :r.size(0)] = 1
        return embeds, mask

    def forward(self, batch: Dict, feats=None) -> torch.Tensor:
        """Logits [B, 10]: over `questions.ANSWERS` for a navigation sample; a task with fewer options (padded
        markers, `marker_mask` False) gets -1e4 in the unused columns. `feats` skips the vision encoder."""
        if feats is None:
            feats = self.encode_frames(batch["pixels"])
        tokens = self.frame_tokens(feats, batch["frame_slot"])
        tags = None
        if self.tagging:
            words = self.row_tags(feats, batch["frame_slot"], batch["frame_owner"], batch["input_ids"].size(0),
                                  batch.get("tag_override"))
            tags = [self.tag_segment(w) for w in words]
        embeds, mask = self.assemble(batch["input_ids"], batch["text_len"], tokens, batch["frame_owner"], tags)
        marker_pos = batch["marker_pos"]
        marker_mask = batch.get("marker_mask")
        if marker_mask is None:
            marker_mask = marker_pos >= 0
        qtype = torch.zeros(marker_pos.size(0), dtype=torch.long, device=marker_pos.device)   # 0 = choice
        return forward_embeds(self.dm, embeds, mask, marker_pos, marker_mask, qtype)

    # ----------------------------------------------------------------------------- training

    def new_parameters(self):
        """The parts Laya and SigLIP2 do not bring: projection, frame and position embeddings (and tag types)."""
        extra = [self.tag_type] if self.tagging else []
        return [*self.vis_proj.parameters(), self.frame_emb.weight, self.pos_cur, self.pos_hist, *extra]

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
    siglip = AutoModel.from_pretrained(cfg.vision_encoder)
    if (cfg.tags_current or cfg.tags_history) and not cfg.tag_vocab:
        cfg.tag_vocab = tag_vocabulary()
    model = LayaNative(cfg, dm, siglip)
    if model.tagging:
        model.tag_text.copy_(siglip_text_table(siglip, cfg.vision_encoder, cfg.tag_vocab))
        model.set_tag_tokens(tok)
    return model, tok


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
    tok = AutoTokenizer.from_pretrained(os.path.join(path, "tokenizer"))
    if model.tagging:
        model.set_tag_tokens(tok)
    return model, tok
