"""Grounding prior for Laya-S2 / LayaNav: where do the instruction's phrases appear in the frames?

The released SigLIP2 aligns text with the pooled image embedding only. Its attention-pooling head applied to
every patch on its own (attention fixed to the patch itself, as MaskCLIP does for CLIP) gives one text-aligned
vector per patch, so the cosine with a phrase embedding is a per-patch "this phrase is here" map. On the KIMM
Isaac Sim views these maps lit up the named objects (bookshelf, desk, TV, sofa, air conditioner, ...) while
LayaNav's own decision barely changed with the object named in the instruction.

The grounder is frozen and not part of the checkpoint (it is loaded from the Hugging Face cache by name).
"""

import re
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_SPLIT = re.compile(r"[.;,!?]|\band then\b|\bthen\b|\band\b|\bafter\b|\bonce\b|\buntil\b", re.IGNORECASE)


# leading verbs and prepositions of a navigation phrase ("stop next to the", "walk past the"); SigLIP2 matches object
# names well and whole sentences poorly, and the directions themselves are read by the text encoder anyway
_LEAD = re.compile(
    r"^(?:(?:please|now|you|will|should|then|and)\s+)*"
    r"(?:(?:walk|go|head|move|turn|stop|wait|continue|proceed|keep|exit|leave|enter|pass|climb|take|step|make|follow|"
    r"get|come|stand|face)\w*\s+)?"
    r"(?:(?:left|right|straight|forward|ahead|around|back|down|up|slightly|all the way|a|an|the)\s+)*"
    r"(?:(?:to|toward|towards|past|into|through|by|near|next to|beside|in front of|at|along|around|across|between|"
    r"onto|out of|inside|from|of|until|with|on|in)\s+)*"
    r"(?:(?:the|a|an|your)\s+)?",
    re.IGNORECASE,
)


def landmark(phrase: str) -> str:
    """'stop next to the TV' -> 'TV'; phrases without an object ('turn left') stay as they are."""
    rest = _LEAD.sub("", phrase.strip()).strip()
    return rest if rest and re.search(r"\w", rest) else phrase.strip()


def split_instruction(text: str, max_chunks: int) -> List[str]:
    """Instruction -> up to max_chunks landmark phrases (one per sub-instruction), in order."""
    parts = [p.strip() for p in _SPLIT.split(text or "")]
    parts = [landmark(p) for p in parts if re.search(r"\w", p)]
    if not parts:
        parts = [text.strip() or "go"]
    return parts[:max_chunks]


class SiglipGrounder(nn.Module):
    """Frozen SigLIP2: phrase embeddings and text-aligned patch embeddings."""

    def __init__(self, name: str = "google/siglip2-base-patch16-224"):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.model = AutoModel.from_pretrained(name).eval().requires_grad_(False)
        self.tok = AutoTokenizer.from_pretrained(name)
        self._text_cache: Dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def phrases(self, chunks: List[List[str]], m: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """chunks per sample -> embeddings [B, m, D] (unit length) and mask [B, m]."""
        dev = next(self.model.parameters()).device
        new = sorted({c for cs in chunks for c in cs[:m] if c not in self._text_cache})
        if new:
            t = self.tok(new, padding="max_length", max_length=64, truncation=True, return_tensors="pt").to(dev)
            emb = F.normalize(self.model.get_text_features(**t).float(), dim=-1)
            if len(self._text_cache) > 50000:  # instructions repeat within an episode; keep the cache bounded
                self._text_cache.clear()
            self._text_cache.update(zip(new, emb))
        d = next(iter(self._text_cache.values())).shape[-1]
        out = torch.zeros(len(chunks), m, d, device=dev)
        mask = torch.zeros(len(chunks), m, dtype=torch.bool, device=dev)
        for b, cs in enumerate(chunks):
            for i, c in enumerate(cs[:m]):
                out[b, i], mask[b, i] = self._text_cache[c].to(dev), True
        return out, mask

    @torch.no_grad()
    def patches(self, pixels: torch.Tensor) -> torch.Tensor:
        """[N, 3, S, S] (SigLIP2 normalisation) -> [N, P, D] text-aligned patch embeddings (unit length)."""
        vm = self.model.vision_model
        h = vm(pixel_values=pixels.to(next(self.model.parameters()).dtype)).last_hidden_state
        head = vm.head
        d = h.shape[-1]
        attn = head.attention
        z = F.linear(F.linear(h, attn.in_proj_weight[2 * d :], attn.in_proj_bias[2 * d :]), attn.out_proj.weight,
                     attn.out_proj.bias)
        z = z + head.mlp(head.layernorm(z))
        return F.normalize(z.float(), dim=-1)


def grounding_maps(grounder, chunks: List[List[str]], m: int, frames: torch.Tensor) -> torch.Tensor:
    """Per-patch phrase match for each frame: [B, F, 3, S, S] -> [B, F, P, m].

    Each phrase's map is standardised over the patches of its frame (where in this frame, not how much),
    and phrases that do not exist (fewer than m) are zero.
    """
    B, n_frames = frames.shape[:2]
    text, mask = grounder.phrases(chunks, m)  # [B, m, D], [B, m]
    patch = grounder.patches(frames.flatten(0, 1)).unflatten(0, (B, n_frames))  # [B, F, P, D]
    s = torch.einsum("bfpd,bmd->bfpm", patch, text)
    s = (s - s.mean(2, keepdim=True)) / (s.std(2, keepdim=True) + 1e-6)
    return s * mask[:, None, None, :].float()
