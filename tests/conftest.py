"""Shared fixtures: a real tokenizer, tiny random encoders (CPU), and the NaVILA annotations when present."""
import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("HF_HOME", os.path.join(ROOT, "assets", "hf_cache"))

DATA = os.path.join(ROOT, "data", "NaVILA-Dataset")
NAVILA = os.path.join(ROOT, "NaVILA")
TOKENIZER = "answerdotai/ModernBERT-large"


@pytest.fixture(scope="session")
def tok():
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(TOKENIZER)
    except Exception as e:  # no network and nothing cached
        pytest.skip("tokenizer %s not available: %s" % (TOKENIZER, e))


def tiny_model(tok, seed: int = 0):
    """A LayaNative with random 64-d encoders: 14 x 14 patches per frame, like the real one."""
    from laya.common import DecisionModel
    from transformers import ModernBertConfig, ModernBertModel, SiglipVisionConfig, SiglipVisionModel

    from lightnav.model import LayaNative, LayaNativeConfig

    torch.manual_seed(seed)
    enc = ModernBertModel(ModernBertConfig(
        vocab_size=len(tok), hidden_size=64, num_hidden_layers=3, num_attention_heads=2, intermediate_size=128,
        local_attention=16, reference_compile=False))
    vision = SiglipVisionModel(SiglipVisionConfig(
        hidden_size=32, num_hidden_layers=1, num_attention_heads=2, intermediate_size=64, image_size=56, patch_size=4))
    cfg = LayaNativeConfig(laya=None, encoder="tiny", image_size=56)
    return LayaNative(cfg, DecisionModel(enc, head_layers=2, n_act=2), vision)


@pytest.fixture()
def model(tok):
    return tiny_model(tok).eval()


@pytest.fixture(scope="session")
def r2r_annotations():
    path = os.path.join(DATA, "R2R", "annotations.json")
    if not os.path.exists(path):
        pytest.skip("no R2R annotations at %s" % path)
    return path


def fake_batch(tok, model, instructions, n_frames, seed: int = 0):
    """A collated batch with random images; `n_frames[i]` is how many frames sample i has seen so far."""
    from laya.train import to_internal

    from lightnav.data import collate, encode_text, frame_slots
    from lightnav.questions import QUESTION

    g = torch.Generator().manual_seed(seed)
    q, cfg, items = to_internal("action", QUESTION), model.cfg, []
    for ins, n in zip(instructions, n_frames):
        ids, markers = encode_text(tok, q, ins, cfg.text_max_len, cfg.head_max_len)
        slots = [s for s, f in enumerate(frame_slots(n, cfg.num_frames)) if f >= 0]
        items.append({"ids": ids, "markers": markers, "slots": slots,
                      "pixels": torch.randn(len(slots), 3, cfg.image_size, cfg.image_size, generator=g)})
    return collate(items, tok.pad_token_id)
