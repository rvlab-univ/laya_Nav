"""Grounding prior and instruction matching (v3) - CPU tests with a fake grounder instead of SigLIP2."""

import importlib.util
import os
from dataclasses import asdict

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

pytest.importorskip("transformers")
from test_laya_nav import KEYS, tiny_nav  # noqa: E402
from test_laya_s2_smoke import _Tok, fake_batch, tiny_model  # noqa: E402

from internnav.model.basemodel.laya_s2 import (  # noqa: E402
    LayaNav,
    LayaS2,
    LossWeights,
    TrajLossWeights,
    instruction_chunks,
    mismatched,
)
from internnav.model.basemodel.laya_s2.grounding import split_instruction  # noqa: E402

INS = ["walk past the sofa, then turn left", "go to the kitchen", "walk past the sofa, then turn left",
       "stop at the door"]


class FakeGrounder(nn.Module):
    """Deterministic stand-in for SiglipGrounder: phrase vectors from a hash, patch vectors from the pixels."""

    def __init__(self, d=8, patch=8):
        super().__init__()
        self.patch = patch
        self.proj = nn.Linear(3, d)

    def phrases(self, chunks, m):
        d = self.proj.out_features
        out, mask = torch.zeros(len(chunks), m, d), torch.zeros(len(chunks), m, dtype=torch.bool)
        for b, cs in enumerate(chunks):
            for i, c in enumerate(cs[:m]):
                g = torch.Generator().manual_seed(sum(map(ord, c)) * 7919 % 2**31)
                out[b, i], mask[b, i] = F.normalize(torch.randn(d, generator=g), dim=0), True
        return out, mask

    def patches(self, pixels):
        x = F.avg_pool2d(pixels, self.patch).flatten(2).transpose(1, 2)  # [N, P, 3]
        return F.normalize(self.proj(x), dim=-1)


def grounded(model):
    model.set_grounder(FakeGrounder())
    return model


def load_trainer(name):
    path = os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "train", "laya_s2", f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_split_instruction():
    # one landmark phrase per sub-instruction: SigLIP2 matches object names, not whole sentences
    assert split_instruction("Walk past the sofa, then turn left and stop at the door.", 6) == ["sofa", "left", "door"]
    assert split_instruction("Stop in front of the painting on the wall.", 6) == ["painting on the wall"]
    assert split_instruction("Walk to the sofa, then stop next to the TV.", 6) == ["sofa", "TV"]
    assert split_instruction("a. b. c. d.", 2) == ["a", "b"]
    assert split_instruction("", 6) == ["go"]
    assert mismatched(["a", "a", "b"]) == [2, 2, 0] and mismatched(["a", "a"]) == [-1, -1]


def test_grounding_added_to_checkpoint_starts_neutral_and_trains(tmp_path):
    s2 = tiny_model().eval()
    s2.save_pretrained(str(tmp_path / "s2"))
    g = grounded(LayaS2.from_pretrained(str(tmp_path / "s2"), grounding=True, match_head=True))
    assert g.cfg.grounding and g.cfg.match_head
    b = fake_batch()
    chunks = instruction_chunks(g, INS)
    g.eval()
    with torch.no_grad():  # zero output layer: exactly the checkpoint's behaviour at first
        x = {k: b[k] for k in KEYS}
        torch.testing.assert_close(g(**x, chunks=chunks)["logits"], s2(**x)["logits"])
    g.train()
    out = g(**{k: b[k] for k in KEYS}, goal_xy=b["goal_xy"], chunks=chunks)
    (out["logits"].logsumexp(-1).sum() + out["match_logit"].sum()).backward()
    assert g.ground_proj[-1].weight.grad.abs().sum() > 0 and g.match_head[-1].weight.grad is not None
    with pytest.raises(AssertionError):  # the phrases are required once grounding is on
        g(**{k: b[k] for k in KEYS})


def test_grounding_makes_the_instruction_matter():
    model = grounded(LayaS2(tiny_model().cfg.__class__(**{**vars(tiny_model().cfg), "grounding": True}),
                            tiny_model().text, tiny_model().vision)).eval()
    b = fake_batch()
    x = {k: b[k] for k in KEYS}
    a, c = [["go to the sofa"]] * 4, [["go to the kitchen"]] * 4
    with torch.no_grad():
        torch.testing.assert_close(model(**x, chunks=a)["logits"], model(**x, chunks=c)["logits"])  # not learned yet
        nn.init.normal_(model.ground_proj[-1].weight, std=0.5)
        assert not torch.allclose(model(**x, chunks=a)["logits"], model(**x, chunks=c)["logits"])


def test_matching_negatives_in_one_forward():
    tr = load_trainer("train_laya_nav")
    nav = tiny_nav(traj_fuse_layers=1, traj_goal_mark=True)
    cfg = nav.cfg.__class__(**{**vars(nav.cfg), "grounding": True, "match_head": True})
    model = grounded(LayaNav(cfg, nav.text, nav.vision))
    b = fake_batch()
    batch = dict(b, instructions=INS, traj_pixels=torch.randn(4, 2, 3, 32, 32),
                 traj_mask=torch.tensor([[1, 1], [0, 0], [1, 0], [0, 0]], dtype=torch.bool),
                 traj=torch.randn(4, 2, 32, 3))
    w = LossWeights(**{**asdict(LossWeights()), "match": 0.5})
    traj = dict(traj_pixels=batch["traj_pixels"], traj_mask=batch["traj_mask"])
    out, l_match, stats = tr.base.decision_forward(model, batch, w, **traj)
    assert out["logits"].shape[0] == 4 and out["traj"].shape[0] == 3 and out["traj_idx"].max() < 4  # real batch only
    assert torch.isfinite(l_match) and 0 <= float(stats["acc_match"]) <= 1
    loss, st = tr.run(model, batch, w, TrajLossWeights(), "c2")
    assert torch.isfinite(loss) and "l_match" in st and "ade" in st
    loss.backward()
    assert model.match_head[-1].weight.grad is not None
    loss_c1, st_c1 = tr.run(model, batch, w, TrajLossWeights(), "c1")  # no matching in c1
    assert "l_match" not in st_c1


def test_agent_with_grounding():
    from internnav.model.basemodel.laya_s2.agent import LayaNavAgent

    nav = tiny_nav()
    model = grounded(LayaNav(nav.cfg.__class__(**{**vars(nav.cfg), "grounding": True}), nav.text, nav.vision)).eval()
    agent = LayaNavAgent(model, _Tok(), torch.device("cpu"))
    img = Image.new("RGB", (128, 96), (120, 110, 100))
    d = agent.decide("walk to the sofa, then stop", [img] * 2, img, img)
    agent.start_goal(d, img, None)
    assert agent.plan(img, None).shape == (1, 32, 3)


def test_phrase_cache_refills_after_clearing():
    """A cached phrase of the batch survives the cache being cleared (KeyError: 'stairs' on the server)."""
    from transformers import BatchEncoding

    from internnav.model.basemodel.laya_s2.grounding import SiglipGrounder

    class TextModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Linear(1, 4)

        def get_text_features(self, input_ids):
            return self.w(input_ids.float())

    g = SiglipGrounder.__new__(SiglipGrounder)
    nn.Module.__init__(g)
    g.model, g._text_cache = TextModel(), {}
    g.tok = lambda texts, **kw: BatchEncoding({"input_ids": torch.tensor([[len(t)] for t in texts])})
    g.phrases([["stairs"]], 2)
    g._text_cache.update({f"x{i}": torch.zeros(4) for i in range(50001)})
    emb, mask = g.phrases([["stairs", "door"]], 2)
    assert mask.all() and torch.allclose(emb.norm(dim=-1), torch.ones(1, 2))
