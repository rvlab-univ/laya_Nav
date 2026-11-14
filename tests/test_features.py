"""Stored frame features (lightnav.features): the student must answer the same from them as from the images."""
import json

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from conftest import tiny_model
from lightnav import features as FS
from lightnav import tasks as T
from lightnav.data import NavDataset, collate, load_frames, load_rows, make_image_transform
from lightnav.model import LayaNative
from lightnav.questions import ANSWERS
from test_tasks import _train_module

VOCAB = ["bathroom", "sofa", "living room", "chairs", "kitchen", "bed"]
EPISODES = {"1": ("Turn left past the sofa and stop at the bed.", [6, 3, 2, 0]),
            "2": ("Turn right into the kitchen and stop at the sink.", [9, 1, 3, 3, 0]),
            "3": ("Walk into the bathroom and wait by the chairs.", [1, 2, 4, 0])}


def student(tok):
    """Run D's layout on tiny encoders: 2 x 2 tokens for the current frame, 1 per past frame, tags."""
    base = tiny_model(tok)
    cfg = base.cfg
    cfg.cur_grid, cfg.hist_grid, cfg.tags_current, cfg.tags_history, cfg.tag_vocab = 2, 1, 3, 1, list(VOCAB)
    model = LayaNative(cfg, base.dm, base.vision)
    torch.manual_seed(1)
    model.tag_text.copy_(F.normalize(torch.randn(len(VOCAB), model.vision.config.hidden_size), dim=-1))
    model.set_tag_tokens(tok)
    return model.eval()


def write(tmp_path, model):
    """Episodes with a different random image per frame, as images and as a feature store."""
    rng = np.random.default_rng(0)
    root, rows = tmp_path / "frames", []
    for ep, (q, labels) in EPISODES.items():
        sizes = [1]
        for a in labels[:-1]:
            sizes.append(sizes[-1] + (a - 1) % 3 + 1)
        (root / ep).mkdir(parents=True)
        for i in range(sizes[-1]):
            Image.fromarray(rng.integers(0, 255, (56, 56, 3), dtype=np.uint8)).save(root / ep / ("frame_%d.png" % i))
            (root / ep / ("frame_%d.png" % i)).rename(root / ep / ("frame_%d.jpg" % i))     # lossless, .jpg name
        for s, (n, a) in enumerate(zip(sizes, labels)):
            rows.append({"video_id": "%s-%d" % (ep, s), "q": q, "a": ANSWERS[a],
                         "frames": ["%s/frame_%d.jpg" % (ep, i) for i in range(n)]})
    (tmp_path / "a.json").write_text(json.dumps(rows))
    rows = load_rows(str(tmp_path / "a.json"), "r2r")
    cfg, store = model.cfg, str(tmp_path / "features")
    transform = make_image_transform(cfg.image_size, cfg.image_mean, cfg.image_std)
    FS.write_meta(store, cfg.vision_encoder, cfg.image_size, 2, model.vision.config.hidden_size)
    for ep in EPISODES:
        n = max(r.n_frames for r in rows if r.episode == ep)
        images = [Image.open(root / ep / ("frame_%d.jpg" % i)).convert("RGB") for i in range(n)]
        FS.save_episode(store, "r2r", ep + "/", FS.encode_images(model.vision, images, transform, 2, torch.device("cpu")))
    return rows, {"r2r": str(root)}, {"r2r": FS.FeatureStore(store, "r2r")}


def test_tokens_and_tags_match_the_image_path_as_rendered_and_mirrored(tok, tmp_path):
    model = student(tok)
    rows, images, stored = write(tmp_path, model)
    cfg = model.cfg
    transform = make_image_transform(cfg.image_size, cfg.image_mean, cfg.image_std)
    r = max(rows, key=lambda x: x.n_frames)
    for flip in (False, True):
        px, slots = load_frames(images["r2r"], r.prefix, r.n_frames, transform, cfg.num_frames, flip)
        feats, slots2 = load_frames(stored["r2r"], r.prefix, r.n_frames, transform, cfg.num_frames, flip)
        assert slots == slots2 and feats["grid"].shape == (len(slots), 4, 32) and feats["pooled"].shape == (len(slots), 32)
        s, owner = torch.tensor(slots), torch.zeros(len(slots), dtype=torch.long)
        with torch.no_grad():
            full = model.encode_frames(px)
            a, b = model.frame_tokens(full, s), model.frame_tokens(feats, s)
            assert all(torch.allclose(x, y, atol=2e-3) for x, y in zip(a, b))          # float16 on disk
            assert torch.allclose(model.tag_scores(full), model.tag_scores_pooled(feats["pooled"]), atol=2e-2)
            assert model.row_tags(full, s, owner, 1) == model.row_tags(feats, s, owner, 1)
    plain, mirrored = (stored["r2r"].frames(r.prefix, [0], f)["grid"] for f in (False, True))
    assert not torch.allclose(plain, mirrored, atol=1e-3)


def test_the_student_answers_the_same_from_stored_features(tok, tmp_path):
    model = student(tok)
    rows, images, stored = write(tmp_path, model)
    a = NavDataset(rows, images, tok, model.cfg)
    b = NavDataset(rows, stored, tok, model.cfg)
    ba, bb = (collate([d[i] for i in range(len(rows))], tok.pad_token_id) for d in (a, b))
    assert "pixels" in ba and "feats" not in ba and "feats" in bb and "pixels" not in bb
    with torch.no_grad():
        la, lb = model(ba), model(bb)
    assert torch.allclose(la, lb, atol=5e-3) and (la.argmax(-1) == lb.argmax(-1)).all()
    match = T.MatchDataset(T.routes_from_rows(rows), stored, tok, model.cfg, fixed=True)
    assert "feats" in match[0]
    with torch.no_grad():
        assert torch.isfinite(model(collate([match[0], b[1]], tok.pad_token_id))).all()


def test_contrast_pass_and_gradients_with_stored_features(tok, tmp_path):
    model = student(tok)
    rows, _, stored = write(tmp_path, model)
    same = NavDataset(rows, stored, tok, model.cfg, negative=lambda r, rng: r.instruction, negative_prob=1.0)
    batch = collate([same[0], same[5], same[3]], tok.pad_token_id)
    train = _train_module()
    with torch.no_grad():
        logits = model(batch)
        loss, gap = train.contrast_loss(model, batch, batch["feats"], logits, margin=0.5)
    assert abs(gap) < 1e-4 and abs(float(loss) - 0.5) < 1e-4
    moved = train.to_device(batch, torch.device("cpu"))
    assert set(moved["feats"]) == {"grid", "pooled"}
    model.train()
    model(batch).sum().backward()
    assert model.vis_proj[0].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.vision.parameters())          # the encoder is not part of the pass


def test_store_bookkeeping(tok, tmp_path):
    model = student(tok)
    rows, _, stored = write(tmp_path, model)
    store = stored["r2r"]
    assert store.has("1/") and store.has("1/", 3) and not store.has("1/", 500) and not store.has("9/")
    with pytest.raises(ValueError):          # the same folder cannot hold another encoder's or another grid's features
        FS.write_meta(store.root, model.cfg.vision_encoder, model.cfg.image_size, 7, 32)
    import pickle
    again = pickle.loads(pickle.dumps(store))                # what a DataLoader worker receives
    assert torch.equal(again.frames("2/", [0, 1])["grid"], store.frames("2/", [0, 1])["grid"])
    assert not (tmp_path / "features" / "r2r" / "1.tmp.npy").exists()
