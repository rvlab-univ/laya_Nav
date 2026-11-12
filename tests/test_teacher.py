"""Teacher scoring (lightnav/teacher.py) on a tiny random Llama, and NaVILA's prompt and answer reading."""
import os
import re

import pytest
import torch

from conftest import NAVILA
from lightnav.questions import ANSWERS, STOP
from lightnav.teacher import answer_from_text, navila_question, repeat_cache, score_answers


def tiny_llama(seed=0):
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=97, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=256)
    return LlamaForCausalLM(cfg).eval()


@torch.no_grad()
def brute_force(llm, prompt_embeds, answer):
    """log P(answer | prompt) from one pass over prompt + answer, without any cache."""
    emb = torch.cat([prompt_embeds, llm.get_input_embeddings()(torch.tensor([answer]))], 1)
    logp = torch.log_softmax(llm(inputs_embeds=emb).logits[0].float(), -1)
    p = prompt_embeds.shape[1]
    return sum(float(logp[p - 1 + t, tok]) for t, tok in enumerate(answer))


def test_score_answers_matches_full_passes():
    llm = tiny_llama()
    g = torch.Generator().manual_seed(1)
    prompt = llm.get_input_embeddings()(torch.randint(0, 97, (1, 23), generator=g))
    answers = [[5, 9, 12, 3], [5, 9, 40], [7], [5, 9, 12, 3, 88, 1, 2]]
    got = score_answers(llm, prompt, answers)
    want = torch.tensor([brute_force(llm, prompt, a) for a in answers])
    assert got.shape == (4,) and got.dtype == torch.float32
    assert torch.allclose(got, want, atol=1e-4), (got, want)


def test_repeat_cache_tuple_and_object():
    llm = tiny_llama()
    out = llm(input_ids=torch.tensor([[1, 2, 3]]), use_cache=True)
    cache = out.past_key_values
    legacy = cache if isinstance(cache, tuple) else cache.to_legacy_cache()
    rep = repeat_cache(legacy, 5)
    assert isinstance(rep, tuple) and rep[0][0].shape[0] == 5 and torch.equal(rep[1][1][4], legacy[1][1][0])
    if not isinstance(cache, tuple):
        obj = repeat_cache(cache, 3)
        assert type(obj) is type(cache) and obj.to_legacy_cache()[0][0].shape[0] == 3


def navila_fstring(path, variable):
    """NaVILA's `question = (...)` f-string from a source file, evaluated with the given image-token variable."""
    path = os.path.join(NAVILA, path)
    if not os.path.exists(path):
        pytest.skip("no NaVILA source at %s" % path)
    with open(path) as f:
        expr = re.search(r"question = \((.*?)\n\s*\)\n", f.read(), re.S).group(1)
    return eval("(" + expr + ")", {variable: "<image>\n" * 7, "instruction": "Walk to the sofa. Stop."})


@pytest.mark.parametrize("path,variable", [("llava/data/dataset.py", "image_tokens"),
                                           ("evaluation/vlnce_baselines/navila_trainer.py", "interleaved_images")])
def test_navila_question_matches_source(path, variable):
    assert navila_question("Walk to the sofa. Stop.", 8) == navila_fstring(path, variable)


def test_answer_from_text():
    for i, a in enumerate(ANSWERS):
        assert answer_from_text(a) == i, a
    assert answer_from_text("The next action is move forward 60 cm.") == 2        # snapped to 50 cm
    assert answer_from_text("The next action is turn left 20 degree.") == 4       # snapped to 15 degrees
    assert answer_from_text("The next action is turn right.") == 7                 # no number: one step
    assert answer_from_text("The next action is move forward 100 cm.") is None    # four steps: not an answer
    assert answer_from_text("I think I should stop here.") == STOP
    assert answer_from_text("Empty video.") is None


# ---------------------------------------------------------------- scripts/extract_teacher_probs.py end to end
import importlib.util  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import types  # noqa: E402

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from conftest import ROOT  # noqa: E402
from lightnav.data import TeacherStore, frame_slots  # noqa: E402

IMAGE = -200


class FakeNavila(torch.nn.Module):
    """Stands in for NaVILA's LlavaLlamaModel: a tiny Llama, each image one vector, generate() a fixed answer."""

    def __init__(self, tok, answer):
        super().__init__()
        self.llm = tiny_llama_vocab(len(tok))
        self.config = types.SimpleNamespace(num_video_frames=8)
        self.device, self.dtype = torch.device("cpu"), torch.float32
        self.answer_ids = tok(answer, add_special_tokens=False).input_ids

    def cuda(self):
        return self

    def prepare_inputs_labels_for_multimodal(self, input_ids, position_ids, mask, past, labels, images):
        emb, parts, img = self.llm.get_input_embeddings(), [], 0
        for t in input_ids[0].tolist():
            if t == IMAGE:
                parts.append(images[img].mean() * torch.ones(1, emb.embedding_dim))
                img += 1
            else:
                parts.append(emb(torch.tensor([t])))
        assert img == 8
        return None, None, None, None, torch.cat(parts)[None], None

    def generate(self, input_ids, images=None, **kw):
        return torch.tensor([self.answer_ids])


def tiny_llama_vocab(vocab):
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    return LlamaForCausalLM(LlamaConfig(vocab_size=vocab, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                        num_attention_heads=4, num_key_value_heads=2,
                                        max_position_embeddings=512)).eval()


def fake_llava(monkeypatch, tok):
    """Modules with the names and calls the script imports from NaVILA."""
    from enum import Enum

    class SeparatorStyle(Enum):
        TWO = 1
        LLAMA_3 = 2

    class Conv:
        roles, sep, sep2, sep_style = ("<user>", "<assistant>"), "<|eot|>", "<|end|>", SeparatorStyle.LLAMA_3

        def __init__(self):
            self.messages = []

        def copy(self):
            return Conv()

        def append_message(self, role, text):
            self.messages.append((role, text))

        def get_prompt(self):
            return "".join(r + (t or "") for r, t in self.messages)

    def tokenizer_image_token(prompt, tokenizer, index, return_tensors=None):
        ids = []
        for i, chunk in enumerate(prompt.split("<image>")):
            ids += ([index] if i else []) + tokenizer(chunk, add_special_tokens=False).input_ids
        return torch.tensor(ids)

    def vlnce_frame_sampling(paths, num_frames):
        return [Image.new("RGB", (4, 4)) if s < 0 else Image.open(paths[s]).convert("RGB")
                for s in frame_slots(len(paths), num_frames)]

    def process_images(images, processor, cfg):
        return torch.stack([torch.tensor(np.asarray(im.resize((4, 4))), dtype=torch.float32).permute(2, 0, 1) / 255
                            for im in images])

    mods = {
        "llava": types.ModuleType("llava"),
        "llava.constants": types.SimpleNamespace(IMAGE_TOKEN_INDEX=IMAGE),
        "llava.conversation": types.SimpleNamespace(SeparatorStyle=SeparatorStyle, conv_templates={"llama_3": Conv()}),
        "llava.mm_utils": types.SimpleNamespace(KeywordsStoppingCriteria=lambda *a: None, process_images=process_images,
                                                tokenizer_image_token=tokenizer_image_token,
                                                vlnce_frame_sampling=vlnce_frame_sampling),
        "llava.model": types.ModuleType("llava.model"),
        "llava.model.builder": types.SimpleNamespace(
            load_pretrained_model=lambda path, name: (tok, FakeNavila(tok, ANSWERS[3]), None, 2048)),
    }
    for name, mod in mods.items():
        monkeypatch.setitem(sys.modules, name, mod)


def load_script():
    spec = importlib.util.spec_from_file_location("extract_teacher_probs",
                                                  os.path.join(ROOT, "scripts", "extract_teacher_probs.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_extract_teacher_probs_end_to_end(tok, tmp_path, monkeypatch):
    fake_llava(monkeypatch, tok)
    script = load_script()
    frames, ann = tmp_path / "frames", tmp_path / "ann.json"
    rows = []
    for ep, labels in (("7", [6, 3, 0]), ("8", [1, 0])):           # left 45 -> forward 75 -> stop; forward 25 -> stop
        n = 1
        for step, label in enumerate(labels):
            rows.append({"video_id": "%s-%d" % (ep, step), "q": "Walk to the sofa and stop.", "a": ANSWERS[label],
                         "frames": ["%s/frame_%d.jpg" % (ep, i) for i in range(n)]})
            n += 0 if label == 0 else (label - 1) % 3 + 1
        (frames / ep).mkdir(parents=True)
        for i in range(n):
            Image.new("RGB", (8, 8), (i * 20, 50, 90)).save(frames / ep / ("frame_%d.jpg" % i))
    rows.append(dict(rows[0]))                                       # a duplicate row, as NaVILA's files have
    ann.write_text(json.dumps(rows))
    out = tmp_path / "teacher"
    argv = ["--ckpt", str(tmp_path / "ckpt"), "--out", str(out), "--sources", "r2r", "--r2r_ann", str(ann),
            "--r2r_frames", str(frames), "--chunk", "2", "--check_generate", "2"]
    monkeypatch.setattr(sys, "argv", ["x"] + argv)
    script.run(script.parse_args())
    parts = sorted(out.glob("shard_0/part_*.npz"))
    assert len(parts) == 3                                           # 5 unique samples, 2 per part
    script.run(script.parse_args())                                  # resumes: nothing left to do
    assert len(sorted(out.glob("shard_0/part_*.npz"))) == 3

    monkeypatch.setattr(sys, "argv", ["x", "--out", str(out), "--merge", "--sources", "r2r", "--r2r_ann", str(ann)])
    script.merge(script.parse_args())
    keys = (out / "keys.txt").read_text().split()
    assert keys == sorted(["r2r:7-0", "r2r:7-1", "r2r:7-2", "r2r:8-0", "r2r:8-1"])
    probs = np.load(out / "teacher_probs.npy")
    assert probs.shape == (5, 10) and np.allclose(probs.sum(1), 1, atol=1e-5)
    stats = json.loads((out / "stats.json").read_text())
    assert stats["n"] == 5 and stats["missing"] == 0 and stats["generate_checked"] == 2
    assert TeacherStore(str(out)).get("r2r:8-1").shape == (10,)

    # a lost frame: the sample is skipped and reported, the others still go through
    (frames / "8" / "frame_1.jpg").unlink()
    out2 = tmp_path / "teacher2"
    monkeypatch.setattr(sys, "argv", ["x"] + [a if a != str(out) else str(out2) for a in argv])
    script.run(script.parse_args())
    got = {k for p in out2.glob("shard_0/part_*.npz") for k in np.load(p)["keys"].tolist()}
    assert got == {"r2r:7-0", "r2r:7-1", "r2r:7-2", "r2r:8-0"}
