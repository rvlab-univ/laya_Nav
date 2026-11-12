"""NaVILA-Dataset annotations as training samples for the Laya-native student.

One annotation row is `{"video_id": "<episode>-<decision>", "q": instruction, "a": answer sentence,
"frames": ["<episode>/frame_0.jpg", ..., current frame]}`. There are no poses and no low-level actions.
"""
import gzip
import json
import os
import random
import re
from typing import Dict, List, NamedTuple, Optional, Sequence, Set

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

from .questions import MIRROR, NUM_ACTIONS, QUESTION, answer_index

NUM_FRAMES = 8


# ------------------------------------------------------------------------------ instructions

def clean_instruction(q: str) -> str:
    """NaVILA's instruction clean-up, same as `LazyVLNCEDataset.__getitem__` (llava/data/dataset.py)."""
    q = q.replace("\r\n", " ").replace("\n", " ")
    q = re.sub(r"(?<=\.\s)([a-z])", lambda m: m.group().upper(), q.capitalize())
    return re.sub(r"\s+\.", ".", q)


# Copied from LayaNav (laya_base/internnav/dataset/laya_s2_dataset.py), unchanged.
_SWAP_WORD = re.compile(r"\b(counter-?clockwise|anti-?clockwise|clockwise|left|right)(?=(?:most|hand|wards?)?\b)",
                        re.IGNORECASE)
# "right" / "left" that are no direction: such instructions are not mirrored (rather than swapped wrongly)
_NOT_DIRECTION = re.compile(
    r"\ball right\b|\bright\s+(?:in front|next|there|here|away|now|behind|beside|below|above|under|inside|outside)\b"
    r"|\b(?:you|you've|have|has|had|just|we|we've|i)\s+left\b|\bleft\s+(?:over|behind)\b",
    re.IGNORECASE,
)
_NOT_ENGLISH = re.compile(r"[^\x00-\x7f‘’“”]")  # e.g. the RxR Hindi / Telugu instructions


def _swapped(m) -> str:
    w = m.group(0)
    low = w.lower()
    t = {"left": "right", "right": "left", "clockwise": "counterclockwise"}.get(low, "clockwise")
    return t.upper() if w.isupper() and len(w) > 1 else (t.capitalize() if w[0].isupper() else t)


def is_english(text: str) -> bool:
    return _NOT_ENGLISH.search(text) is None


def mirror_instruction(text: str) -> Optional[str]:
    """The instruction for the left-right mirrored scene, or None when it cannot be mirrored safely
    (not English, or "right" / "left" used as no direction)."""
    if _NOT_ENGLISH.search(text) or _NOT_DIRECTION.search(text):
        return None
    return _SWAP_WORD.sub(_swapped, text)


# ------------------------------------------------------------------------------------ frames

def frame_slots(n_frames: int, num_frames: int = NUM_FRAMES) -> List[int]:
    """Which frame each of the `num_frames` slots shows, as NaVILA picks them; -1 is a black padding frame.

    Same rule as `vlnce_frame_sampling` (llava/mm_utils.py) and `sample_and_pad_images`
    (evaluation/vlnce_baselines/navila_trainer.py): pad at the front up to `num_frames`, take
    `num_frames - 1` evenly spaced frames from the start, then the latest frame.
    """
    if n_frames < 1:
        raise ValueError("n_frames must be at least 1, got %r" % (n_frames,))
    pad = max(0, num_frames - n_frames)
    length = n_frames + pad
    picked = np.linspace(0, length - 1, num=num_frames - 1, endpoint=False, dtype=int).tolist() + [length - 1]
    return [i - pad if i >= pad else -1 for i in picked]


# --------------------------------------------------------------------------------------- rows

class Row(NamedTuple):
    source: str         # "r2r", "rxr"
    video_id: str       # "<episode>-<decision>"
    episode: str
    step: int           # decision index inside the episode, 0-based; the last one is the stop
    instruction: str    # as written in the annotation
    label: int          # index into questions.ANSWERS
    n_frames: int       # frames frame_0 .. frame_{n-1}; the last one is the current view
    prefix: str         # "<episode>/"

    @property
    def key(self) -> str:
        return "%s:%s" % (self.source, self.video_id)

    def frame(self, i: int) -> str:
        return "%sframe_%d.jpg" % (self.prefix, i)


def load_rows(path: str, source: str, english_only: bool = True, dedupe: bool = False) -> List[Row]:
    """Rows of one `annotations.json`. The file already repeats some samples for balance (R2R: stop x3 and
    30/45 degree turns x2, RxR: stop x5); `dedupe=True` keeps one row per `video_id`."""
    with open(path) as f:
        data = json.load(f)
    rows, seen, texts = [], set(), {}
    for x in data:
        vid = x["video_id"]
        if dedupe:
            if vid in seen:
                continue
            seen.add(vid)
        q = x["q"]
        if english_only and not is_english(q):
            continue
        frames = x["frames"]
        n = len(frames)
        prefix = frames[-1].rsplit("frame_", 1)[0]
        if frames[-1] != "%sframe_%d.jpg" % (prefix, n - 1) or frames[0] != "%sframe_0.jpg" % prefix:
            raise ValueError("%s: frames are not frame_0..frame_%d (%s .. %s)" % (vid, n - 1, frames[0], frames[-1]))
        episode, step = vid.rsplit("-", 1)
        rows.append(Row(source, vid, episode, int(step), texts.setdefault(q, q), answer_index(x["a"]), n, prefix))
    return rows


def low_level_actions(rows: Sequence[Row]) -> Dict[str, List[int]]:
    """Per episode ("<source>:<episode>"), the low-level action taken at every frame (0 stop, 1 forward,
    2 left, 3 right), rebuilt from the answers. Raises if a decision's frame count does not match."""
    from .questions import low_level

    episodes: Dict[str, Dict[int, Row]] = {}
    for r in rows:
        episodes.setdefault("%s:%s" % (r.source, r.episode), {})[r.step] = r
    out = {}
    for ep, steps in episodes.items():
        actions = []
        for s in sorted(steps):
            r = steps[s]
            if r.n_frames != len(actions) + 1:
                raise ValueError("%s step %d: %d frames but %d actions before it" % (ep, s, r.n_frames, len(actions)))
            action, repeat = low_level(r.label)
            actions += [action] * repeat
        out[ep] = actions
    return out


# ---------------------------------------------------------------------------- validation split

def episode_scenes(path: str) -> Dict[str, str]:
    """`episode_id -> scene` from a VLN-CE episode file (`train.json.gz`)."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        data = json.load(f)
    return {str(e["episode_id"]): os.path.basename(e["scene_id"]).split(".")[0] for e in data["episodes"]}


def choose_val_episodes(episodes: Sequence[str], scenes: Optional[Dict[str, str]] = None,
                        frac: float = 0.05, seed: int = 0) -> Set[str]:
    """Validation episodes: whole scenes when `scenes` is given (about `frac` of the scenes), else about
    `frac` of the episodes. Depends only on the arguments, so every run holds out the same episodes."""
    episodes = sorted(set(episodes))
    rng = random.Random(seed)
    if scenes is None:
        return set(rng.sample(episodes, max(1, round(len(episodes) * frac))))
    missing = [e for e in episodes if e not in scenes]
    if missing:
        raise KeyError("%d episode(s) have no scene, e.g. %s" % (len(missing), missing[:3]))
    names = sorted({scenes[e] for e in episodes})
    held = set(rng.sample(names, max(1, round(len(names) * frac))))
    return {e for e in episodes if scenes[e] in held}


def save_split(path: str, val: Dict[str, Set[str]], mode: str) -> None:
    with open(path, "w") as f:
        json.dump({"mode": mode, "val_episodes": {k: sorted(v) for k, v in val.items()}}, f)


def load_split(path: str) -> Dict[str, Set[str]]:
    with open(path) as f:
        return {k: set(v) for k, v in json.load(f)["val_episodes"].items()}


def split_rows(rows: Sequence[Row], val: Dict[str, Set[str]]):
    """`(train, validation)`. Every copy of a sample goes to the same side, since the split is by episode."""
    train = [r for r in rows if r.episode not in val.get(r.source, ())]
    held = [r for r in rows if r.episode in val.get(r.source, ())]
    return train, held


# ------------------------------------------------------------------------------ teacher labels

class TeacherStore:
    """`keys.txt` + `teacher_probs.npy` ([N, 10] float), memory-mapped. Keys are `Row.key`."""

    def __init__(self, root: str):
        with open(os.path.join(root, "keys.txt")) as f:
            self.index = {k.rstrip("\n"): i for i, k in enumerate(f)}
        self.root = root
        self._arr = None

    def get(self, key: str) -> Optional[np.ndarray]:
        i = self.index.get(key)
        if i is None:
            return None
        if self._arr is None:  # opened lazily so each dataloader worker gets its own mmap
            self._arr = np.load(os.path.join(self.root, "teacher_probs.npy"), mmap_mode="r")
        return np.asarray(self._arr[i], dtype=np.float32)


# ------------------------------------------------------------------------------------ dataset

def make_image_transform(image_size: int, mean, std, augment: bool = False):
    from torchvision.transforms import v2

    ops = [v2.ToImage(), v2.Resize((image_size, image_size), antialias=True)]
    if augment:
        ops.append(v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2))
    ops += [v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean, std)]
    return v2.Compose(ops)


def target_distribution(label: int, teacher: Optional[np.ndarray] = None, alpha: float = 0.5) -> np.ndarray:
    """One-hot on `label`, or `(1 - alpha) * one-hot + alpha * teacher` when teacher probabilities are given."""
    t = np.zeros(NUM_ACTIONS, dtype=np.float32)
    t[label] = 1.0
    if teacher is not None:
        t = (1.0 - alpha) * t + alpha * teacher / max(float(teacher.sum()), 1e-9)
    return t


class NavDataset(Dataset):
    """Decision samples: up to 8 frames (black padding left out), the text half of the sequence, a target."""

    def __init__(self, rows: Sequence[Row], image_roots: Dict[str, str], tok, cfg, augment: bool = False,
                 mirror_prob: float = 0.0, teacher: Optional[TeacherStore] = None, alpha: float = 0.5):
        from laya.train import to_internal

        self.rows, self.image_roots, self.tok, self.cfg = list(rows), image_roots, tok, cfg
        self.mirror_prob, self.teacher, self.alpha = mirror_prob, teacher, alpha
        self.transform = make_image_transform(cfg.image_size, cfg.image_mean, cfg.image_std, augment)
        self.question = to_internal("action", QUESTION)

    def __len__(self):
        return len(self.rows)

    def encode_text(self, instruction: str):
        """`(ids, markers)` of `[CLS] question [SEP] [MASK] option ... [SEP] state [SEP]`, without any image."""
        return encode_text(self.tok, self.question, instruction, self.cfg.text_max_len, self.cfg.head_max_len)

    def __getitem__(self, i):
        r = self.rows[i]
        instruction, label, flip = r.instruction, r.label, False
        teacher = self.teacher.get(r.key) if self.teacher is not None else None
        if self.mirror_prob > 0 and random.random() < self.mirror_prob:
            mirrored = mirror_instruction(instruction)
            if mirrored is not None:
                instruction, label, flip = mirrored, MIRROR[label], True
                if teacher is not None:
                    teacher = teacher[MIRROR]
        images, slots = [], []
        root = self.image_roots[r.source]
        for slot, frame in enumerate(frame_slots(r.n_frames, self.cfg.num_frames)):
            if frame < 0:
                continue
            img = Image.open(os.path.join(root, r.frame(frame))).convert("RGB")
            if flip:
                img = ImageOps.mirror(img)
            images.append(self.transform(img))
            slots.append(slot)
        ids, markers = self.encode_text(instruction)
        return {"ids": ids, "markers": markers, "pixels": torch.stack(images), "slots": slots,
                "target": target_distribution(label, teacher, self.alpha), "label": label, "key": r.key,
                "mirrored": flip}


def encode_text(tok, question: Dict, instruction: str, max_len: int, head_max_len: int):
    """The text half of a sample in Laya's layout. The instruction goes into the state, cleaned as NaVILA does."""
    from laya.common import build_sequence

    ids, markers = build_sequence(tok, {"instruction": clean_instruction(instruction)}, question,
                                  max_len=max_len, head_max_len=head_max_len)
    if len(markers) != NUM_ACTIONS:
        raise ValueError("only %d of %d option markers fit in max_len=%d" % (len(markers), NUM_ACTIONS, max_len))
    return ids, markers


def dropped_instruction_tokens(tok, question: Dict, instruction: str, max_len: int, head_max_len: int) -> int:
    """How many tokens of the state (the instruction) `max_len` cuts off; 0 when it fits."""
    from laya.common import build_sequence

    _, _, stats = build_sequence(tok, {"instruction": clean_instruction(instruction)}, question, max_len=max_len,
                                 head_max_len=head_max_len, return_truncation_stats=True)
    return stats["state_tokens_dropped"]


def collate(batch: Sequence[Dict], pad_id: int) -> Dict:
    n, length = len(batch), max(len(b["ids"]) for b in batch)
    ids = torch.full((n, length), pad_id, dtype=torch.long)
    text_len = torch.zeros(n, dtype=torch.long)
    for i, b in enumerate(batch):
        ids[i, :len(b["ids"])] = torch.tensor(b["ids"], dtype=torch.long)
        text_len[i] = len(b["ids"])
    out = {
        "input_ids": ids,
        "text_len": text_len,
        "marker_pos": torch.tensor([b["markers"] for b in batch], dtype=torch.long),
        "pixels": torch.cat([b["pixels"] for b in batch]),
        "frame_owner": torch.tensor([i for i, b in enumerate(batch) for _ in b["slots"]], dtype=torch.long),
        "frame_slot": torch.tensor([s for b in batch for s in b["slots"]], dtype=torch.long),
    }
    if "target" in batch[0]:
        out["target"] = torch.from_numpy(np.stack([b["target"] for b in batch]))
        out["label"] = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        out["keys"] = [b["key"] for b in batch]
    return out
