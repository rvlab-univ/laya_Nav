"""Auxiliary task of design B 4.1 (A): does the instruction describe the route shown by the 8 frames? yes / no.

Why it is needed (design B 7.2.1): every R2R path is rendered once per instruction with the same frames and the
same actions, so the navigation labels never reward reading the instruction, and N1 / N2 learned to ignore it.
This task asks about the instruction directly.

- A route is one episode: the frames of its last decision (the whole path), picked by NaVILA's 8-frame rule.
- Positive: the episode's own instruction.
- Negatives:
  - "swap": one landmark replaced by another of the same kind (room / object, singular / plural), so only the
    frames can tell. Every occurrence of that word is replaced, so the text stays consistent with itself.
  - "other": the instruction of another episode of the same dataset whose path differs. Episodes that share a
    path (R2R renders each path once per instruction) are recognised by their frame count and action sequence,
    which needs no VLN-CE file on the server.
- Structural words (door, room, hallway, stairs, wall, floor, ...) are never swapped or put in.
"""
import random
import re
from collections import defaultdict
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
from torch.utils.data import Dataset

from . import data as D
from .questions import MATCH_QUESTION, NO, NUM_ACTIONS, YES

# Landmarks, from the head words of R2R's landmark phrases (laya_base grounding.split_instruction), 2026-11-13.
# Rooms and objects are kept apart so that a swap does not make the sentence odd ("walk into the piano").
ROOMS = ("bedroom", "bathroom", "kitchen", "closet", "office", "patio", "foyer", "balcony", "porch", "garage",
         "library", "pantry", "sauna", "restroom", "cellar", "gym", "theater", "kitchenette", "lobby",
         "nursery", "basement", "attic")
ROOM_MODIFIERS = ("living", "dining", "family", "laundry", "game", "sitting", "billiard", "exercise", "media")
OBJECTS = ("table", "bed", "sink", "couch", "chair", "fireplace", "desk", "counter", "rug", "pool", "mirror", "piano",
           "refrigerator", "bar", "shower", "tub", "painting", "sofa", "bathtub", "bench", "sign", "tv", "carpet",
           "window", "plant", "fridge", "stove", "railing", "toilet", "dryer", "dresser", "vase", "television",
           "cabinet", "pillar", "bookshelf", "oven", "lamp", "washer", "elevator", "fountain", "tree", "column",
           "shelf", "mat", "statue", "guitar", "ottoman", "treadmill", "sculpture", "picture", "bookcase",
           "banister", "thermostat", "nightstand", "chandelier", "vanity", "recliner", "cooler", "rack", "clock",
           "hammock", "bike", "armchair", "microwave", "basket", "doormat", "heater", "stool", "dishwasher",
           "poster", "radiator", "wardrobe")
OBJECTS_PLURAL = ("chairs", "couches", "tables", "plants", "sinks", "beds", "windows", "cabinets", "shelves",
                  "columns", "pillars", "stools", "bookshelves", "sofas", "benches", "paintings", "pictures", "lockers",
                  "armchairs", "desks", "flowers", "lamps", "mirrors", "toilets", "dressers", "vases", "rugs",
                  "statues", "ovens")
# Never swapped and never put in: they appear on almost every route.
STRUCTURAL = ("door", "doors", "doorway", "doorways", "room", "rooms", "hallway", "hall", "corridor", "stairs",
              "stair", "staircase", "stairway", "stairwell", "steps", "step", "wall", "walls", "floor", "area",
              "entrance", "entryway", "entry", "exit", "landing", "corner", "archway", "arch", "opening", "way",
              "path", "pathway", "walkway", "house", "building", "end", "side")

GROUPS: Dict[str, Tuple[str, ...]] = {"room": ROOMS, "room_mod": ROOM_MODIFIERS, "object": OBJECTS,
                                      "objects": OBJECTS_PLURAL}
_AFTER = {"room_mod": r"(?=\s+room\b)", "object": r"(?!\s*-?\s*clockwise)"}   # "living room"; not "counter clockwise"


def _pattern(group: str, words: Sequence[str]):
    return re.compile(r"\b(%s)\b%s" % ("|".join(sorted(words, key=len, reverse=True)), _AFTER.get(group, "")),
                      re.IGNORECASE)


_PATTERNS = {g: _pattern(g, w) for g, w in GROUPS.items()}
assert not set(STRUCTURAL) & {w for ws in GROUPS.values() for w in ws}


def phrase(group: str, word: str) -> str:
    """The landmark as a tag phrase: "living" before "room" is "living room"."""
    return word + " room" if group == "room_mod" else word


def tag_phrases() -> List[str]:
    """Every landmark phrase, the vocabulary of the student's tags (lightnav.model)."""
    return [phrase(g, w) for g, words in GROUPS.items() for w in words]


def landmarks(text: str) -> List[Tuple[str, str]]:
    """`(group, word)` of every swappable landmark in `text`, lower case, without repeats, in order."""
    out = []
    for g, pat in _PATTERNS.items():
        for m in pat.finditer(text):
            k = (g, m.group(1).lower())
            if k not in out:
                out.append(k)
    return out


def _cased(new: str, old: str) -> str:
    if old.isupper() and len(old) > 1:
        return new.upper()
    return new.capitalize() if old[0].isupper() else new


def swap_landmark(text: str, rng: random.Random) -> Optional[Tuple[str, str, str]]:
    """`(new text, old word, new word)` with one landmark replaced everywhere by another of its group that the text
    does not mention, or None when the text has no swappable landmark."""
    found = landmarks(text)
    if not found:
        return None
    group, old = found[rng.randrange(len(found))]
    present = set(re.findall(r"[a-z]+", text.lower()))
    choices = [w for w in GROUPS[group] if w != old and w not in present]
    if not choices:
        return None
    new = rng.choice(choices)
    pat = re.compile(r"\b%s\b%s" % (re.escape(old), _AFTER.get(group, "")), re.IGNORECASE)
    return pat.sub(lambda m: _cased(new, m.group(0)), text), old, new


# ------------------------------------------------------------------------------------- routes

class Route(NamedTuple):
    source: str
    episode: str
    instruction: str
    n_frames: int       # frames of the whole path (its last decision)
    prefix: str
    path: Tuple         # (source, n_frames, actions): equal for episodes that share a path


def routes_from_rows(rows: Sequence[D.Row]) -> List[Route]:
    """One route per episode, sorted by (source, episode)."""
    eps = defaultdict(dict)
    for r in rows:
        eps[(r.source, r.episode)][r.step] = r
    out = []
    for (source, episode), steps in sorted(eps.items()):
        last = steps[max(steps)]
        actions = tuple(steps[k].label for k in sorted(steps))
        out.append(Route(source, episode, last.instruction, last.n_frames, last.prefix,
                         (source, last.n_frames, actions)))
    return out


# ------------------------------------------------------------------------------------ dataset

class MatchDataset(Dataset):
    """Matching samples over `routes`.

    Training (`fixed=False`): index i shows route i % len(routes); positive or negative and the negative are drawn
    afresh each time. Validation (`fixed=True`): two samples per route, 2k positive and 2k + 1 negative, the same
    every time (`seed`). Negatives of kind "other" come from routes of the same source in `routes`.

    `synthetic`: share of samples whose tag words (a model with tags, cfg.tag_vocab) are not what the frames show
    but a separable stand-in: up to 3 landmarks of the shown instruction among random words for a positive, only
    words the shown instruction does not mention for a negative ("tag_override"). Without this warm-up the student
    never finds the comparison in the noisy real tags (design B 7.2.3).
    """

    def __init__(self, routes: Sequence[Route], image_roots: Dict[str, str], tok, cfg, fixed: bool = False,
                 neg_frac: float = 0.5, swap_frac: float = 0.6, seed: int = 0, synthetic: float = 0.0,
                 synthetic_words: int = 8):
        from laya.train import to_internal

        self.routes, self.image_roots, self.tok, self.cfg = list(routes), image_roots, tok, cfg
        self.fixed, self.neg_frac, self.swap_frac, self.seed = fixed, neg_frac, swap_frac, seed
        self.synthetic, self.synthetic_words = synthetic, synthetic_words
        vocab = list(getattr(cfg, "tag_vocab", None) or [])
        if synthetic > 0 and not vocab:
            raise ValueError("synthetic tags need a model with tags (cfg.tag_vocab)")
        self.vocab_index = {p: i for i, p in enumerate(vocab)}
        self.question = to_internal("match", MATCH_QUESTION)
        self.transform = D.make_image_transform(cfg.image_size, cfg.image_mean, cfg.image_std, augment=not fixed)
        self.by_source = defaultdict(list)
        for i, r in enumerate(self.routes):
            self.by_source[r.source].append(i)
        if not self.routes:
            raise ValueError("no routes")

    def __len__(self):
        return 2 * len(self.routes) if self.fixed else len(self.routes)

    def other_instruction(self, route: Route, rng: random.Random) -> str:
        pool = self.by_source[route.source]
        for _ in range(100):
            o = self.routes[pool[rng.randrange(len(pool))]]
            if o.path != route.path and o.instruction != route.instruction:
                return o.instruction
        raise ValueError("no route with another path in %s" % route.source)

    def sample(self, i: int) -> Tuple[Route, str, str]:
        """`(route, instruction shown, kind)`; kind is "pos", "swap" or "other"."""
        if self.fixed:
            route, negative = self.routes[i // 2], i % 2 == 1
            rng = random.Random(self.seed * 1000003 + i)
        else:
            route = self.routes[i % len(self.routes)]
            rng = random.Random(random.getrandbits(64))
            negative = rng.random() < self.neg_frac
        if not negative:
            return route, route.instruction, "pos"
        if rng.random() < self.swap_frac:
            swapped = swap_landmark(route.instruction, rng)
            if swapped is not None:
                return route, swapped[0], "swap"
        return route, self.other_instruction(route, rng), "other"

    def encode_text(self, instruction: str):
        from laya.common import build_sequence

        ids, markers = build_sequence(self.tok, {"instruction": D.clean_instruction(instruction)}, self.question,
                                      max_len=self.cfg.text_max_len, head_max_len=self.cfg.head_max_len)
        if len(markers) != len(MATCH_QUESTION["criteria"]):
            raise ValueError("only %d option markers fit" % len(markers))
        return ids, markers

    def synthetic_tags(self, instruction: str, positive: bool, rng: random.Random) -> Optional[List[int]]:
        """Vocabulary indices of the separable stand-in tags, or None when the instruction names no landmark."""
        own = [self.vocab_index[p] for p in dict.fromkeys(phrase(g, w) for g, w in landmarks(instruction))
               if p in self.vocab_index]
        if not own:
            return None
        rest = [v for v in range(len(self.vocab_index)) if v not in set(own)]
        words = (own[:3] if positive else []) + rng.sample(rest, self.synthetic_words - (min(3, len(own))
                                                                                          if positive else 0))
        rng.shuffle(words)
        return words

    def __getitem__(self, i):
        route, instruction, kind = self.sample(i)
        pixels, slots = D.load_frames(self.image_roots[route.source], route.prefix, route.n_frames, self.transform,
                                      self.cfg.num_frames)
        ids, markers = self.encode_text(instruction)
        label = YES if kind == "pos" else NO
        target = np.zeros(NUM_ACTIONS, dtype=np.float32)
        target[label] = 1.0
        out = {"ids": ids, "markers": markers, D.frames_key(pixels): pixels, "slots": slots, "target": target,
               "label": label, "key": "match:%s:%s:%s" % (route.source, route.episode, kind), "mirrored": False,
               "task": "match", "kind": kind}
        if self.synthetic > 0 and not self.fixed:
            rng = random.Random(random.getrandbits(64))
            if rng.random() < self.synthetic:
                words = self.synthetic_tags(instruction, kind == "pos", rng)
                if words is not None:
                    out["tag_override"] = words
                    out["key"] += ":synthetic"
        return out


class OtherInstruction:
    """`(row, rng) -> instruction` of a random episode of the same source whose path differs from the row's: the
    negative of the instruction contrast loss (scripts/train.py --contrast). Picklable, for DataLoader workers.

    `alias`: sources without routes of their own that are other walks of another source's episodes (DAgger rollouts
    of R2R, `{"dagger1": "r2r"}`); their rows are treated as that source's."""

    def __init__(self, routes: Sequence[Route], alias: Optional[Dict[str, str]] = None):
        self.path = {(r.source, r.episode): r.path for r in routes}
        self.pool = defaultdict(list)
        for r in routes:
            self.pool[r.source].append((r.path, r.instruction))
        self.alias = dict(alias or {})

    def __call__(self, row: D.Row, rng: random.Random) -> Optional[str]:
        source = self.alias.get(row.source, row.source)
        pool, own = self.pool.get(source), self.path.get((source, row.episode))
        if not pool:
            return None
        for _ in range(100):
            path, text = pool[rng.randrange(len(pool))]
            if path != own and text != row.instruction:
                return text
        return None


class Mixed(Dataset):
    """Navigation samples followed by `aux` samples, `frac` of the whole; a shuffling DataLoader mixes them."""

    def __init__(self, nav: Dataset, aux: Optional[Dataset], frac: float):
        if aux is None or frac <= 0:
            frac = 0.0
        if not 0 <= frac < 1:
            raise ValueError("frac must be in [0, 1), got %r" % (frac,))
        self.nav, self.aux = nav, aux
        self.n_aux = int(round(len(nav) * frac / (1 - frac))) if frac > 0 else 0

    def __len__(self):
        return len(self.nav) + self.n_aux

    def __getitem__(self, i):
        if i < len(self.nav):
            return self.nav[i]
        return self.aux[(i - len(self.nav)) % len(self.aux)]


def match_metrics(kinds: Sequence[str], probs: np.ndarray) -> Dict:
    """Accuracy of the matching samples, overall and by kind (pos / swap / other)."""
    kinds = np.asarray(kinds)
    pred = np.asarray(probs)[:, :2].argmax(-1)
    correct = pred == np.where(kinds == "pos", YES, NO)
    out = {"n": int(len(kinds)), "acc": float(correct.mean()) if len(kinds) else None}
    for k in ("pos", "swap", "other"):
        m = kinds == k
        out["acc_" + k] = float(correct[m].mean()) if m.any() else None
        out["n_" + k] = int(m.sum())
    return out
