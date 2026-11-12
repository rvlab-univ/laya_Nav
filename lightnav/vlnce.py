"""VLN-CE evaluation pieces shared by the evaluation loop (Python 3.8, Habitat 0.1.7) and the student process.

Nothing here imports Habitat or torch. NaVILA's own rules are copied from
NaVILA/evaluation/vlnce_baselines/navila_trainer.py (answer parsing and the action queue) and
llava/mm_utils.py (frame selection, via lightnav.data.frame_slots).
"""
import io
import json
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .questions import ANSWERS, STOP, from_low_level

STOP_ACTION, FORWARD, LEFT, RIGHT = 0, 1, 2, 3

_PATTERNS = [(STOP_ACTION, re.compile(r"\bstop\b", re.IGNORECASE)),
             (FORWARD, re.compile(r"\bis move forward\b", re.IGNORECASE)),
             (LEFT, re.compile(r"\bis turn left\b", re.IGNORECASE)),
             (RIGHT, re.compile(r"\bis turn right\b", re.IGNORECASE))]


def navila_actions(text: str) -> Optional[List[int]]:
    """The low-level actions NaVILA's evaluation loop runs for one answer text, or None if it reads no action.

    Same rules as navila_trainer.py: "stop" is looked for first, then the action; the number is cm or degrees, a value
    that is not a multiple of the step is snapped to the nearest of 25/50/75 cm or 15/30/45 degrees, a missing number
    means one step, and a multiple of the step runs that many steps (100 cm runs 4). NaVILA itself would crash on an
    unreadable answer; the caller decides what to do with None.
    """
    action = next((a for a, pat in _PATTERNS if pat.search(text)), None)
    if action is None:
        return None
    if action == STOP_ACTION:
        return [STOP_ACTION]
    if action == FORWARD:
        m = re.search(r"move forward (\d+) cm", text)
        value, step, choices = (int(m.group(1)) if m else 25), 25, (25, 50, 75)
    else:
        m = re.search(r"turn %s (\d+) degree" % ("left" if action == LEFT else "right"), text)
        value, step, choices = (int(m.group(1)) if m else 15), 15, (15, 30, 45)
    if value % step:
        value = min(choices, key=lambda c: abs(c - value))
    return [action] * max(1, value // step)


def gt_answers(actions: Sequence[int]) -> List[str]:
    """Ground-truth low-level actions (VLN-CE *_gt.json.gz) as NaVILA answer sentences: runs of the same action in
    chunks of at most 3, as NaVILA's training data merges them. The STOP at the end becomes the stop sentence."""
    out, i = [], 0
    while i < len(actions):
        a = int(actions[i])
        if a == STOP_ACTION:
            out.append(ANSWERS[STOP])
            break
        n = 1
        while i + n < len(actions) and int(actions[i + n]) == a and n < 3:
            n += 1
        out.append(ANSWERS[from_low_level(a, n)])
        i += n
    else:
        out.append(ANSWERS[STOP])
    return out


# ------------------------------------------------------------------------------------------- request format
# A request is an uncompressed npz: frames uint8 [k, H, W, 3] (the real frames among NaVILA's 8, black padding left
# out), slots int64 [k] (their position among the 8, the current frame is 7), and the instruction and episode id as
# strings. The response is JSON: {"answer": str, "probs": [10 floats], "ms": float}.

def encode_request(instruction: str, frames: Sequence[np.ndarray], slots: Sequence[int], episode_id: str = "") -> bytes:
    buf = io.BytesIO()
    np.savez(buf, frames=np.stack([np.asarray(f, dtype=np.uint8)[..., :3] for f in frames]),
             slots=np.asarray(slots, dtype=np.int64), instruction=np.array(instruction), episode=np.array(episode_id))
    return buf.getvalue()


def decode_request(data: bytes) -> Dict:
    with np.load(io.BytesIO(data), allow_pickle=False) as z:
        return {"frames": list(z["frames"]), "slots": z["slots"].tolist(), "instruction": str(z["instruction"]),
                "episode": str(z["episode"])}


def encode_response(answer: str, probs: Sequence[float], ms: float) -> bytes:
    return json.dumps({"answer": answer, "probs": [float(p) for p in probs], "ms": float(ms)}).encode()


def decode_response(data: bytes) -> Tuple[str, List[float], float]:
    d = json.loads(data.decode())
    return d["answer"], d.get("probs", []), d.get("ms", 0.0)
