"""The navigation question in Laya's format, and the mapping to NaVILA's 10 answer sentences."""
from typing import List, Optional, Sequence, Tuple

import numpy as np

# NaVILA's answer sentences, verbatim (NaVILA-Dataset annotations, field "a"). The index is the option order.
ANSWERS: List[str] = [
    "I think I should stop because I have finished the instruction.",
    "The next action is move forward 25 cm.",
    "The next action is move forward 50 cm.",
    "The next action is move forward 75 cm.",
    "The next action is turn left 15 degree.",
    "The next action is turn left 30 degree.",
    "The next action is turn left 45 degree.",
    "The next action is turn right 15 degree.",
    "The next action is turn right 30 degree.",
    "The next action is turn right 45 degree.",
]
_ANSWER_INDEX = {a: i for i, a in enumerate(ANSWERS)}
STOP = 0
NUM_ACTIONS = len(ANSWERS)

# The question as `laya.Agent` takes it. Keys stay short so that all 10 options fit in the head budget.
QUESTION = {
    "type": "choice",
    "instructions": "What should the robot do next to follow the instruction?",
    "criteria": {
        "stop": "the instruction is finished, stop here",
        "forward 25": "move forward 25 cm",
        "forward 50": "move forward 50 cm",
        "forward 75": "move forward 75 cm",
        "left 15": "turn left 15 degree",
        "left 30": "turn left 30 degree",
        "left 45": "turn left 45 degree",
        "right 15": "turn right 15 degree",
        "right 30": "turn right 30 degree",
        "right 45": "turn right 45 degree",
    },
}
KEYS: List[str] = list(QUESTION["criteria"])

# Option index after a left-right mirror of the scene: left <-> right, stop and forward unchanged.
MIRROR: List[int] = [0, 1, 2, 3, 7, 8, 9, 4, 5, 6]

KINDS = ("stop", "forward", "left", "right")


def answer_index(sentence: str) -> int:
    """Option index of a NaVILA answer sentence. Raises KeyError on anything else."""
    return _ANSWER_INDEX[sentence]


def low_level(index: int) -> Tuple[int, int]:
    """(Habitat action, repeat count): 0 stop, 1 forward 0.25 m, 2 left 15 deg, 3 right 15 deg."""
    if index == STOP:
        return 0, 1
    return (index - 1) // 3 + 1, (index - 1) % 3 + 1


def kind(index: int) -> int:
    """0 stop, 1 forward, 2 left, 3 right."""
    return low_level(index)[0]


def from_low_level(action: int, steps: int) -> int:
    """Inverse of `low_level` for a run of `steps` (1..3) identical low-level actions."""
    if action == 0:
        return STOP
    if not 1 <= steps <= 3:
        raise ValueError("steps must be 1..3, got %r" % (steps,))
    return (action - 1) * 3 + steps


def decide(probs: Sequence[float], tau_stop: Optional[float] = None) -> int:
    """The chosen option. With `tau_stop`, stop only when P(stop) exceeds it, else the best non-stop option."""
    p = np.asarray(probs, dtype=np.float64)
    if tau_stop is None:
        return int(p.argmax())
    if p[STOP] > tau_stop:
        return STOP
    return int(p[1:].argmax()) + 1


# Auxiliary task (design B 4.1 (A)): does the instruction describe the route shown by the 8 frames?
MATCH_QUESTION = {
    "type": "choice",
    "instructions": "Does the instruction describe this route?",
    "criteria": {
        "yes": "the instruction describes this route",
        "no": "the instruction describes a different route",
    },
}
MATCH_KEYS: List[str] = list(MATCH_QUESTION["criteria"])
YES, NO = 0, 1
