"""Offline decision metrics over the 10 answers."""
from typing import Dict

import numpy as np

from .questions import KEYS, NUM_ACTIONS, STOP, kind

_KIND = np.array([kind(i) for i in range(NUM_ACTIONS)])


def ece(conf: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(total)


def confusion(labels: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """[true, predicted] counts."""
    out = np.zeros((NUM_ACTIONS, NUM_ACTIONS), dtype=np.int64)
    np.add.at(out, (labels, pred), 1)
    return out


def summarize(labels: np.ndarray, probs: np.ndarray) -> Dict:
    """Accuracy over 10 answers and over the 4 kinds, stop precision / recall, per-answer recall, ECE."""
    labels, probs = np.asarray(labels), np.asarray(probs, dtype=np.float64)
    pred = probs.argmax(-1)
    correct = pred == labels
    said_stop, is_stop = pred == STOP, labels == STOP
    hit = (said_stop & is_stop).sum()
    return {
        "n": int(len(labels)),
        "acc": float(correct.mean()),
        "kind_acc": float((_KIND[pred] == _KIND[labels]).mean()),
        "stop_precision": float(hit / said_stop.sum()) if said_stop.any() else None,
        "stop_recall": float(hit / is_stop.sum()) if is_stop.any() else None,
        "ece": ece(probs.max(-1), correct),
        "recall": {KEYS[i]: float(correct[labels == i].mean()) for i in range(NUM_ACTIONS) if (labels == i).any()},
        # what the best constant answer scores on these samples: a model at this level has learned only the prior
        # (N1, N2 and D sat within 0.03 of constant_kind_acc for the whole of their training)
        "constant_acc": float(np.bincount(labels, minlength=NUM_ACTIONS).max() / len(labels)),
        "constant_kind_acc": float(np.bincount(_KIND[labels]).max() / len(labels)),
    }
