import time
from contextlib import contextmanager

import torch


class EpisodeTimers:
    """Per-episode GPU-synchronized wall time of System 2 / System 1 calls, written to progress.json."""

    NAMES = ("s2", "s2_latent", "s1")

    def __init__(self):
        self.time = {n: 0.0 for n in self.NAMES}
        self.calls = {n: 0 for n in self.NAMES}

    @contextmanager
    def _timed(self, name):
        torch.cuda.synchronize()
        t = time.perf_counter()
        yield
        torch.cuda.synchronize()
        self.time[name] += time.perf_counter() - t
        self.calls[name] += 1

    def s2(self):
        return self._timed("s2")

    def s2_latent(self):
        return self._timed("s2_latent")

    def s1(self):
        return self._timed("s1")

    def summary(self):
        out = {}
        for n in self.NAMES:
            out[f"{n}_calls"] = self.calls[n]
            out[f"{n}_time"] = round(self.time[n], 4)
        return out
