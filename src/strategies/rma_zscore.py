from __future__ import annotations
import numpy as np
from collections import deque

from src.strategies.base import BaseStrategy


class RMAZScoreStrategy(BaseStrategy):
    """
    z-score of close vs rolling mean/std:
    enter long if z < -z_entry, short if z > z_entry
    exit when |z| < z_exit
    """
    def __init__(self, window: int = 120, z_entry: float = 1.2, z_exit: float = 0.3, allow_short: bool = True):
        self.window = window
        self.z_entry = z_entry
        self.z_exit = z_exit
        self.allow_short = allow_short
        self.buf = deque(maxlen=window)
        self.pos = 0.0

    def on_bar(self, ts, open_, high, low, close) -> float:
        self.buf.append(float(close))
        if len(self.buf) < self.window:
            return 0.0

        x = np.array(self.buf, dtype=float)
        m = x.mean()
        s = x.std(ddof=1)
        if s == 0:
            return 0.0

        z = (float(close) - m) / s

        if z <= -self.z_entry:
            self.pos = 1.0
        elif z >= self.z_entry and self.allow_short:
            self.pos = -1.0
        elif abs(z) <= self.z_exit:
            self.pos = 0.0

        return self.pos
