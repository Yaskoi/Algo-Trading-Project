from __future__ import annotations
import numpy as np
from collections import deque

from src.strategies.base import BaseStrategy


class BollingerMRStrategy(BaseStrategy):
    def __init__(self, window: int = 20, k: float = 2.0, allow_short: bool = True):
        self.window = window
        self.k = k
        self.allow_short = allow_short
        self.buf = deque(maxlen=window)
        self.pos = 0.0

    def on_bar(self, ts, open_, high, low, close) -> float:
        self.buf.append(float(close))
        if len(self.buf) < self.window:
            return 0.0

        x = np.array(self.buf, dtype=float)
        m = x.mean()
        s = x.std(ddof=1) if len(x) > 1 else 0.0
        if s == 0:
            return 0.0

        upper = m + self.k * s
        lower = m - self.k * s

        c = float(close)

        # Mean reversion: sell upper, buy lower, exit around mean
        if c < lower:
            self.pos = 1.0
        elif c > upper and self.allow_short:
            self.pos = -1.0
        elif (self.pos > 0 and c >= m) or (self.pos < 0 and c <= m):
            self.pos = 0.0

        return self.pos
