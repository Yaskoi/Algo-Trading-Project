from __future__ import annotations
import numpy as np
from collections import deque

from src.strategies.base import BaseStrategy


class MACrossStrategy(BaseStrategy):
    def __init__(self, fast: int = 20, slow: int = 60, allow_short: bool = True):
        assert fast < slow
        self.fast = fast
        self.slow = slow
        self.allow_short = allow_short
        self.buf = deque(maxlen=slow)

    def on_bar(self, ts, open_, high, low, close) -> float:
        self.buf.append(float(close))
        if len(self.buf) < self.slow:
            return 0.0

        x = np.array(self.buf, dtype=float)
        ma_fast = x[-self.fast:].mean()
        ma_slow = x.mean()

        if ma_fast > ma_slow:
            return 1.0
        if ma_fast < ma_slow and self.allow_short:
            return -1.0
        return 0.0
