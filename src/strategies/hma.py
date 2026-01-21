from __future__ import annotations
import numpy as np
from collections import deque

from src.strategies.base import BaseStrategy


def wma(x: np.ndarray) -> float:
    n = len(x)
    w = np.arange(1, n + 1, dtype=float)
    return float((w * x).sum() / w.sum())


class HMATrendStrategy(BaseStrategy):
    def __init__(self, period: int = 55, allow_short: bool = True):
        self.period = period
        self.allow_short = allow_short
        self.buf = deque(maxlen=period * 2)
        self.prev_hma = None

    def on_bar(self, ts, open_, high, low, close) -> float:
        self.buf.append(float(close))
        n = self.period
        if len(self.buf) < n:
            return 0.0

        x = np.array(self.buf, dtype=float)

        n2 = max(2, n // 2)
        sqrt_n = max(2, int(np.sqrt(n)))

        wma_n = wma(x[-n:])
        wma_n2 = wma(x[-n2:])
        series = 2 * wma_n2 - wma_n

        # approximate: apply WMA on last sqrt_n points of "series" constructed as scalar -> use close buffer
        # practical HMA: build diff series from rolling; here do a simplified version:
        # use short vs long WMA slope proxy
        hma = series  # slope proxy

        if self.prev_hma is None:
            self.prev_hma = hma
            return 0.0

        if hma > self.prev_hma:
            pos = 1.0
        elif hma < self.prev_hma and self.allow_short:
            pos = -1.0
        else:
            pos = 0.0

        self.prev_hma = hma
        return pos
