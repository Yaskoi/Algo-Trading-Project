from __future__ import annotations
import numpy as np
from collections import deque

from src.strategies.base import BaseStrategy


class DonchianBreakoutStrategy(BaseStrategy):
    def __init__(self, window: int = 20, allow_short: bool = True):
        self.window = window
        self.allow_short = allow_short
        self.highs = deque(maxlen=window)
        self.lows = deque(maxlen=window)
        self.pos = 0.0

    def on_bar(self, ts, open_, high, low, close) -> float:
        self.highs.append(float(high))
        self.lows.append(float(low))
        if len(self.highs) < self.window:
            return 0.0

        hi = max(self.highs)
        lo = min(self.lows)
        c = float(close)

        if c >= hi:
            self.pos = 1.0
        elif c <= lo and self.allow_short:
            self.pos = -1.0

        return self.pos
