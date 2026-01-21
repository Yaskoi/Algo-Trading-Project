from __future__ import annotations
import numpy as np
from collections import deque

from src.strategies.base import BaseStrategy


class VolTargetStrategy(BaseStrategy):
    """
    Position size ~ target_vol / realized_vol
    Direction: sign of momentum (close - mean)
    """
    def __init__(self, window: int = 120, target_vol: float = 0.0015, max_leverage: float = 2.0, allow_short: bool = True):
        self.window = window
        self.target_vol = target_vol
        self.max_leverage = max_leverage
        self.allow_short = allow_short

        self.closes = deque(maxlen=window)
        self.prev_close = None

    def on_bar(self, ts, open_, high, low, close) -> float:
        c = float(close)
        if self.prev_close is None:
            self.prev_close = c
            self.closes.append(c)
            return 0.0

        self.closes.append(c)
        if len(self.closes) < self.window:
            self.prev_close = c
            return 0.0

        x = np.array(self.closes, dtype=float)
        rets = np.diff(x)  # close-to-close
        vol = np.std(rets, ddof=1)
        if vol <= 1e-12:
            self.prev_close = c
            return 0.0

        # direction = momentum vs mean
        direction = np.sign(c - x.mean())
        if direction < 0 and not self.allow_short:
            direction = 0.0

        lev = min(self.max_leverage, self.target_vol / vol)
        pos = float(direction * lev)

        self.prev_close = c
        return pos
