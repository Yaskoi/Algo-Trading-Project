from __future__ import annotations
import numpy as np
from collections import deque

from src.strategies.base import BaseStrategy


def ema_update(prev: float, x: float, alpha: float) -> float:
    return alpha * x + (1 - alpha) * prev


class MACDHistStrategy(BaseStrategy):
    def __init__(self, n_short=12, n_long=26, n_signal=9, allow_short=True, min_hold: int = 5):
        self.n_short = n_short
        self.n_long = n_long
        self.n_signal = n_signal
        self.allow_short = allow_short
        self.min_hold = min_hold

        self.init = False
        self.ema_s = 0.0
        self.ema_l = 0.0
        self.sig = 0.0
        self.prev_hist = 0.0
        self.hold = 0
        self.pos = 0.0

        self.alpha_s = 2 / (n_short + 1)
        self.alpha_l = 2 / (n_long + 1)
        self.alpha_sig = 2 / (n_signal + 1)

    def on_bar(self, ts, open_, high, low, close) -> float:
        x = float(close)

        if not self.init:
            self.ema_s = x
            self.ema_l = x
            self.sig = 0.0
            self.prev_hist = 0.0
            self.init = True
            return 0.0

        self.ema_s = ema_update(self.ema_s, x, self.alpha_s)
        self.ema_l = ema_update(self.ema_l, x, self.alpha_l)
        macd = self.ema_s - self.ema_l
        self.sig = ema_update(self.sig, macd, self.alpha_sig)
        hist = macd - self.sig

        # cross 0
        desired = self.pos
        if self.hold > 0:
            self.hold -= 1
            self.prev_hist = hist
            return desired

        if self.prev_hist <= 0 and hist > 0:
            desired = 1.0
            self.hold = self.min_hold
        elif self.prev_hist >= 0 and hist < 0:
            desired = -1.0 if self.allow_short else 0.0
            self.hold = self.min_hold

        self.prev_hist = hist
        self.pos = desired
        return desired
