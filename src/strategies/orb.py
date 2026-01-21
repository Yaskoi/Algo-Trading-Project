# src/strategies/orb.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from typing import Optional

import pandas as pd

from src.strategies.base import BaseStrategy


@dataclass
class ORBStrategy(BaseStrategy):
    """
    Opening Range Breakout (ORB) intraday.

    - Build opening range on first `orb_minutes` bars of the session.
    - After range is set:
        Long  if close > range_high * (1 + breakout_k)
        Short if close < range_low  * (1 - breakout_k)  (if allow_short)
    - Exit (flat) at end of day handled by engine (it closes remaining position).
    """

    orb_minutes: int = 30
    breakout_k: float = 0.0
    allow_short: bool = True

    def __post_init__(self) -> None:
        self._current_date: Optional[pd.Timestamp] = None
        self._range_high: Optional[float] = None
        self._range_low: Optional[float] = None
        self._bars_count: int = 0
        self._range_ready: bool = False
        self._position: float = 0.0

    def _reset_day(self, ts: pd.Timestamp) -> None:
        self._current_date = ts.normalize()
        self._range_high = None
        self._range_low = None
        self._bars_count = 0
        self._range_ready = False
        self._position = 0.0

    def on_bar(self, ts, open_, high, low, close) -> float:
        ts = pd.Timestamp(ts)

        # New day detection
        if self._current_date is None or ts.normalize() != self._current_date:
            self._reset_day(ts)

        # Count bars in the day
        self._bars_count += 1

        # Build opening range (first orb_minutes bars)
        if not self._range_ready:
            if self._range_high is None:
                self._range_high = float(high)
                self._range_low = float(low)
            else:
                self._range_high = max(self._range_high, float(high))
                self._range_low = min(self._range_low, float(low))

            if self._bars_count >= self.orb_minutes:
                self._range_ready = True

            # During range-building, stay flat
            self._position = 0.0
            return self._position

        # After range is ready -> breakout logic
        assert self._range_high is not None and self._range_low is not None

        up_level = self._range_high * (1.0 + float(self.breakout_k))
        dn_level = self._range_low * (1.0 - float(self.breakout_k))

        if close > up_level:
            self._position = 1.0
        elif self.allow_short and close < dn_level:
            self._position = -1.0
        else:
            # Hold previous position (trend-following style after breakout)
            # If you prefer mean-revert to flat when re-enters range, replace by: self._position = 0.0
            self._position = self._position

        return float(self._position)
