from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class TradeState:
    position: float = 0.0          # -1, 0, +1 (or leverage)
    entry_price: Optional[float] = None
    entry_atr: Optional[float] = None
    stop: Optional[float] = None
    take: Optional[float] = None


def compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    # True Range
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))

    atr = np.full_like(close, np.nan, dtype=float)
    if len(tr) < period:
        return atr

    # simple moving average ATR (robuste & rapide)
    cumsum = np.cumsum(tr, dtype=float)
    atr[period - 1] = cumsum[period - 1] / period
    for i in range(period, len(tr)):
        atr[i] = (cumsum[i] - cumsum[i - period]) / period
    return atr


def set_sl_tp(state: TradeState, sl_atr: float, tp_atr: float):
    if state.entry_price is None or state.entry_atr is None or state.position == 0:
        state.stop, state.take = None, None
        return

    if state.position > 0:  # long
        state.stop = state.entry_price - sl_atr * state.entry_atr
        state.take = state.entry_price + tp_atr * state.entry_atr
    else:  # short
        state.stop = state.entry_price + sl_atr * state.entry_atr
        state.take = state.entry_price - tp_atr * state.entry_atr


def check_sl_tp_hit(state: TradeState, bar_high: float, bar_low: float):
    """
    Return exit_price if SL/TP touched during bar, else None.
    Conservative:
    - long: SL if low <= stop, TP if high >= take
    - short: SL if high >= stop, TP if low <= take
    """
    if state.position == 0 or state.stop is None or state.take is None:
        return None

    if state.position > 0:
        # SL first (conservative)
        if bar_low <= state.stop:
            return state.stop
        if bar_high >= state.take:
            return state.take
    else:
        if bar_high >= state.stop:
            return state.stop
        if bar_low <= state.take:
            return state.take

    return None
