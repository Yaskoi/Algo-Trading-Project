from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass(frozen=True)
class BacktestConfig:
    data_root: Path
    results_root: Path

    tickers: List[str]
    weights: Optional[List[float]] = None

    is_ratio: float = 5 / 6

    # fees
    bp_fee: float = 0.0001

    # columns
    price_col: str = "Close"
    open_col: str = "Open"
    high_col: str = "High"
    low_col: str = "Low"

    # risk (ATR SL/TP)
    atr_period: int = 14
    sl_atr: float = 1.5
    tp_atr: float = 2.0

    # notional
    unit_size: float = 1.0

    # execution
    exec_at: str = "next_open"  # "next_open" recommended
    max_days: Optional[int] = None

    seed: int = 42
