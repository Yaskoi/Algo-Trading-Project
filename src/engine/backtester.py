from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import numpy as np
import pandas as pd

from src.config import BacktestConfig
from src.data.loader import extract_ticker_from_filename, load_pickle_df, ticker_matches
from src.engine.execution import FeeModel, RoundTripFeeTracker
from src.engine.risk import TradeState, check_sl_tp_hit, compute_atr, set_sl_tp
from src.strategies.base import BaseStrategy


def run_backtest_days(
    cfg: BacktestConfig,
    day_dirs: List[Path],
    strategy_cls: Type[BaseStrategy],
    strategy_params: Dict[str, Any],
    tag: str = "OOS",
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for day_dir in day_dirs:
        daily_rows = run_one_day(cfg, day_dir, strategy_cls, strategy_params)
        rows.extend(daily_rows)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["Tag"] = tag
    return df


def run_one_day(
    cfg: BacktestConfig,
    day_dir: Path,
    strategy_cls: Type[BaseStrategy],
    strategy_params: Dict[str, Any],
) -> List[Dict[str, Any]]:
    files = sorted([p for p in day_dir.iterdir() if p.is_file() and p.name.startswith("df_") and p.suffix == ".pkl"])
    if not files:
        return []

    out: List[Dict[str, Any]] = []
    fee_tracker = RoundTripFeeTracker(FeeModel(bp=cfg.bp_fee))

    for f in files:
        ticker_file = extract_ticker_from_filename(f.name)
        if ticker_file is None:
            continue

        # Filter: keep only tickers asked by cfg if possible
        if cfg.tickers:
            keep = any(ticker_matches(t, ticker_file) for t in cfg.tickers)
            if not keep:
                continue

        df = load_pickle_df(f)
        if df.empty:
            continue

        # required columns
        for c in [cfg.open_col, cfg.high_col, cfg.low_col, cfg.price_col]:
            if c not in df.columns:
                # sometimes columns are multi-indexed (Price/Ticker) => if so, user should flatten upstream
                # here we skip safely
                return []

        # compute ATR on day
        high = df[cfg.high_col].to_numpy(dtype=float)
        low = df[cfg.low_col].to_numpy(dtype=float)
        close = df[cfg.price_col].to_numpy(dtype=float)
        open_ = df[cfg.open_col].to_numpy(dtype=float)

        atr = compute_atr(high, low, close, cfg.atr_period)

        strat = strategy_cls(**strategy_params)
        state = TradeState(position=0.0)

        gross_pnl = 0.0
        net_pnl = 0.0
        fees = 0.0
        num_trades = 0

        # We execute at next open to avoid look-ahead
        # loop until n-2 so we can execute at i+1 open
        n = len(df)
        if n < 3:
            continue

        # track last mark price for pnl
        last_price = close[0]

        for i in range(n - 1):
            # mark-to-market PnL on close-to-close
            # holding PnL for position during bar i (from last close to current close)
            price_now = close[i]
            holding = state.position * cfg.unit_size * (price_now - last_price)
            gross_pnl += holding
            net_pnl += holding
            last_price = price_now

            # risk check uses bar i HIGH/LOW AFTER entry (this is ok bar-by-bar)
            exit_price = check_sl_tp_hit(state, bar_high=high[i], bar_low=low[i])
            if exit_price is not None:
                # close position at exit_price (assume touched)
                # adjust pnl from last close to exit_price if needed (intrabar)
                # we already booked close-to-close; approximate: override by adding delta (exit - close[i])
                adj = state.position * cfg.unit_size * (exit_price - close[i])
                gross_pnl += adj
                net_pnl += adj

                # fees round-trip
                fees += fee_tracker.charge_round_trip(abs(state.position), state.entry_price, exit_price)
                net_pnl -= fee_tracker.charge_round_trip(0.0, 0.0, 0.0)  # no-op (keeps structure)
                net_pnl -= 0.0

                net_pnl -= 0.0  # placeholder

                # apply fee properly once (avoid double)
                # (we charged in fee_tracker; now reflect in net)
                net_pnl -= fee_tracker.fee_model.bp * 0.0  # no-op

                # Actually subtract the fee we computed:
                last_fee = abs(state.position) * cfg.bp_fee * (state.entry_price + exit_price)
                net_pnl -= last_fee
                fees += 0.0  # already counted above in fees via last_fee? -> we set fees += ... below instead

                # fix: keep consistent:
                # overwrite: use last_fee only
                fees -= last_fee  # remove previous
                fees += last_fee

                # reset state
                state.position = 0.0
                state.entry_price = None
                state.entry_atr = None
                state.stop = None
                state.take = None
                num_trades += 1

                continue  # after forced exit, skip signal action at same bar

            # signal computed on bar close i
            desired_pos = strat.on_bar(
                ts=df.index[i],
                open_=open_[i],
                high=high[i],
                low=low[i],
                close=close[i],
            )

            desired_pos = float(desired_pos)

            # execute at next open (i+1)
            exec_price = open_[i + 1]

            # if change position
            if desired_pos != state.position:
                # if closing existing pos => charge fees for round-trip
                if state.position != 0.0:
                    last_fee = abs(state.position) * cfg.bp_fee * (state.entry_price + exec_price)
                    fees += last_fee
                    net_pnl -= last_fee
                    num_trades += 1  # closing trade

                # if opening new pos
                if desired_pos != 0.0:
                    state.position = desired_pos
                    state.entry_price = exec_price
                    # entry ATR: use atr[i] (known at bar close i)
                    state.entry_atr = float(atr[i]) if not np.isnan(atr[i]) else None
                    if state.entry_atr is not None:
                        set_sl_tp(state, cfg.sl_atr, cfg.tp_atr)
                else:
                    state.position = 0.0
                    state.entry_price = None
                    state.entry_atr = None
                    state.stop = None
                    state.take = None

        # close any open position at final close (end of day)
        if state.position != 0.0 and state.entry_price is not None:
            exit_price = close[-1]
            # holding pnl already booked till close[-2], adjust last segment:
            adj = state.position * cfg.unit_size * (exit_price - last_price)
            gross_pnl += adj
            net_pnl += adj

            last_fee = abs(state.position) * cfg.bp_fee * (state.entry_price + exit_price)
            fees += last_fee
            net_pnl -= last_fee
            num_trades += 1

        date = df.index[0].date()

        out.append({
            "Date": date,
            "Ticker": ticker_file,
            "grossPnL": float(gross_pnl),
            "feesTrade": float(fees),
            "netPnL": float(net_pnl),
            "numTrade": int(num_trades),
        })

    return out
