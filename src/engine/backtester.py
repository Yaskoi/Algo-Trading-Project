import pandas as pd
from pathlib import Path
from ..data.loader import load_pkl, list_pkl_files, ticker_from_filename
from .broker import Broker

def _to_float(x):
    # gère float, numpy scalar, list([x]), np.array([x])
    if isinstance(x, (list, tuple)) and len(x) > 0:
        x = x[0]
    try:
        import numpy as np
        if hasattr(x, "item"):
            return float(x.item())
    except Exception:
        pass
    return float(x)

def run_day_backtest(day_dir: Path, tickers: list[str], strategy_factory, bp_fee: float, price_col="Close"):
    """
    strategy_factory(ticker) -> Strategy instance
    Returns list of dicts results per ticker for that day.
    """
    results = []
    files = list_pkl_files(day_dir)
    files = [p for p in files if ticker_from_filename(p) in tickers]

    for p in files:
        ticker = ticker_from_filename(p)
        df = load_pkl(p)
        if df.empty:
            continue

        strat = strategy_factory(ticker)
        broker = Broker(bp_fee=bp_fee)

        prices = df[price_col].values
        for i in range(len(df)):
            row = df.iloc[i]
            hist = df.iloc[:i+1]  # causal
            next_price = prices[i+1] if i+1 < len(df) else prices[i]

            target_pos = strat.on_bar(df.index[i], row, hist)
            broker.execute_target_position(target_pos, price=prices[i], next_price=next_price)

        gross = _to_float(broker.state.realized_pnl)
        fees  = _to_float(broker.state.fees)
        net   = gross - fees

        results.append({
            "Date": df.index[0].date(),
            "Ticker": ticker,
            "grossPnL": gross,
            "feesTrade": fees,
            "netPnL": net,
            "numTrade": int(broker.state.num_trades),
        })
    return results
