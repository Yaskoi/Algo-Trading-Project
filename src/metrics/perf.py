import numpy as np
import pandas as pd

def equity_curve_from_returns(daily_returns: pd.Series) -> np.ndarray:
    r = daily_returns.fillna(0.0).to_numpy(dtype=float)
    return np.cumprod(1.0 + r)

def max_drawdown_from_equity(equity: np.ndarray) -> float:
    # equity must be strictly positive and start at 1 ideally
    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1.0
    return float(np.min(dd))

def sharpe_ratio(daily_returns: pd.Series, annual_factor: int = 252) -> float:
    r = daily_returns.dropna().astype(float)
    if len(r) < 2:
        return 0.0
    std = r.std(ddof=1)
    if std == 0 or np.isnan(std):
        return 0.0
    return float((r.mean() / std) * np.sqrt(annual_factor))

def annualized_return(daily_returns: pd.Series, annual_factor: int = 252) -> float:
    r = daily_returns.dropna().astype(float)
    if len(r) == 0:
        return 0.0
    equity_end = float(np.prod(1.0 + r))
    return float(equity_end ** (annual_factor / len(r)) - 1.0)

def build_oos_matrix(pnl_df: pd.DataFrame, tickers: list[str], portfolio_col="Portfolio", notional: float = 1e6):
    out = []
    for tk in tickers + [portfolio_col]:
        sub = pnl_df[pnl_df["Ticker"] == tk].copy()
        if sub.empty:
            continue

        # daily returns = daily net PnL / notional
        rets = (sub["netPnL"].astype(float) / float(notional))

        equity = equity_curve_from_returns(rets)
        out.append({
            "Asset": tk,
            "Net Return Ann.": annualized_return(rets),
            "Sharpe": sharpe_ratio(rets),
            "MaxDD": max_drawdown_from_equity(equity),
            "Avg Daily Trades": float(sub["numTrade"].mean())
        })

    return pd.DataFrame(out)
