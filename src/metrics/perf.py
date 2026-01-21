from __future__ import annotations

import numpy as np
import pandas as pd


def _daily_portfolio_returns(df: pd.DataFrame, portfolio_name: str = "Portfolio") -> pd.Series:
    # df has Date/Ticker/netPnL ; we build a daily portfolio return proxy
    # return = netPnL (unit notional). For ranking IS, this is enough and consistent cross-strategy.
    port = (
        df.groupby("Date", as_index=True)["netPnL"]
          .sum()
          .sort_index()
    )
    return port


def sharpe_ratio(daily_returns: pd.Series, ann_factor: float = 252.0) -> float:
    if daily_returns.empty:
        return 0.0
    mu = float(daily_returns.mean())
    sd = float(daily_returns.std(ddof=1))
    if sd == 0 or np.isnan(sd):
        return 0.0
    return (mu / sd) * np.sqrt(ann_factor)


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    return float(dd.min())


def annualized_return(daily_returns: pd.Series, ann_factor: float = 252.0) -> float:
    if daily_returns.empty:
        return 0.0
    return float(daily_returns.mean() * ann_factor)


def build_oos_matrix(df: pd.DataFrame, portfolio_name: str = "Portfolio") -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Asset", "Net Return Ann.", "Sharpe", "MaxDD", "Avg Daily Trades"])

    assets = sorted(df["Ticker"].unique().tolist())

    rows = []
    for a in assets:
        sub = df[df["Ticker"] == a].copy()
        rets = sub.groupby("Date")["netPnL"].sum().sort_index()
        eq = rets.cumsum()

        rows.append({
            "Asset": a,
            "Net Return Ann.": annualized_return(rets),
            "Sharpe": sharpe_ratio(rets),
            "MaxDD": max_drawdown(eq),
            "Avg Daily Trades": float(sub.groupby("Date")["numTrade"].sum().mean()) if not sub.empty else 0.0
        })

    # Portfolio row (sum netPnL across tickers)
    port_rets = df.groupby("Date")["netPnL"].sum().sort_index()
    port_eq = port_rets.cumsum()

    rows.append({
        "Asset": portfolio_name,
        "Net Return Ann.": annualized_return(port_rets),
        "Sharpe": sharpe_ratio(port_rets),
        "MaxDD": max_drawdown(port_eq),
        "Avg Daily Trades": float(df.groupby("Date")["numTrade"].sum().mean())
    })

    return pd.DataFrame(rows)


def score_is_for_selection(df_is: pd.DataFrame) -> float:
    # Score simple : Sharpe portfolio (IS)
    if df_is is None or df_is.empty:
        return float("-inf")
    rets = df_is.groupby("Date")["netPnL"].sum().sort_index()
    return sharpe_ratio(rets)
