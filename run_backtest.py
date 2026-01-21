from pathlib import Path
import pandas as pd

from src.config import BacktestConfig
from src.data.calendar import list_day_directories
from src.engine.backtester import run_day_backtest
from src.metrics.perf import build_oos_matrix

from src.strategies.macd_hist import MACDHistStrategy

def strategy_factory(ticker):
    return MACDHistStrategy(n_short=12, n_long=26, n_signal=9, eps=0.0, allow_short=True)

def main():
    DATA_ROOT = Path("Data")

    portfolio_tickers = [
        '^GSPC','^FTSE','^DJI','^RUT','AMD','NVDA','AMZN','GME','AMGN','UNH',
        'TSLA','OPTT','PSTX','GOOG','MSFT','AAPL','QQQ','NG=F','JPYUSD=X','GBPUSD=X'
    ]

    weights = [1 / len(portfolio_tickers)] * len(portfolio_tickers)

    cfg = BacktestConfig(
        data_root=DATA_ROOT,
        tickers=portfolio_tickers,
        portfolio_tickers=portfolio_tickers,
        weights=weights
    )

    day_dirs = list_day_directories(cfg.data_root)
    n = len(day_dirs)
    split = int(n * cfg.is_ratio)
    oos_dirs = day_dirs[split:]

    all_results = []
    for d in oos_dirs:
        daily = run_day_backtest(
            d,
            cfg.portfolio_tickers,
            strategy_factory,
            cfg.bp_fee,
            price_col=cfg.price_col
        )
        all_results.extend(daily)

    pnl_df = pd.DataFrame(all_results)
    print(pnl_df.head())

    w = dict(zip(cfg.portfolio_tickers, cfg.weights))

    port = (
        pnl_df.assign(weight=pnl_df["Ticker"].map(w).fillna(0.0))
            .assign(netPnL=lambda x: x["netPnL"] * x["weight"])
            .groupby("Date", as_index=False)[["netPnL"]].sum()
    )

    port["Ticker"] = "Portfolio"
    port["grossPnL"] = port["netPnL"]
    port["feesTrade"] = 0.0
    port["numTrade"] = 0


    pnl_all = pd.concat([pnl_df, port], ignore_index=True)

    matrix = build_oos_matrix(pnl_all, cfg.portfolio_tickers, portfolio_col="Portfolio", notional=1e6)
    print(matrix)

if __name__ == "__main__":
    main()
