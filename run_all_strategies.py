# run_all_strategies.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Type

import pandas as pd

from src.config import BacktestConfig
from src.data.calendar import list_day_directories
from src.engine.backtester import run_backtest_days
from src.metrics.perf import build_oos_matrix, score_is_for_selection

# Strategies
from src.strategies.ma_cross import MACrossStrategy
from src.strategies.macd_hist import MACDHistStrategy
from src.strategies.bollinger import BollingerMRStrategy
from src.strategies.hma import HMATrendStrategy
from src.strategies.donchian import DonchianBreakoutStrategy
from src.strategies.rma_zscore import RMAZScoreStrategy
from src.strategies.vol_target import VolTargetStrategy
from src.strategies.orb import ORBStrategy


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _uniform_weights(tickers: List[str]) -> List[float]:
    if not tickers:
        return []
    w = 1.0 / len(tickers)
    return [w] * len(tickers)


def main() -> None:
    # =======================
    # CONFIG (project rules)
    # =======================
    cfg = BacktestConfig(
        data_root=Path("Data"),
        results_root=Path("Results"),
        tickers=[
            "^GSPC", "^FTSE", "^DJI", "^RUT",
            "AMD", "NVDA", "AMZN", "GME", "AMGN", "UNH",
            "TSLA", "OPTT", "PSTX", "GOOG", "MSFT", "AAPL",
            "QQQ", "NG=F", "JPYUSD=X", "GBPUSD=X",
        ],
        # if your BacktestConfig expects weights, set uniform by default
        weights=_uniform_weights([
            "^GSPC", "^FTSE", "^DJI", "^RUT",
            "AMD", "NVDA", "AMZN", "GME", "AMGN", "UNH",
            "TSLA", "OPTT", "PSTX", "GOOG", "MSFT", "AAPL",
            "QQQ", "NG=F", "JPYUSD=X", "GBPUSD=X",
        ]),
        is_ratio=5 / 6,          # ~5 months IS, ~1 month OOS (day-based split)
        bp_fee=0.0001,           # basis-point fee
        price_col="Close",
        open_col="Open",
        high_col="High",
        low_col="Low",
        # Risk (ATR SL/TP)
        atr_period=14,
        sl_atr=1.5,
        tp_atr=2.0,
        # Notional (unit size)
        unit_size=1.0,
        # Execution (no look-ahead: decide on bar i close, execute i+1 open)
        exec_at="next_open",
        max_days=None,           # set e.g. 30 to test faster
        seed=42,
    )

    _ensure_dir(cfg.results_root)

    # =======================
    # DATA SPLIT
    # =======================
    day_dirs = list_day_directories(cfg.data_root)
    if cfg.max_days:
        day_dirs = day_dirs[: cfg.max_days]

    if not day_dirs:
        print("❌ Aucun dossier Yahoo_1m_* trouvé dans Data/")
        return

    n = len(day_dirs)
    split = int(n * cfg.is_ratio)
    is_days = day_dirs[:split]
    oos_days = day_dirs[split:]

    print(f"✅ Days total={n} | IS={len(is_days)} | OOS={len(oos_days)}")
    print(f"📁 Results -> {cfg.results_root.resolve()}")

    # =======================
    # STRATEGY SPECS + GRIDS
    # =======================
    StrategySpec = Tuple[str, Type, List[Dict[str, Any]]]
    strategy_specs: List[StrategySpec] = [
        ("MA_Cross", MACrossStrategy, [
            {"fast": 10, "slow": 30, "allow_short": True},
            {"fast": 20, "slow": 60, "allow_short": True},
            {"fast": 30, "slow": 90, "allow_short": True},
        ]),
        ("MACD_Hist", MACDHistStrategy, [
            {"n_short": 12, "n_long": 26, "n_signal": 9, "allow_short": True, "min_hold": 5},
            {"n_short": 8,  "n_long": 21, "n_signal": 5, "allow_short": True, "min_hold": 5},
        ]),
        ("Bollinger_MR", BollingerMRStrategy, [
            {"window": 20, "k": 2.0, "allow_short": True},
            {"window": 30, "k": 2.0, "allow_short": True},
        ]),
        ("HMA_Trend", HMATrendStrategy, [
            {"period": 55, "allow_short": True},
            {"period": 34, "allow_short": True},
        ]),
        ("Donchian_BO", DonchianBreakoutStrategy, [
            {"window": 20, "allow_short": True},
            {"window": 55, "allow_short": True},
        ]),
        ("RMA_ZScore", RMAZScoreStrategy, [
            {"window": 60, "z_entry": 1.0, "z_exit": 0.2, "allow_short": True},
            {"window": 120, "z_entry": 1.2, "z_exit": 0.3, "allow_short": True},
        ]),
        ("Vol_Target", VolTargetStrategy, [
            {"window": 60, "target_vol": 0.002, "max_leverage": 2.0, "allow_short": True},
            {"window": 120, "target_vol": 0.0015, "max_leverage": 2.0, "allow_short": True},
        ]),
        ("ORB", ORBStrategy, [
            {"orb_minutes": 15, "breakout_k": 0.0, "allow_short": True},
            {"orb_minutes": 30, "breakout_k": 0.0, "allow_short": True},
        ]),
    ]

    all_oos_rows: List[pd.DataFrame] = []
    summary_rows: List[Dict[str, Any]] = []

    for strat_name, strat_cls, grid in strategy_specs:
        print(f"\n================= {strat_name} =================")

        strat_dir = _ensure_dir(cfg.results_root / strat_name)

        # 1) Tuning on IS
        best_params: Dict[str, Any] | None = None
        best_score = float("-inf")
        best_is_df: pd.DataFrame | None = None

        for params in grid:
            is_df = run_backtest_days(cfg, is_days, strat_cls, params, tag="IS")

            # Save each grid run (optional but useful)
            grid_tag = "_".join([f"{k}={v}" for k, v in params.items()])
            grid_path = strat_dir / f"grid_IS_{grid_tag}.csv"
            is_df.to_csv(grid_path, index=False)

            score = score_is_for_selection(is_df)
            if score > best_score:
                best_score = score
                best_params = params
                best_is_df = is_df

        if best_params is None:
            print("⚠️ Aucun résultat IS (données manquantes ?) => skip stratégie")
            continue

        # persist best params + IS best pnl
        (strat_dir / "best_params.json").write_text(json.dumps(best_params, indent=2), encoding="utf-8")
        if best_is_df is not None:
            best_is_df.to_csv(strat_dir / "daily_pnl_IS_best.csv", index=False)

        print(f"✅ Best params (IS): {best_params} | score={best_score:.4f}")

        # 2) Run OOS with best params
        oos_df = run_backtest_days(cfg, oos_days, strat_cls, best_params, tag="OOS")
        oos_df.to_csv(strat_dir / "daily_pnl_OOS.csv", index=False)

        # 3) OOS Matrix
        matrix = build_oos_matrix(oos_df, portfolio_name="Portfolio")
        matrix.to_csv(strat_dir / "oos_matrix.csv", index=False)

        # Keep for global files
        all_oos_rows.append(oos_df.assign(Strategy=strat_name))

        # Small one-line summary for Portfolio row
        port_row = matrix[matrix["Asset"] == "Portfolio"].copy()
        if not port_row.empty:
            r = port_row.iloc[0].to_dict()
            r["Strategy"] = strat_name
            summary_rows.append(r)

        print(matrix)

    # =======================
    # GLOBAL EXPORTS
    # =======================
    if all_oos_rows:
        df_all = pd.concat(all_oos_rows, ignore_index=True)
        df_all.to_csv(cfg.results_root / "ALL_strategies_daily_pnl_OOS.csv", index=False)

    if summary_rows:
        df_summary = pd.DataFrame(summary_rows)
        # Keep a clean set of columns if present
        cols = ["Strategy", "Net Return Ann.", "Sharpe", "MaxDD", "Avg Daily Trades"]
        cols = [c for c in cols if c in df_summary.columns]
        df_summary = df_summary[cols] if cols else df_summary
        df_summary.to_csv(cfg.results_root / "SUMMARY_Portfolio_OOS.csv", index=False)

    # Save config snapshot for reproducibility
    (cfg.results_root / "config_snapshot.json").write_text(
        json.dumps(cfg.__dict__, indent=2, default=str),
        encoding="utf-8"
    )

    print("\n✅ Done. Check Results/<StrategyName>/ for outputs.")
    print("✅ Global files:")
    print("   - Results/ALL_strategies_daily_pnl_OOS.csv")
    print("   - Results/SUMMARY_Portfolio_OOS.csv")
    print("   - Results/config_snapshot.json")


if __name__ == "__main__":
    main()
