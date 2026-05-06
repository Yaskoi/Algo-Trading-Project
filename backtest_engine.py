from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from pathlib import Path
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ═════════════════════════════════════════════════════════════════════════════
# 0. CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    """
    Central configuration — pass one instance to every component.

    Data layout expected on disk
    ────────────────────────────
    data_dir/
      Yahoo_1m_DD_MM_YY/                       ← one folder per trading day
        df_{TICKER}_{YYYYMMDD}_{HHMMSS}.pkl    ← one pickle per ticker

    Tickers follow Yahoo Finance naming: ^GSPC, ^FTSE, AAPL, GC=F, EURUSD=X …
    """

    # ── Tickers ───────────────────────────────────────────────────────────
    tickers: List[str] = field(default_factory=lambda: [
        "^GSPC", "^FTSE", "^DJI", "^RUT",
        "AMD", "NVDA", "AMZN", "GME", "AMGN", "UNH",
        "TSLA", "OPTT", "PSTX", "GOOG", "MSFT", "AAPL", "QQQ",
        "NG=F", "JPYUSD=X", "GBPUSD=X",
    ])

    # ── Data ──────────────────────────────────────────────────────────────
    data_dir: str = "./data"

    # ── IS / OOS split ────────────────────────────────────────────────────
    is_months:  int = 5            # months used for calibration  (max)
    oos_months: int = 1            # months reserved for validation (min)

    # ── Execution & fees (project spec) ───────────────────────────────────
    basis_point: float = 0.0001    # fee = |pos| * bp * (S_start + S_end)

    # ── Trade sizing ──────────────────────────────────────────────────────
    sizing_method: str   = "fixed"  # "fixed" | "volatility" | "atr"
    target_vol:    float = 0.01     # daily vol target (used if sizing_method != fixed)
    max_position:  float = 1.0      # maximum absolute position size (notional units)
    atr_period:    int   = 14       # ATR lookback period for atr-based sizing

    # ── Risk management ───────────────────────────────────────────────────
    stop_loss:        Optional[float] = None   # e.g. 0.005 → 0.5% adverse move closes position
    take_profit:      Optional[float] = None   # e.g. 0.010 → 1.0% favourable move closes position
    daily_loss_limit: Optional[float] = None   # e.g. 0.02  → halt trading if daily net loss > 2%

    # ── Portfolio ─────────────────────────────────────────────────────────
    weights: Optional[Dict[str, float]] = None  # None → equal weight

    # ── Output ────────────────────────────────────────────────────────────
    output_dir: str  = "./results"
    plot:        bool = True


# ═════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADER
# ═════════════════════════════════════════════════════════════════════════════

class DataLoader:
    """
    Loads 1-min OHLCV data from Yahoo pickle files and performs IS/OOS split.

    Folder convention : Yahoo_1m_DD_MM_YY/
    File   convention : df_{TICKER}_{YYYYMMDD}_{HHMMSS}.pkl

    Each pickle  = one ticker x one trading day (390 bars, UTC index).
    Columns      = MultiIndex (Price, Ticker) -> flattened to open/high/low/close/volume.
    """

    def __init__(self, cfg: Config):
        self.cfg  = cfg
        self.root = Path(cfg.data_dir)

    def load(self, ticker: str) -> pd.DataFrame:
        daily_frames = []
        for folder in sorted(self.root.glob("Yahoo_1m_*")):
            if not folder.is_dir():
                continue
            matches = list(folder.glob(f"df_{ticker}_*.pkl"))
            if not matches:
                continue
            try:
                df_day = self._load_one(matches[0], ticker)
                if df_day is not None and len(df_day) > 0:
                    daily_frames.append(df_day)
            except Exception as e:
                print(f"[DataLoader] Warning - skipping {matches[0].name}: {e}")

        if not daily_frames:
            raise FileNotFoundError(
                f"[DataLoader] No data found for ticker '{ticker}' under {self.root}"
            )

        df = pd.concat(daily_frames).sort_index()
        df = df[~df.index.duplicated(keep="first")]
        print(
            f"[DataLoader] {ticker:12s} - "
            f"{df.index[0].date()} -> {df.index[-1].date()}  "
            f"({len(df):,} bars, {df.index.normalize().nunique()} days)"
        )
        return df

    def split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        idx_naive = df.index.tz_convert("UTC").tz_localize(None)
        months    = idx_naive.to_period("M").unique().sort_values()
        total     = len(months)
        need      = self.cfg.is_months + self.cfg.oos_months

        if total < need:
            raise ValueError(
                f"[DataLoader] Need >= {need} months of data; found {total}."
            )

        is_end    = months[self.cfg.is_months - 1].to_timestamp(how="end").tz_localize("UTC")
        oos_start = months[self.cfg.is_months].to_timestamp(how="start").tz_localize("UTC")

        df_is  = df.loc[df.index <= is_end].copy()
        df_oos = df.loc[df.index >= oos_start].copy()

        print(
            f"           IS  : {df_is.index[0].date()} -> {df_is.index[-1].date()} "
            f"({df_is.index.normalize().nunique()} days)"
        )
        print(
            f"           OOS : {df_oos.index[0].date()} -> {df_oos.index[-1].date()} "
            f"({df_oos.index.normalize().nunique()} days)"
        )
        return df_is, df_oos

    def _load_one(self, path: Path, ticker: str) -> Optional[pd.DataFrame]:
        raw = pd.read_pickle(path)

        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [price.lower() for price, _ in raw.columns]
        else:
            raw.columns = [c.lower() for c in raw.columns]

        raw = raw.rename(columns={"adj close": "adj_close"})

        required = {"open", "high", "low", "close", "volume"}
        missing  = required - set(raw.columns)
        if missing:
            raise ValueError(f"Missing columns {missing} in {path.name}")

        df = raw[["open", "high", "low", "close", "volume"]].copy()

        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True)
        elif df.index.tz is None:
            df.index = df.index.tz_localize("UTC")

        df["volume"] = df["volume"].fillna(0)
        return df.dropna(subset=["open", "high", "low", "close"])


# ═════════════════════════════════════════════════════════════════════════════
# 2. BASE STRATEGY  (plug-in interface)
# ═════════════════════════════════════════════════════════════════════════════

class BaseStrategy(ABC):
    """
    Plug-in interface — subclass this to implement any strategy.

    Contract
    --------
    fit(df_is)
        Calibrate / optimise parameters on IS data. Store results as
        instance attributes.

    generate_signal(bar, state) -> int in {-1, 0, +1}
        Direction only — the RiskManager handles sizing.
        bar   : pd.Series  — current 1-min candle (open/high/low/close/volume)
        state : dict       — mutable, persists across bars within a day,
                             reset to {} at the start of each new day.

    on_day_start(date, state)   [optional hook]
    on_day_end(date, state)     [optional hook]

    Minimal example
    ---------------
    class MyStrategy(BaseStrategy):
        def fit(self, df_is):
            self.window = 20

        def generate_signal(self, bar, state):
            hist = state.setdefault("hist", [])
            hist.append(bar["close"])
            if len(hist) < self.window:
                return 0
            ma = sum(hist[-self.window:]) / self.window
            return 1 if bar["close"] > ma else -1
    """

    @abstractmethod
    def fit(self, df_is: pd.DataFrame) -> None:
        ...

    @abstractmethod
    def generate_signal(self, bar: pd.Series, state: dict) -> int:
        ...

    def on_day_start(self, date, state: dict) -> None:
        pass

    def on_day_end(self, date, state: dict) -> None:
        pass

    def validate_signal(self, signal: int) -> int:
        if signal not in (-1, 0, 1):
            raise ValueError(
                f"[BaseStrategy] generate_signal() must return -1, 0 or 1; got {signal!r}"
            )
        return signal


# ═════════════════════════════════════════════════════════════════════════════
# 3. RISK MANAGER
# ═════════════════════════════════════════════════════════════════════════════

class RiskManager:
    """
    Sits between the strategy signal and the execution engine.

    Responsibilities
    ----------------
    1. Trade sizing
       Scales the raw {-1, 0, +1} direction signal to a float position size.
       Three methods available via Config.sizing_method:

         "fixed"      : size = max_position (always full unit, default)

         "volatility" : size = target_vol / realised_intraday_vol
                        where realised_vol is the std of log-returns over
                        the rolling buffer of closes seen so far today + history.

         "atr"        : size = target_vol / (ATR / price)
                        where ATR = mean(high - low) over atr_period bars.

       In all cases size is capped at max_position.

    2. Stop-loss
       If the adverse move from entry price >= stop_loss (fraction), force flat.

    3. Take-profit
       If the favourable move from entry price >= take_profit (fraction), force flat.

    4. Daily loss limit
       Once the cumulative net P&L for the day (normalised by first close)
       breaches -daily_loss_limit, all signals are forced to 0 for the rest
       of that day.

    All parameters come from Config. Setting a parameter to None disables it.

    State keys managed
    ------------------
    _rm_entry_price  : float
    _rm_halted       : bool
    _rm_vol_buffer   : list
    _rm_atr_buffer   : list
    """

    def __init__(self, cfg: Config):
        self.method      = cfg.sizing_method
        self.target_vol  = cfg.target_vol
        self.max_pos     = cfg.max_position
        self.atr_period  = cfg.atr_period
        self.stop_loss   = cfg.stop_loss
        self.take_profit = cfg.take_profit
        self.daily_limit = cfg.daily_loss_limit

    def on_day_start(self, state: dict) -> None:
        state["_rm_halted"]      = False
        state["_rm_entry_price"] = 0.0
        state.setdefault("_rm_vol_buffer", [])
        state.setdefault("_rm_atr_buffer", [])

    def process(
        self,
        raw_signal:   int,
        position:     float,
        bar:          pd.Series,
        state:        dict,
        day_net_pnl:  float,
        first_close:  float,
    ) -> float:
        price = bar["close"]

        state["_rm_vol_buffer"].append(price)
        state["_rm_atr_buffer"].append(bar["high"] - bar["low"])

        # ── 1. daily loss limit ───────────────────────────────────────────
        if self.daily_limit is not None and not state["_rm_halted"]:
            if first_close > 0 and (-day_net_pnl / first_close) >= self.daily_limit:
                state["_rm_halted"] = True
                print(f"  [RiskManager] Daily loss limit breached — trading halted.")

        if state["_rm_halted"]:
            return 0.0

        # ── 2. stop-loss / take-profit ────────────────────────────────────
        if position != 0 and state["_rm_entry_price"] > 0:
            move     = (price - state["_rm_entry_price"]) / state["_rm_entry_price"]
            pnl_frac = position * move

            if self.stop_loss is not None and pnl_frac <= -self.stop_loss:
                return 0.0

            if self.take_profit is not None and pnl_frac >= self.take_profit:
                return 0.0

        if raw_signal == 0:
            return 0.0

        # ── 3. trade sizing ───────────────────────────────────────────────
        size = self._compute_size(bar, state)

        if raw_signal != 0 and raw_signal != int(np.sign(position)):
            state["_rm_entry_price"] = price

        return float(raw_signal) * size

    def on_position_closed(self, state: dict) -> None:
        state["_rm_entry_price"] = 0.0

    def _compute_size(self, bar: pd.Series, state: dict) -> float:
        if self.method == "fixed":
            return self.max_pos

        elif self.method == "volatility":
            buf = state["_rm_vol_buffer"]
            if len(buf) < 2:
                return self.max_pos
            closes  = np.array(buf, dtype=float)
            returns = np.diff(np.log(closes))
            vol     = returns.std() if len(returns) > 1 else 0.0
            if vol == 0:
                return self.max_pos
            return min(self.target_vol / vol, self.max_pos)

        elif self.method == "atr":
            buf   = state["_rm_atr_buffer"]
            price = bar["close"]
            if len(buf) < self.atr_period or price == 0:
                return self.max_pos
            atr  = float(np.mean(buf[-self.atr_period:]))
            norm = atr / price
            if norm == 0:
                return self.max_pos
            return min(self.target_vol / norm, self.max_pos)

        else:
            raise ValueError(
                f"[RiskManager] Unknown sizing_method '{self.method}'. "
                "Choose 'fixed', 'volatility', or 'atr'."
            )


# ═════════════════════════════════════════════════════════════════════════════
# 4. EXECUTION ENGINE
# ═════════════════════════════════════════════════════════════════════════════

class ExecutionEngine:
    """
    Executes trades bar-by-bar and computes gross / net P&L.

    Fee model
    ---------
        fee_close  = |position|  * bp * (entry_price + price_now)
        fee_open   = |desired|   * bp * (price_now   + price_next)

        price_next : open price of the next bar (realistic round-trip estimate).
                     Falls back to price_now if unavailable.

    P&L model
    ---------
        gross_pnl_bar = position_{t-1} * (price_now - price_prev)
        net_pnl_bar   = gross_pnl_bar  - fee_this_bar

    Execution timing
    ----------------
        Decision made on close of bar t  →  executed at open of bar t+1.
        This is enforced by the Backtester (_replay loop with pending_desired).
    """

    def __init__(self, cfg: Config):
        self.bp = cfg.basis_point

    def execute(
        self,
        desired:     float,
        position:    float,
        price_now:   float,
        price_prev:  float,
        entry_price: float,
        price_next:  float = None,   # open of next bar for realistic fee estimate
    ) -> Tuple[float, float, float, float, float]:
        """
        Returns
        -------
        new_position, gross_pnl, net_pnl, fee, new_entry_price
        """
        gross_pnl = position * (price_now - price_prev)

        fee       = 0.0
        new_entry = entry_price

        if desired != position:
            # closing leg
            if position != 0:
                fee += abs(position) * self.bp * (entry_price + price_now)
            # opening leg
            if desired != 0:
                p_next    = price_next if price_next is not None else price_now
                fee      += abs(desired) * self.bp * (price_now + p_next)
                new_entry = price_now
            else:
                new_entry = 0.0

        return desired, gross_pnl, gross_pnl - fee, fee, new_entry


# ═════════════════════════════════════════════════════════════════════════════
# 5. BACKTESTER
# ═════════════════════════════════════════════════════════════════════════════

class Backtester:
    """
    Orchestrates the full IS -> OOS walk-forward pipeline for one ticker.

    Pipeline
    --------
    1. Load data         ->  DataLoader
    2. IS / OOS split        (strictly chronological, no overlap)
    3. strategy.fit(df_is)
    4. _replay(df_is,  "IS")    diagnostic
    5. _replay(df_oos, "OOS")   official evaluation

    Bar-by-bar loop — strictly causal
    ----------------------------------
    Bar t  : strategy.generate_signal()  ->  pending_desired
    Bar t+1: ExecutionEngine.execute()   ->  executed at open(t+1)

    This one-bar delay eliminates execution-on-signal look-ahead bias.
    """

    def __init__(self, cfg: Config, strategy: BaseStrategy):
        self.cfg      = cfg
        self.strategy = strategy
        self.engine   = ExecutionEngine(cfg)
        self.risk     = RiskManager(cfg)
        self.loader   = DataLoader(cfg)

    def run(self, ticker: str) -> pd.DataFrame:
        df            = self.loader.load(ticker)
        df_is, df_oos = self.loader.split(df)

        print(f"  [Backtester] Fitting strategy on IS data ...")
        self.strategy.fit(df_is)

        daily_is  = self._replay(df_is,  period="IS")
        daily_oos = self._replay(df_oos, period="OOS")

        return pd.concat([daily_is, daily_oos]).sort_index()

    def _replay(self, df: pd.DataFrame, period: str) -> pd.DataFrame:
        records        = []
        prev_day_close = None
        dates          = df.index.normalize().unique().sort_values()

        for date in dates:
            day_df = df[df.index.normalize() == date]
            if len(day_df) < 2:
                continue

            state           = {}
            position        = 0.0
            entry_price     = 0.0
            day_gross       = 0.0
            day_net         = 0.0
            day_fees        = 0.0
            n_trades        = 0
            first_close     = day_df["close"].iloc[0]
            day_open_ref    = prev_day_close if prev_day_close is not None else first_close
            pending_desired = 0.0   # signal decided on bar t, executed at open of bar t+1

            self.strategy.on_day_start(date, state)
            self.risk.on_day_start(state)

            bars = list(day_df.itertuples())

            for i, bar in enumerate(bars):
                price_prev = bars[i - 1].close if i > 0 else day_open_ref
                price_next = bars[i + 1].close if i < len(bars) - 1 else bar.close

                # ── Execute pending order at open of current bar ──────────
                if i > 0:
                    if pending_desired == 0.0 and position != 0.0:
                        self.risk.on_position_closed(state)

                    position, gross, net, fee, entry_price = self.engine.execute(
                        pending_desired, position, bar.open, price_prev, entry_price, price_next
                    )
                    day_gross += gross
                    day_net   += net
                    day_fees  += fee
                    if fee > 0:
                        n_trades += 1

                # ── Decide signal on close of current bar ─────────────────
                if i == len(bars) - 1:
                    # force flat — no overnight positions
                    pending_desired = 0.0
                else:
                    bar_series = pd.Series(
                        {"open": bar.open, "high": bar.high,
                         "low": bar.low, "close": bar.close, "volume": bar.volume},
                        name=bar.Index,
                    )
                    raw_signal      = self.strategy.validate_signal(
                        self.strategy.generate_signal(bar_series, state)
                    )
                    pending_desired = self.risk.process(
                        raw_signal  = raw_signal,
                        position    = position,
                        bar         = bar_series,
                        state       = state,
                        day_net_pnl = day_net,
                        first_close = first_close,
                    )

            self.strategy.on_day_end(date, state)
            prev_day_close = day_df["close"].iloc[-1]

            if first_close > 0:
                records.append({
                    "date":         pd.Timestamp(date),
                    "gross_return": day_gross / first_close,
                    "net_return":   day_net   / first_close,
                    "fees":         day_fees  / first_close,
                    "n_trades":     n_trades,
                    "period":       period,
                })

        daily = pd.DataFrame(records).set_index("date")
        print(
            f"  [Backtester] {period} replay - "
            f"{len(daily)} days | "
            f"avg {daily['n_trades'].mean():.1f} trades/day | "
            f"net return: {daily['net_return'].sum()*100:.2f}%"
        )
        return daily


# ═════════════════════════════════════════════════════════════════════════════
# 6. PORTFOLIO ENGINE
# ═════════════════════════════════════════════════════════════════════════════

class PortfolioEngine:
    """
    Combines per-ticker daily returns into a weighted portfolio.

    portfolio_return_t = sum_i  w_i * daily_return_i_t
    Weights are normalised to sum to 1.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def combine(self, daily_returns: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        tickers = list(daily_returns.keys())
        weights = self._resolve_weights(tickers)

        gross  = pd.DataFrame({t: daily_returns[t]["gross_return"] for t in tickers})
        net    = pd.DataFrame({t: daily_returns[t]["net_return"]   for t in tickers})
        trades = pd.DataFrame({t: daily_returns[t]["n_trades"]     for t in tickers})

        w = pd.Series(weights)
        return pd.DataFrame({
            "gross_return": gross.fillna(0).dot(w),
            "net_return":   net.fillna(0).dot(w),
            "n_trades":     trades.fillna(0).sum(axis=1),
            "period":       "OOS",
        })

    def _resolve_weights(self, tickers: List[str]) -> Dict[str, float]:
        if self.cfg.weights:
            w = {t: self.cfg.weights.get(t, 0.0) for t in tickers}
        else:
            w = {t: 1.0 for t in tickers}
        total = sum(w.values())
        if total == 0:
            raise ValueError("[PortfolioEngine] All weights are zero.")
        return {t: v / total for t, v in w.items()}


# ═════════════════════════════════════════════════════════════════════════════
# 7. PERFORMANCE REPORTER
# ═════════════════════════════════════════════════════════════════════════════

class PerformanceReporter:
    """
    Computes and displays the OOS performance matrix:

    | Asset | Net Return | Ann. Sharpe | Max DD | Avg Daily Trades |
    """

    TRADING_DAYS = 252

    def compute_metrics(self, daily: pd.DataFrame, label: str) -> dict:
        r = daily["net_return"].dropna()
        return {
            "label":      label,
            "net_return": self._annualised_return(r),
            "sharpe":     self._annualised_sharpe(r),
            "max_dd":     self._max_drawdown(r),
            "calmar":     self._calmar_ratio(r),
            "avg_trades": daily["n_trades"].mean(),
        }

    def _annualised_return(self, r: pd.Series) -> float:
        if len(r) == 0:
            return 0.0
        return (1 + r).prod() ** (self.TRADING_DAYS / len(r)) - 1

    def _annualised_sharpe(self, r: pd.Series) -> float:
        if len(r) < 2 or r.std() == 0:
            return 0.0
        return (r.mean() / r.std()) * np.sqrt(self.TRADING_DAYS)

    def _max_drawdown(self, r: pd.Series) -> float:
        cum  = (1 + r).cumprod()
        peak = cum.cummax()
        return ((cum - peak) / peak).min()
    
    def _calmar_ratio(self, r: pd.Series) -> float:
        ann_return = self._annualised_return(r)
        max_dd     = self._max_drawdown(r)
        if max_dd == 0:
            return 0.0
        return ann_return / abs(max_dd)

    def print_table(self, rows: List[dict]) -> None:
        header = (f"{'Asset':<14} {'Net Return':>11} {'Ann. Sharpe':>12} "
                f"{'Calmar':>8} {'Max DD':>8} {'Avg Daily Trades':>17}")
        sep = "-" * len(header)
        print(f"\n{sep}\n{header}\n{sep}")
        for row in rows:
            is_port = row["label"] == "Portfolio"
            name    = f"{'> ' if is_port else ''}{row['label']}"
            print(
                f"{name:<14} "
                f"{row['net_return']*100:>10.2f}% "
                f"{row['sharpe']:>12.2f} "
                f"{row['calmar']:>8.2f} "
                f"{row['max_dd']*100:>7.2f}% "
                f"{row['avg_trades']:>17.1f}"
            )
        print(sep)

    def export_trading_log(
        self,
        oos_daily:  Dict[str, pd.DataFrame],
        portfolio:  pd.DataFrame,
        output_dir: str,
    ) -> pd.DataFrame:

        frames    = []
        all_items = list(oos_daily.items()) + [("Portfolio", portfolio)]

        for ticker, df in all_items:
            if len(df) == 0:
                continue

            r      = df["net_return"].copy()
            g      = df["gross_return"].copy()
            fees   = df["fees"].copy() if "fees" in df.columns else (g - r)
            trades = df["n_trades"].copy()

            cum_gross      = (1 + g).cumprod() - 1
            cum_net        = (1 + r).cumprod() - 1
            peak           = (1 + r).cumprod().cummax()
            drawdown       = ((1 + r).cumprod() - peak) / peak
            daily_vol      = r.rolling(20, min_periods=2).std() * np.sqrt(self.TRADING_DAYS)
            roll_mean      = r.rolling(20, min_periods=2).mean()
            roll_std       = r.rolling(20, min_periods=2).std()
            rolling_sharpe = (roll_mean / roll_std.replace(0, np.nan)) * np.sqrt(self.TRADING_DAYS)

            log = pd.DataFrame({
                "date":           df.index,
                "ticker":         ticker,
                "gross_return":   (g      * 100).round(4),
                "net_return":     (r      * 100).round(4),
                "fees":           (fees   * 100).round(4),
                "n_trades":       trades.astype(int),
                "cum_gross":      (cum_gross     * 100).round(4),
                "cum_net":        (cum_net       * 100).round(4),
                "drawdown":       (drawdown      * 100).round(4),
                "daily_vol":      (daily_vol     * 100).round(4),
                "rolling_sharpe": rolling_sharpe.round(4),
            }).reset_index(drop=True)

            frames.append(log)

        full_log = (
            pd.concat(frames, ignore_index=True)
            .sort_values(["date", "ticker"])
        )

        out = Path(output_dir) / "trading_log_oos.csv"
        full_log.to_csv(out, index=False)
        print(f"[Reporter] Trading log saved -> {out}  ({len(full_log)} rows)")
        return full_log

    def plot_equity_curves(
        self,
        results:    Dict[str, pd.DataFrame],
        portfolio:  pd.DataFrame,
        output_dir: str,
    ) -> None:
        all_items = list(results.items()) + [("Portfolio", portfolio)]
        n     = len(all_items)
        ncols = 3
        nrows = (n + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
        axes = axes.flatten()

        for ax, (label, daily) in zip(axes, all_items):
            gross_cum = (1 + daily["gross_return"]).cumprod() - 1
            net_cum   = (1 + daily["net_return"]).cumprod()   - 1
            ax.plot(gross_cum.index, gross_cum * 100, label="Gross", alpha=0.7, linewidth=1.2)
            ax.plot(net_cum.index,   net_cum   * 100, label="Net",   linewidth=1.4)
            ax.axhline(0, color="grey", linewidth=0.6, linestyle="--")
            ax.set_title(label, fontweight="bold")
            ax.set_ylabel("Cumulative Return (%)")
            ax.legend(fontsize=8)
            ax.tick_params(axis="x", rotation=30)
            ax.grid(True, alpha=0.3)

        for ax in axes[n:]:
            ax.set_visible(False)

        fig.suptitle("OOS Equity Curves - Gross vs Net", fontsize=14, fontweight="bold", y=1.01)
        plt.tight_layout()
        out = Path(output_dir) / "equity_curves.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.show()
        print(f"[Reporter] Saved -> {out}")


# ═════════════════════════════════════════════════════════════════════════════
# 8. RUN BACKTEST
# ═════════════════════════════════════════════════════════════════════════════

def run_backtest(
    strategy,          # BaseStrategy  OR  Dict[str, BaseStrategy]
    cfg: Config,
) -> Dict[str, pd.DataFrame]:
    """
    Main entry point.

    Parameters
    ----------
    strategy : BaseStrategy or dict[ticker -> BaseStrategy]
        Pass a dict to use a different strategy per ticker (e.g. pair trading
        on some tickers, momentum on others).
    cfg : Config

    Returns
    -------
    dict with keys: tickers, oos, portfolio, metrics, trading_log
    """

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    portfolio = PortfolioEngine(cfg)
    reporter  = PerformanceReporter()

    # ── Resolve ticker -> strategy mapping ───────────────────────────────
    if isinstance(strategy, dict):
        strategy_map = strategy
    else:
        strategy_map = {t: strategy for t in cfg.tickers}

    # ── Run backtester per ticker ─────────────────────────────────────────
    all_daily: Dict[str, pd.DataFrame] = {}

    for ticker in cfg.tickers:
        print(f"\n{'='*60}\n  {ticker}\n{'='*60}")
        strat = strategy_map.get(ticker)
        if strat is None:
            print(f"  [run_backtest] No strategy for {ticker}, skipping.")
            continue
        try:
            bt = Backtester(cfg, strat)
            all_daily[ticker] = bt.run(ticker)
        except FileNotFoundError as e:
            print(f"  [run_backtest] Skipping {ticker} - {e}")
        except ValueError as e:
            print(f"  [run_backtest] Skipping {ticker} - {e}")

    if not all_daily:
        raise RuntimeError("[run_backtest] No ticker could be loaded. Check data_dir.")

    # ── OOS only for portfolio & metrics ─────────────────────────────────
    oos_daily = {
        t: df[df["period"] == "OOS"]
        for t, df in all_daily.items()
        if (df["period"] == "OOS").any()
    }

    port_daily = portfolio.combine(oos_daily)

    # ── Performance table ─────────────────────────────────────────────────
    rows = [reporter.compute_metrics(df, t) for t, df in oos_daily.items()]
    rows.append(reporter.compute_metrics(port_daily, "Portfolio"))
    reporter.print_table(rows)

    # ── Export CSVs ───────────────────────────────────────────────────────
    for ticker, df in all_daily.items():
        safe_name = ticker.replace("^", "").replace("=", "_")
        df.to_csv(Path(cfg.output_dir) / f"{safe_name}_daily.csv")
    port_daily.to_csv(Path(cfg.output_dir) / "portfolio_daily.csv")
    print(f"\n[run_backtest] CSVs saved -> {cfg.output_dir}/")

    # ── Trading log + equity curves ───────────────────────────────────────
    trading_log = reporter.export_trading_log(oos_daily, port_daily, cfg.output_dir)

    if cfg.plot:
        reporter.plot_equity_curves(oos_daily, port_daily, cfg.output_dir)

    return {
        "tickers":     all_daily,
        "oos":         oos_daily,
        "portfolio":   port_daily,
        "metrics":     rows,
        "trading_log": trading_log,
    }