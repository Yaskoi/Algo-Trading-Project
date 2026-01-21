from dataclasses import dataclass
from pathlib import Path

BP = 1e-4  # basis-point = 0.0001

@dataclass
class BacktestConfig:
    data_root: Path
    tickers: list[str]
    portfolio_tickers: list[str]
    weights: list[float]

    # Split IS/OOS (en jours)
    is_ratio: float = 5/6   # ~ 5 mois sur 6
    timezone: str = "UTC"

    # Exécution baseline
    price_col: str = "Close"     # prix utilisé pour exécution
    allow_short: bool = True

    # Risk management (à activer plus tard)
    use_stoploss: bool = True
    use_takeprofit: bool = True

    # Frais
    bp_fee: float = BP
