import numpy as np
from .base import Strategy

class DMACStrategy(Strategy):
    """
    Double Moving Average Crossover.
    Live/causal: uses only history up to current bar.
    Returns target position in {-1, 0, +1}.
    """
    def __init__(self, fast: int = 20, slow: int = 60, allow_short: bool = True):
        if fast >= slow:
            raise ValueError("fast must be < slow")
        self.fast = fast
        self.slow = slow
        self.allow_short = allow_short

    def on_bar(self, t, row, history):
        # history is a DataFrame up to current time
        close = history["Close"].to_numpy()
        if close.size < self.slow:
            return 0.0

        fast_ma = close[-self.fast:].mean()
        slow_ma = close[-self.slow:].mean()

        if fast_ma > slow_ma:
            return 1.0
        else:
            return -1.0 if self.allow_short else 0.0
