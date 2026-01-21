import numpy as np
from .base import Strategy

def ema_last(x: np.ndarray, span: int) -> float:
    """
    EMA causal, renvoie la dernière valeur (pas de look-ahead).
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return float("nan")
    alpha = 2.0 / (span + 1.0)
    e = x[0]
    for v in x[1:]:
        e = alpha * v + (1.0 - alpha) * e
    return float(e)

class MACDHistStrategy(Strategy):
    """
    MACD Histogram strategy (cours):
      DIFF_t = EMA_short(Close) - EMA_long(Close)
      Signal_t = EMA(DIFF, n_signal)
      Hist_t = DIFF_t - Signal_t

    Rule (simple):
      long  if Hist_t > eps
      short if Hist_t < -eps (si allow_short)
      else flat

    Tout est causal: basé uniquement sur history[:t].
    """
    def __init__(self, n_short: int = 12, n_long: int = 26, n_signal: int = 9, eps: float = 0.0, allow_short: bool = True):
        if n_short >= n_long:
            raise ValueError("n_short must be < n_long")
        self.n_short = n_short
        self.n_long = n_long
        self.n_signal = n_signal
        self.eps = float(eps)
        self.allow_short = allow_short

    def on_bar(self, t, row, history):
        close = history["Close"].to_numpy(dtype=float)
        if close.size < self.n_long + self.n_signal + 5:
            return 0.0

        # Fenêtre suffisante pour stabiliser les EMA
        m = min(close.size, 5 * self.n_long)
        window = close[-m:]

        # Série DIFF causale sur window
        diffs = np.empty_like(window, dtype=float)
        for i in range(window.size):
            sub = window[: i + 1]
            if sub.size < self.n_long:
                diffs[i] = 0.0
            else:
                diffs[i] = ema_last(sub, self.n_short) - ema_last(sub, self.n_long)

        diff_t = diffs[-1]
        signal_t = ema_last(diffs, self.n_signal)
        hist_t = diff_t - signal_t

        if hist_t > self.eps:
            return 1.0
        if hist_t < -self.eps:
            return -1.0 if self.allow_short else 0.0
        return 0.0
