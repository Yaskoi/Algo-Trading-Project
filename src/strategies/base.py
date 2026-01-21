from __future__ import annotations
from abc import ABC, abstractmethod


class BaseStrategy(ABC):
    @abstractmethod
    def on_bar(self, ts, open_: float, high: float, low: float, close: float) -> float:
        """
        Return desired position AFTER observing bar close:
        -1 short, 0 flat, +1 long (or leverage allowed in vol targeting)
        """
        raise NotImplementedError
