from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class FeeModel:
    bp: float = 0.0001  # basis-point


class RoundTripFeeTracker:
    """
    Ta consigne: total_fees += |position| * bp * (Sstart + Send)
    On l'applique lors de la FERMETURE d'une position (round-trip).
    Pour un flip, on ferme puis on ouvre.
    """
    def __init__(self, fee_model: FeeModel):
        self.fee_model = fee_model
        self.total_fees = 0.0

    def charge_round_trip(self, position_abs: float, entry_price: float, exit_price: float) -> float:
        fee = position_abs * self.fee_model.bp * (entry_price + exit_price)
        self.total_fees += fee
        return fee
