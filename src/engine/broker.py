from dataclasses import dataclass

@dataclass
class TradeState:
    position: float = 0.0     # -1, 0, +1 (baseline) ou sizing continu
    entry_price: float | None = None
    realized_pnl: float = 0.0
    fees: float = 0.0
    num_trades: int = 0

class Broker:
    def __init__(self, bp_fee: float = 1e-4):
        self.bp_fee = bp_fee
        self.state = TradeState()

    def _fee(self, pos_abs: float, s_start: float, s_end: float) -> float:
        return pos_abs * self.bp_fee * (s_start + s_end)

    def mark_to_market(self, price: float) -> float:
        """Unrealized PnL at current price."""
        st = self.state
        if st.position == 0 or st.entry_price is None:
            return 0.0
        return st.position * (price - st.entry_price)

    def execute_target_position(self, target_pos: float, price: float, next_price: float | None = None):
        """
        Change position to target_pos at 'price'.
        Fees use (Sstart + Send). If next_price not available, reuse price.
        """
        st = self.state
        if next_price is None:
            next_price = price

        # no change
        if target_pos == st.position:
            return

        # If closing existing position, realize PnL
        if st.position != 0 and st.entry_price is not None:
            st.realized_pnl += st.position * (price - st.entry_price)

        # Fees paid on change of exposure
        st.fees += self._fee(abs(target_pos - st.position), price, next_price)

        # Update entry for new position
        st.position = target_pos
        st.entry_price = price if target_pos != 0 else None
        st.num_trades += 1
