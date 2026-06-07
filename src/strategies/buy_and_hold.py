"""
Buy & Hold strategy - baseline benchmark.
Enters a fixed position at the start and holds until the end.
"""
import pandas as pd
from .base import BaseStrategy

class BuyAndHoldStrategy(BaseStrategy):
    """Simple buy-and-hold baseline for performance comparison."""

    def __init__(self, position_size: float = 1.0):
        """
        Args:
            position_size: Number of contracts/shares to hold.
                           Positive = long, Negative = short.
        """
        self.position_size = position_size

    @property
    def name(self) -> str:
        return f"BuyAndHold_{abs(self.position_size)}{'_Short' if self.position_size < 0 else ''}"

    def generate_positions(self, data: pd.DataFrame) -> pd.Series:
        """Return constant position size across all time points."""
        return pd.Series(self.position_size, index=data.index, dtype=float)