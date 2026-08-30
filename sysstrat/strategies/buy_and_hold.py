# src/strategies/buy_and_hold.py
import pandas as pd
from .base import BaseStrategy

class BuyAndHoldStrategy(BaseStrategy):
    """
    A baseline strategy that outputs a constant signal of 1.0 
    throughout the entire backtest period.
    """
    
    def __init__(self):
        super().__init__(name="Buy and Hold")

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        # Return a Series of 1.0s matching the data's index
        return pd.Series(1.0, index=data.index, name='signal')

    def get_parameters(self) -> dict:
        return {"constant_signal": 1.0}