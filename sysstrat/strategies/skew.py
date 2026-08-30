# src/strategies/skew.py
import pandas as pd
from .base import BaseStrategy
from .transforms import skew_forecast


class SkewStrategy(BaseStrategy):
    """
    Carver's skew targeting rule.

    Goes long instruments with negative return skew and short instruments with
    positive skew (the "negative-skew premium": crash-prone assets earn a return
    premium, lottery-like assets underperform). The forecast is the negative rolling
    skew of daily returns, smoothed, and scaled to a signal in [-2, 2].
    """

    def __init__(self, window: int = 60, forecast_scalar: float = 0.4095):
        super().__init__(name=f"Skew ({window})")

        if window < 3:
            raise ValueError("window must be at least 3 (skew needs >= 3 observations)")

        self.window = window
        self.forecast_scalar = forecast_scalar

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close = data['close']

        raw = skew_forecast(close, self.window)

        # Scale by the (calibrated) scalar, then clip to [-2, 2].
        signal = (raw / self.forecast_scalar).clip(lower=-2.0, upper=2.0)

        # Warm-up: no signal until the full skew window has formed.
        signal.iloc[:self.window] = 0.0

        return signal.fillna(0.0)

    def get_parameters(self) -> dict:
        return {
            "window": self.window,
            "forecast_scalar": self.forecast_scalar,
        }
