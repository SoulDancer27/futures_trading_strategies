# src/strategies/breakout.py
import pandas as pd
from .base import BaseStrategy
from .transforms import breakout_forecast


class BreakoutStrategy(BaseStrategy):
    """
    Carver's rolling breakout (Strategy 18), single-horizon version.

    The price's position within a rolling ``horizon``-day high/low channel,
    smoothed and scaled to a continuous signal in [-2, 2]. A new high drives a
    positive signal (long); a new low, a negative signal (short).
    """

    def __init__(self, horizon: int = 40, forecast_scalar: float = 11.665):
        super().__init__(name=f"Breakout ({horizon})")

        if horizon < 2:
            raise ValueError("horizon must be at least 2")

        self.horizon = horizon
        self.forecast_scalar = forecast_scalar

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close = data['close']

        raw = breakout_forecast(close, self.horizon)

        # Scale the (smoothed) raw forecast by its calibrated scalar, clip to [-2, 2].
        signal = (raw / self.forecast_scalar).clip(lower=-2.0, upper=2.0)

        # Warm-up: no signal until the full channel has formed.
        signal.iloc[:self.horizon] = 0.0

        return signal.fillna(0.0)

    def get_parameters(self) -> dict:
        return {
            "horizon": self.horizon,
            "forecast_scalar": self.forecast_scalar,
        }
