# src/strategies/mean_reversion.py
import pandas as pd
from .base import BaseStrategy
from .transforms import mean_reversion_forecast


class MeanReversionStrategy(BaseStrategy):
    """
    Carver's fast mean-reversion rule (with trend overlay and volatility multiplier).

    Goes long when price is below its short-term equilibrium (and the trend is up),
    short when it is above (and the trend is down), scaled down in high-volatility
    regimes. Produces a continuous signal in [-2, 2].
    """

    def __init__(
        self,
        forecast_scalar: float = 0.6425,
        equil_span: int = 5,
        vol_span: int = 25,
        trend_fast: int = 16,
        trend_slow: int = 64,
        quantile_window: int = 2560,
        quantile_smooth: int = 10,
    ):
        super().__init__(name="Mean Reversion")

        self.forecast_scalar = forecast_scalar
        self.equil_span = equil_span
        self.vol_span = vol_span
        self.trend_fast = trend_fast
        self.trend_slow = trend_slow
        self.quantile_window = quantile_window
        self.quantile_smooth = quantile_smooth

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close = data['close']

        raw = mean_reversion_forecast(
            close,
            equil_span=self.equil_span,
            vol_span=self.vol_span,
            trend_fast=self.trend_fast,
            trend_slow=self.trend_slow,
            quantile_window=self.quantile_window,
            quantile_smooth=self.quantile_smooth,
        )

        signal = (raw / self.forecast_scalar).clip(lower=-2.0, upper=2.0)

        # Warm-up: need the trend EWMAC and the volatility estimate.
        warmup = max(self.trend_slow, self.vol_span * 2)
        signal.iloc[:warmup] = 0.0

        return signal.fillna(0.0)

    def get_parameters(self) -> dict:
        return {
            "forecast_scalar": self.forecast_scalar,
            "equil_span": self.equil_span,
            "vol_span": self.vol_span,
            "trend_fast": self.trend_fast,
            "trend_slow": self.trend_slow,
            "quantile_window": self.quantile_window,
            "quantile_smooth": self.quantile_smooth,
        }
