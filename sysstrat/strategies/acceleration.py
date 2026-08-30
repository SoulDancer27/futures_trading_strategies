# src/strategies/acceleration.py
import pandas as pd
from .base import BaseStrategy
from .transforms import ewmac_raw_forecast


class AccelerationStrategy(BaseStrategy):
    """
    Carver's acceleration rule: the N-day change in the EWMAC trend forecast.

    A fast EWMAC(N, 4N) forecast is computed, and the acceleration is its change
    over the last N days — positive while the trend is strengthening, negative
    while it is weakening. Produces a continuous signal in [-2, 2].
    """

    def __init__(self, fast_span: int = 16, forecast_scalar: float = 1.4379, vol_span: int = 25):
        super().__init__(name=f"Acceleration ({fast_span}/{4 * fast_span})")

        if fast_span < 2:
            raise ValueError("fast_span must be at least 2")

        self.fast_span = fast_span
        self.slow_span = 4 * fast_span
        self.vol_span = vol_span
        self.forecast_scalar = forecast_scalar

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close = data['close']

        # 1. Raw EWMAC trend forecast (continuous).
        raw_ewmac = ewmac_raw_forecast(close, self.fast_span, self.slow_span, self.vol_span)

        # 2. Acceleration = N-day change in the trend forecast.
        raw_accel = raw_ewmac - raw_ewmac.shift(self.fast_span)

        # 3. Scale by the (calibrated) scalar, then clip to [-2, 2].
        signal = (raw_accel / self.forecast_scalar).clip(lower=-2.0, upper=2.0)

        # 4. Warm-up: need the slow EWMA plus the N-day lag.
        warmup = self.slow_span + self.fast_span
        signal.iloc[:warmup] = 0.0

        return signal.fillna(0.0)

    def get_parameters(self) -> dict:
        return {
            "fast_span": self.fast_span,
            "slow_span": self.slow_span,
            "vol_span": self.vol_span,
            "forecast_scalar": self.forecast_scalar,
        }
