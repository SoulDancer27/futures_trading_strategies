# src/strategies/normalised_trend.py
import pandas as pd
import numpy as np
from .base import BaseStrategy
from .transforms import normalise_price, ewmac_crossover


class NormalisedTrendStrategy(BaseStrategy):
    """
    Carver's 'normalised trend' (Strategy 17).

    Identical to the EWMAC rule, except the EWMAC crossover is computed on a
    volatility-normalised price rather than the raw price. Produces a continuous
    signal in [-2, 2] that is imperfectly correlated with the standard EWMAC.
    """

    def __init__(
        self,
        fast_span: int = 16,
        slow_span: int = 64,
        vol_window: int = 25,
        scale: float = 100.0,
        forecast_scalar: float = 293.779,
    ):
        super().__init__(name=f"Normalised Trend ({fast_span}/{slow_span})")

        if fast_span >= slow_span:
            raise ValueError("fast_span must be strictly less than slow_span")

        self.fast_span = fast_span
        self.slow_span = slow_span
        self.vol_window = vol_window
        self.scale = scale
        # Calibrated divisor: average |raw forecast| maps to a full signal.
        self.forecast_scalar = forecast_scalar

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close = data['close']

        # 1. Volatility-normalise the price.
        pn = normalise_price(close, self.vol_window, self.scale)

        # 2. EWMAC crossover on the normalised price (already in vol units).
        crossover = ewmac_crossover(pn, self.fast_span, self.slow_span)

        # 3. Scale the raw forecast by its (calibrated) scalar, then clip to [-2, 2].
        signal = (crossover / self.forecast_scalar).clip(lower=-2.0, upper=2.0)

        # 4. Warm-up: no signal until both the vol estimate and the slow EWMA exist.
        warmup = max(self.slow_span, self.vol_window * 2)
        signal.iloc[:warmup] = 0.0

        return signal.fillna(0.0)

    def get_parameters(self) -> dict:
        return {
            "fast_span": self.fast_span,
            "slow_span": self.slow_span,
            "vol_window": self.vol_window,
            "scale": self.scale,
            "forecast_scalar": self.forecast_scalar,
        }
