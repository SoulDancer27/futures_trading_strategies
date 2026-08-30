# src/strategies/ewmac.py
import pandas as pd
from .base import BaseStrategy
from .transforms import ewmac_raw_forecast

class EWMACStrategy(BaseStrategy):
    """
    EWMAC Trend Strength Strategy.
    Outputs a continuous signal in [-2, 2] based on trend strength (Carver's
    forecast, scaled to average |.| = 1 and clipped at ±2). Stronger trends =
    larger signal = larger position (up to the Sizer's limits).
    """
    
    def __init__(
        self, 
        fast_window: int = 16, 
        slow_window: int = 64,
        vol_span: int = 25,
        forecast_scalar: float = 2.467
    ):
        super().__init__(name=f"EWMAC ({fast_window}/{slow_window})")
        
        if fast_window >= slow_window:
            raise ValueError("fast_window must be strictly less than slow_window")
            
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.vol_span = vol_span
        self.forecast_scalar = forecast_scalar

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close = data['close']

        # 1. Raw forecast: crossover expressed in daily-volatility units.
        raw_forecast = ewmac_raw_forecast(close, self.fast_window, self.slow_window, self.vol_span)

        # 2. Scale by the (calibrated) scalar, then clip to [-2, 2].
        # The scalar is calibrated (on the clipped signal) so the average |signal|
        # is ~1, which is the "full" volatility-targeted position.
        signal = (raw_forecast / self.forecast_scalar).clip(lower=-2.0, upper=2.0)

        # 3. Warm-up period
        min_warmup = max(self.slow_window, self.vol_span * 2)
        signal.iloc[:min_warmup] = 0.0

        return signal.fillna(0.0)

    def get_parameters(self) -> dict:
        return {
            "fast_window": self.fast_window,
            "slow_window": self.slow_window,
            "vol_span": self.vol_span,
            "forecast_scalar": self.forecast_scalar,
        }