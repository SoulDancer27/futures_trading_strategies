"""
Strategies module: Trading strategy implementations.
"""
from .base import BaseStrategy
from .buy_and_hold import BuyAndHoldStrategy
from .ma_crossover import MACrossoverStrategy
from .ewmac import EWMACStrategy
from .normalised_trend import NormalisedTrendStrategy
from .breakout import BreakoutStrategy
from .acceleration import AccelerationStrategy
from .skew import SkewStrategy
from .mean_reversion import MeanReversionStrategy
from .combined_forecast import CombinedForecastStrategy, forecast_diversification_multiplier

__all__ = [
    "BaseStrategy",
    "BuyAndHoldStrategy",
    "MACrossoverStrategy",
    "EWMACStrategy",
    "NormalisedTrendStrategy",
    "BreakoutStrategy",
    "AccelerationStrategy",
    "SkewStrategy",
    "MeanReversionStrategy",
    "CombinedForecastStrategy",
    "forecast_diversification_multiplier"
]