"""
Strategies module: Trading strategy implementations.
"""
from .base import BaseStrategy
from .buy_and_hold import BuyAndHoldStrategy
from .ma_crossover import MACrossoverStrategy
from .ewmac import EWMACStrategy

__all__ = [
    "BaseStrategy",
    "BuyAndHoldStrategy",
    "MACrossoverStrategy",
    "EWMACStrategy"
]