"""Standalone Vectorized Backtesting Framework."""
from .data.loader import load_simple_price_csv
from .engine.vectorized import VectorizedEngine, BacktestResult
from .strategies.base import BaseStrategy
from .strategies.buy_and_hold import BuyAndHoldStrategy
from .plots.mpl_plots import plot_results

__all__ = [
    "load_simple_price_csv",
    "VectorizedEngine",
    "BacktestResult",
    "BaseStrategy",
    "MomentumStrategy",
    "BuyAndHoldStrategy",
    "plot_results",
]