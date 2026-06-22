"""Standalone Vectorized Backtesting Framework."""

# --- Data ---
from .data.loader import load_simple_price_csv

# --- Engine ---
from .engine.vectorized import VectorizedEngine, BacktestResult

# --- Strategies ---
from .strategies.base import BaseStrategy
from .strategies.buy_and_hold import BuyAndHoldStrategy
from .strategies.fixed_risk_position import FixedRiskPositionStrategy
from .strategies.vol_scaled_bnh import VolatilityScaledBNH

# --- Plots / Visualization ---
from .plots.plotting import plot_results
from .plots.plot_backtest_results import plot_backtest_results
from .plots.analysis import compare_strategies, print_comparison_table

__all__ = [
    # Core Tools
    "load_simple_price_csv",
    "VectorizedEngine",
    "BacktestResult",
    
    # Strategies
    "BaseStrategy",
    "BuyAndHoldStrategy",
    "FixedRiskPositionStrategy",
    "VolatilityScaledBNH",
    
    # Visualization
    "plot_results",
    "plot_backtest_results",
    "compare_strategies",
    "print_comparison_table"

]