"""Standalone Vectorized Backtesting Framework."""

# --- Data ---
from .data.loader import load_simple_price_csv

# --- Engine ---
from .engine.vectorized import VectorizedEngine
from .engine.models import BacktestResult, PortfolioResult
from .engine.portfolio_analyzer import PortfolioAnalyzer

# --- Strategies ---
from .strategies.base import BaseStrategy
from .strategies.buy_and_hold import BuyAndHoldStrategy
from .strategies.fixed_risk_position import FixedRiskPositionStrategy
from .strategies.ma_crossover import MACrossoverStrategy
from .strategies.ewmac_forecast import EWMACForecastStrategy

# --- Plots / Visualization ---
from .plots.plotting import plot_results
from .plots.plot_backtest_results import plot_backtest_results
from .plots.analysis import compare_strategies, print_comparison_table

__all__ = [
    # Core Tools
    "load_simple_price_csv",
    "VectorizedEngine",
    "BacktestResult",
    "PortfolioResult",
    "PortfolioAnalyzer",
    
    # Strategies
    "BaseStrategy",
    "BuyAndHoldStrategy",
    "FixedRiskPositionStrategy",
    "VolatilityScaledBNH",
    "MACrossoverStrategy",
    "EWMACForecastStrategy",
    
    # Visualization
    "plot_results",
    "plot_backtest_results",
    "compare_strategies",
    "print_comparison_table"

]