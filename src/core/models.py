"""
Core Data Models.
Pure data containers for backtest and portfolio results.
"""
from dataclasses import dataclass
from typing import Dict, List
import pandas as pd
import numpy as np

from .asset import Asset 

@dataclass
class BacktestResult:
    """
    Immutable container for a single strategy backtest.
    All fields are mandatory to ensure strict contracts between Engine and Analyzer/Plotter.
    """
    equity: pd.Series                       # Daily portfolio value
    positions: pd.Series                    # Contracts/shares held
    daily_pnl: pd.Series                    # Day-over-day P&L
    metrics: Dict[str, float]               # Performance statistics
    
    strategy_name: str = ""                 # Human-readable strategy name
    asset: Asset                            # Stores ticker, price_data, point_value, etc.
    
    cumulative_fees: pd.Series              # Cumulative transaction costs (can be 0)
    cumulative_turnover: pd.Series          # Cumulative contracts traded (can be 0)
    leverage: pd.Series                     # jPortfolio leverage/exposure
    drawdown: pd.Series


@dataclass
class PortfolioResult:
    """
    Immutable container for a multi-asset portfolio analysis.
    """
    portfolio_equity: pd.Series
    portfolio_returns: pd.Series
    individual_returns: pd.DataFrame
    weights: np.ndarray
    strategy_names: List[str]
    metrics: Dict[str, float]
    correlation_matrix: pd.DataFrame
    individual_metrics: List[Dict[str, float]]
    trading_days: int = 252
    
    cumulative_fees: pd.Series              # MANDATORY
    cumulative_turnover: pd.Series          # MANDATORY
    portfolio_leverage: pd.Series           # MANDATORY