"""
Core Data Models.
Pure data containers for execution and analysis.
"""
from dataclasses import dataclass
from typing import Optional, Dict
import pandas as pd
import numpy as np
from .asset import Asset 

@dataclass
class ExecutionResult:
    """
    Pure time-series data representing the day-by-day execution of a strategy.
    Contains NO aggregate metrics.
    """
    equity: pd.Series                       
    daily_pnl: pd.Series                    
    positions: pd.Series                    
    leverage: pd.Series                     
    drawdown: pd.Series                     
    returns: pd.Series                      
    realized_vol: pd.Series                 
    cumulative_fees: pd.Series              
    cumulative_turnover: pd.Series          

    asset: Asset 
    strategy_name: str = ""                                            
    risk_free_rate: float = 0.0             

@dataclass
class PerformanceMetrics:
    """
    Aggregate scalar statistics derived from an ExecutionResult.
    """
    total_return_pct: float
    gross_return_pct: float
    cagr_pct: float
    annual_volatility_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    avg_drawdown_pct: float
    win_rate_pct: float
    skew: float
    kurtosis: float
    lower_tail: float
    upper_tail: float
    tail_risk: float
    gross_pnl: float
    net_pnl: float
    total_fees_currency: float
    fee_drag_ratio: float
    cost_efficiency: float
    gross_sharpe_ratio: float
    sharpe_drag: float
    turnover_adjusted_sharpe: float
    avg_daily_turnover: float
    total_turnover: float
    total_fee_drag_pct: float
    annualized_fee_drag_pct: float
    num_years: float


@dataclass
class RegressionResult:
    """
    Single-factor benchmark regression (Jensen-style alpha/beta).
    Strategy monthly returns regressed on benchmark monthly returns.
    """
    strategy_name: str
    benchmark_name: str
    alpha_monthly_pct: float
    alpha_annualized_pct: float
    beta: float
    r_squared: float
    alpha_t_stat: float
    beta_t_stat: float
    n_observations: int