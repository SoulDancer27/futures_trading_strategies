"""
Performance metrics calculation module.
Calculates standard, net-of-fees performance metrics.
Fee-adjusted metrics (gross return, fee drag, Sharpe drag, etc.)
are calculated separately in the VectorizedEngine for exact accuracy.
"""
import numpy as np
import pandas as pd
from typing import Dict

def calculate_tail_ratios(
    returns: pd.Series, 
    lower_p: float = 1.0, 
    upper_p: float = 99.0, 
    inner_lower_p: float = 30.0, 
    inner_upper_p: float = 70.0,
    gaussian_ratio: float = 4.43
) -> Dict[str, float]:
    """
    Calculate percentile-based tail ratios (fat-tail measure).
    
    For a Gaussian distribution, both ratios ≈ 4.43.
    Values > 1.0 indicate fatter tails than normal.
    """
    demeaned = returns - returns.mean()
    
    p1 = demeaned.quantile(lower_p / 100)
    p30 = demeaned.quantile(inner_lower_p / 100)
    p70 = demeaned.quantile(inner_upper_p / 100)
    p99 = demeaned.quantile(upper_p / 100)
    
    # Guard against division by zero in flat/low-volatility series
    if abs(p30) < 1e-10 or abs(p70) < 1e-10:
        return {"lower_tail": 0.0, "upper_tail": 0.0, "tail_risk": 0.0}
    
    lower_tail = (abs(p1) / abs(p30)) / gaussian_ratio
    upper_tail = (abs(p99) / abs(p70)) / gaussian_ratio
    tail_risk = np.sqrt(lower_tail * upper_tail)
    
    return {
        "lower_tail": round(lower_tail, 3),
        "upper_tail": round(upper_tail, 3),
        "tail_risk": round(tail_risk, 3)
    }

def calculate_average_drawdown(equity: pd.Series) -> float:
    """
    Calculate average drawdown across all distinct drawdown periods.
    
    A drawdown period starts when equity falls from a peak and ends
    when it recovers to a new peak. Returns the average of each period's
    maximum drawdown as a percentage.
    """
    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max
    
    in_drawdown = drawdown < 0
    period_id = (~in_drawdown).cumsum()
    
    max_drawdowns = drawdown[in_drawdown].groupby(period_id[in_drawdown]).min()
    
    return max_drawdowns.mean() * 100 if len(max_drawdowns) > 0 else 0.0

def calculate_metrics(
    equity: pd.Series,
    daily_pnl: pd.Series,
    positions: pd.Series,
    initial_capital: float,
    risk_free_rate: float = 0.0,
    trading_days: int = 252
) -> dict:
    """
    Calculate standard, net-of-fees performance metrics.
    Returns a dict ready to be merged with engine-calculated fee metrics.
    """
    returns = equity.pct_change().dropna()
    total_return_pct = ((equity.iloc[-1] / initial_capital) - 1) * 100
    n_years = len(equity) / trading_days
    cagr_pct = ((1 + total_return_pct/100) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0.0
    
    vol_pct = returns.std() * np.sqrt(trading_days) * 100
    sharpe_ratio = (cagr_pct/100 - risk_free_rate) / (vol_pct/100) if vol_pct > 0 else 0.0
    
    # Drawdown metrics
    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max
    max_drawdown_pct = drawdown.min() * 100
    avg_drawdown_pct = calculate_average_drawdown(equity)
    
    # Tail risk metrics (fat-tail ratios)
    tail_metrics = calculate_tail_ratios(returns)
    
    # Win rate
    win_rate_pct = (daily_pnl > 0).sum() / len(daily_pnl) * 100 if len(daily_pnl) > 0 else 0.0
    
    return {
        # Returns & Risk (all in %)
        'total_return_pct': total_return_pct,
        'cagr_pct': cagr_pct,
        'annual_volatility_pct': vol_pct,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown_pct': max_drawdown_pct,
        'avg_drawdown_pct': avg_drawdown_pct,
        'win_rate_pct': win_rate_pct,
        
        # Distribution & Tail Metrics (dimensionless ratios)
        **tail_metrics
    }