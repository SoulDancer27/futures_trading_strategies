"""
Performance & risk metric calculations for vectorized backtests.
Implements return/risk statistics, tail-ratio analysis, and drawdown period averaging.
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
    
    Args:
        returns: Return series (will be demeaned internally)
        lower_p: Lower extreme percentile (default: 1st)
        upper_p: Upper extreme percentile (default: 99th)
        inner_lower_p: Lower "normal" percentile (default: 30th ≈ -1σ)
        inner_upper_p: Upper "normal" percentile (default: 70th ≈ +1σ)
        gaussian_ratio: Expected ratio for normal distribution (≈4.43)
    
    Returns:
        Dict with lower_tail, upper_tail, and combined tail_risk metrics
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
    initial_capital: float
) -> Dict[str, float]:
    """
    Compute standard backtest performance & risk metrics.
    
    Args:
        equity: Daily portfolio value series
        daily_pnl: Day-over-day P&L series
        positions: Contract/share positions series
        initial_capital: Starting portfolio value
        
    Returns:
        Dictionary of rounded metrics ready for BacktestResult
    """
    returns = equity.pct_change().dropna()
    
    total_return = (equity.iloc[-1] / initial_capital - 1) * 100
    cagr = (1 + total_return / 100) ** (252 / len(returns)) - 1
    volatility = returns.std() * np.sqrt(252)
    sharpe = cagr / volatility if volatility > 0 else 0.0
    
    rolling_max = equity.cummax()
    max_dd = ((equity - rolling_max) / rolling_max).min() * 100
    avg_dd = calculate_average_drawdown(equity)
    
    # Tail ratios require sufficient data for stable percentile estimates
    tail_metrics = calculate_tail_ratios(returns) if len(returns) >= 100 else {
        "lower_tail": np.nan, "upper_tail": np.nan, "tail_risk": np.nan
    }
    
    return {
        "total_return_pct": round(total_return, 2),
        "cagr_pct": round(cagr * 100, 2),
        "annual_volatility_pct": round(volatility * 100, 2),
        "sharpe_ratio": round(sharpe, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "avg_drawdown_pct": round(avg_dd, 2),
        "final_equity": round(equity.iloc[-1], 2),
        "skew": round(returns.skew(), 3),
        **tail_metrics
    }

__all__ = ["calculate_metrics", "calculate_tail_ratios", "calculate_average_drawdown"]