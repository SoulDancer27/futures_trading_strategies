"""
Performance Metrics Calculator.
Single source of truth for all analytical math.
Delegates capital-specific calculations to the injected CapitalModel.
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict
from ..core.capital import Capital


# ==========================================
# 1. OUTPUT CONTAINER
# ==========================================
@dataclass
class MetricsOutput:
    """Container for all calculated analytical series and scalar metrics."""
    metrics: Dict[str, float]
    returns: pd.Series
    drawdown: pd.Series
    realized_vol: pd.Series


# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def calculate_tail_ratios(
    returns: pd.Series, 
    lower_p: float = 1.0, 
    upper_p: float = 99.0, 
    inner_lower_p: float = 30.0, 
    inner_upper_p: float = 70.0,
    gaussian_ratio: float = 4.43
) -> Dict[str, float]:
    """Calculate percentile-based tail ratios (fat-tail measure)."""
    if returns.empty:
        return {"lower_tail": 0.0, "upper_tail": 0.0, "tail_risk": 0.0}

    demeaned = returns - returns.mean()
    
    p1 = demeaned.quantile(lower_p / 100)
    p30 = demeaned.quantile(inner_lower_p / 100)
    p70 = demeaned.quantile(inner_upper_p / 100)
    p99 = demeaned.quantile(upper_p / 100)
    
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


def calculate_average_drawdown(drawdown_series: pd.Series) -> float:
    """Calculate average drawdown across all distinct drawdown periods."""
    if drawdown_series.empty:
        return 0.0
        
    in_drawdown = drawdown_series < 0
    if not in_drawdown.any():
        return 0.0
        
    period_id = (~in_drawdown).cumsum()
    max_drawdowns = drawdown_series[in_drawdown].groupby(period_id[in_drawdown]).min()
    
    return max_drawdowns.mean() if len(max_drawdowns) > 0 else 0.0


# ==========================================
# 3. MAIN CALCULATOR
# ==========================================
def calculate_metrics(
    equity: pd.Series,
    daily_pnl: pd.Series,
    capital: Capital,
    cumulative_fees: pd.Series,
    cumulative_turnover: pd.Series,
    trading_days: int = 252
) -> MetricsOutput:
    """
    Unified metric calculation.
    Returns both the scalar metrics dictionary and the pandas Series needed for plotting.
    """
    n_years = len(equity) / trading_days if len(equity) > 0 else 0.0
    
    # Extract from Capital object
    initial_capital = capital.initial_capital
    risk_free_rate = capital.risk_free_rate
    capital_model = capital.capital_model
    
    # --- 1. Calculate Series (Delegated to Capital Model) ---
    returns = capital_model.calculate_returns(equity, daily_pnl, initial_capital)
    drawdown = capital_model.calculate_drawdown(equity, initial_capital) * 100  # Convert to percentage
    
    # Realized Volatility (21-day rolling window, annualized, in %)
    realized_vol = returns.rolling(window=21, min_periods=1).std() * np.sqrt(trading_days) * 100
    
    # --- 2. Calculate Scalar Metrics ---
    total_return = capital_model.calculate_total_return(returns)
    cagr = capital_model.calculate_cagr(total_return, n_years)
    
    vol_pct = returns.std() * np.sqrt(trading_days) * 100 if not returns.empty else 0.0
    sharpe_ratio = (cagr - risk_free_rate) / (vol_pct / 100) if vol_pct > 0 else 0.0
    
    # Sortino Ratio
    downside_returns = returns[returns < 0]
    downside_vol = downside_returns.std() * np.sqrt(trading_days) if len(downside_returns) > 0 else 0.0
    sortino_ratio = (cagr - risk_free_rate) / (downside_vol / 100) if downside_vol > 0 else 0.0
    
    max_drawdown_pct = drawdown.min() if not drawdown.empty else 0.0
    avg_drawdown_pct = calculate_average_drawdown(drawdown)
    
    # Tail Risk & Win Rate
    tail_metrics = calculate_tail_ratios(returns)
    win_rate_pct = (daily_pnl > 0).sum() / len(daily_pnl) * 100 if len(daily_pnl) > 0 else 0.0
    
    # Skew and Kurtosis
    skew_val = float(returns.skew()) if not pd.isna(returns.skew()) else 0.0
    kurtosis_val = float(returns.kurtosis()) if not pd.isna(returns.kurtosis()) else 0.0
    
    # --- 3. Fee & Turnover Metrics ---
    daily_fees = cumulative_fees.diff().fillna(0)
    daily_turnover = cumulative_turnover.diff().fillna(0)
    
    raw_pnl = daily_pnl + daily_fees
    net_pnl = daily_pnl.sum()
    gross_pnl = raw_pnl.sum()
    total_fees = daily_fees.sum()
    
    fee_drag_ratio = total_fees / abs(gross_pnl) if gross_pnl != 0 else 0.0
    cost_efficiency = net_pnl / gross_pnl if gross_pnl != 0 else 1.0
    
    gross_return = gross_pnl / initial_capital
    gross_cagr = (1 + gross_return) ** (1 / n_years) - 1 if n_years > 0 and gross_return > -1 else 0.0
    
    equity_prev = equity.shift(1).fillna(initial_capital)
    gross_returns = raw_pnl / equity_prev
    gross_vol = gross_returns.std() * np.sqrt(trading_days) if not gross_returns.empty else 0.0
    gross_sharpe = gross_cagr / (gross_vol / 100) if gross_vol > 0 else 0.0
    
    net_sharpe = sharpe_ratio
    sharpe_drag = gross_sharpe - net_sharpe
    
    avg_daily_turnover = daily_turnover.mean() if not daily_turnover.empty else 0.0
    turnover_penalty = avg_daily_turnover * 0.1
    turnover_adjusted_sharpe = max(0.0, net_sharpe - turnover_penalty)
    
    # --- 4. Assemble Dictionary ---
    metrics = {
        'total_return_pct': total_return * 100,
        'cagr_pct': cagr * 100,
        'annual_volatility_pct': vol_pct,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown_pct': max_drawdown_pct,
        'avg_drawdown_pct': avg_drawdown_pct,
        'win_rate_pct': win_rate_pct,
        'skew': skew_val,
        'kurtosis': kurtosis_val,
        **tail_metrics,
        'gross_pnl': gross_pnl,
        'net_pnl': net_pnl,
        'total_fees_currency': total_fees,
        'fee_drag_ratio': fee_drag_ratio,
        'cost_efficiency': cost_efficiency,
        'gross_sharpe_ratio': gross_sharpe,
        'sharpe_drag': sharpe_drag,
        'turnover_adjusted_sharpe': turnover_adjusted_sharpe,
        'avg_daily_turnover': avg_daily_turnover,
        'total_turnover': daily_turnover.sum(),
        'total_fee_drag_pct': (total_fees / initial_capital) * 100,
        'annualized_fee_drag_pct': ((total_fees / initial_capital) * 100 / n_years) if n_years > 0 else 0.0
    }
        
    # --- 5. Return Unified Output ---
    return MetricsOutput(
        metrics=metrics,
        returns=returns,
        drawdown=drawdown,
        realized_vol=realized_vol
    )