"""
Performance Analyzer.
Takes an ExecutionResult and calculates aggregate scalar metrics.
Completely stateless.
"""
import pandas as pd
import numpy as np
from typing import Dict

from ..core.models import ExecutionResult, PerformanceMetrics


class PerformanceAnalyzer:
    """
    Stateless analyzer. 
    """
    
    def analyze(self, result: ExecutionResult) -> PerformanceMetrics:
        """Main entry point: calculates all scalar metrics from the ExecutionResult."""
        
        returns = result.returns
        daily_pnl = result.daily_pnl
        equity = result.equity
        drawdown = result.drawdown
        
        # Extract environment parameters directly from the result
        risk_free_rate = result.risk_free_rate
        trading_days = result.asset.trading_days
        n_years = len(equity) / trading_days
        
        # --- Core Metrics ---
        total_return = returns.sum()
        cagr = total_return / n_years if n_years > 0 else 0.0
        
        vol_pct = returns.std() * np.sqrt(trading_days) * 100
        sharpe = (cagr - risk_free_rate) / (vol_pct / 100) if vol_pct > 0 else 0.0
        
        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(trading_days) if len(downside_returns) > 0 else 0.0
        sortino = (cagr - risk_free_rate) / (downside_vol / 100) if downside_vol > 0 else 0.0
        
        max_dd = drawdown.min()
        avg_dd = self._calculate_avg_drawdown(drawdown)
        
        # --- Distribution & Tail Risk ---
        win_rate = (daily_pnl > 0).sum() / len(daily_pnl) * 100 if len(daily_pnl) > 0 else 0.0
        skew_val = float(returns.skew()) if not pd.isna(returns.skew()) else 0.0
        kurtosis_val = float(returns.kurtosis()) if not pd.isna(returns.kurtosis()) else 0.0
        
        tail_metrics = self._calculate_tail_ratios(returns)
        
        # --- Fee & Turnover Analysis ---
        daily_fees = result.cumulative_fees.diff().fillna(0)
        daily_turnover = result.cumulative_turnover.diff().fillna(0)
        
        raw_pnl = daily_pnl + daily_fees
        net_pnl = daily_pnl.sum()
        gross_pnl = raw_pnl.sum()
        total_fees = daily_fees.sum()
        
        fee_drag_ratio = total_fees / abs(gross_pnl) if gross_pnl != 0 else 0.0
        cost_efficiency = net_pnl / gross_pnl if gross_pnl != 0 else 1.0
        
        initial_capital = equity.iloc[0]
        gross_return = gross_pnl / initial_capital
        gross_cagr = gross_return / n_years if n_years > 0 else 0.0
        
        equity_prev = equity.shift(1).fillna(initial_capital)
        gross_returns = raw_pnl / equity_prev
        gross_vol = gross_returns.std() * np.sqrt(trading_days) * 100
        gross_sharpe = gross_cagr / gross_vol if gross_vol > 0 else 0.0
        
        sharpe_drag = gross_sharpe - sharpe
        avg_daily_turnover = daily_turnover.mean()
        turnover_penalty = avg_daily_turnover * 0.1
        turnover_adjusted_sharpe = max(0.0, sharpe - turnover_penalty)
        
        # --- Assemble PerformanceMetrics ---
        return PerformanceMetrics(
            total_return_pct=total_return * 100,
            cagr_pct=cagr * 100,
            annual_volatility_pct=vol_pct,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown_pct=max_dd,
            avg_drawdown_pct=avg_dd,
            win_rate_pct=win_rate,
            skew=skew_val,
            kurtosis=kurtosis_val,
            lower_tail=tail_metrics['lower_tail'],
            upper_tail=tail_metrics['upper_tail'],
            tail_risk=tail_metrics['tail_risk'],
            gross_pnl=gross_pnl,
            net_pnl=net_pnl,
            total_fees_currency=total_fees,
            fee_drag_ratio=fee_drag_ratio,
            cost_efficiency=cost_efficiency,
            gross_sharpe_ratio=gross_sharpe,
            sharpe_drag=sharpe_drag,
            turnover_adjusted_sharpe=turnover_adjusted_sharpe,
            avg_daily_turnover=avg_daily_turnover,
            total_turnover=daily_turnover.sum(),
            total_fee_drag_pct=(total_fees / initial_capital) * 100,
            annualized_fee_drag_pct=((total_fees / initial_capital) * 100 / n_years) if n_years > 0 else 0.0
        )

    def _calculate_avg_drawdown(self, drawdown_series: pd.Series) -> float:
        in_drawdown = drawdown_series < 0
        if not in_drawdown.any(): return 0.0
        period_id = (~in_drawdown).cumsum()
        max_drawdowns = drawdown_series[in_drawdown].groupby(period_id[in_drawdown]).min()
        return max_drawdowns.mean()

    def _calculate_tail_ratios(self, returns: pd.Series) -> Dict[str, float]:
        if returns.empty:
            return {"lower_tail": 0.0, "upper_tail": 0.0, "tail_risk": 0.0}

        demeaned = returns - returns.mean()
        p1 = demeaned.quantile(0.01)
        p30 = demeaned.quantile(0.30)
        p70 = demeaned.quantile(0.70)
        p99 = demeaned.quantile(0.99)
        
        if abs(p30) < 1e-10 or abs(p70) < 1e-10:
            return {"lower_tail": 0.0, "upper_tail": 0.0, "tail_risk": 0.0}
        
        gaussian_ratio = 4.43
        lower_tail = (abs(p1) / abs(p30)) / gaussian_ratio
        upper_tail = (abs(p99) / abs(p70)) / gaussian_ratio
        tail_risk = np.sqrt(lower_tail * upper_tail)
        
        return {
            "lower_tail": round(lower_tail, 3),
            "upper_tail": round(upper_tail, 3),
            "tail_risk": round(tail_risk, 3)
        }