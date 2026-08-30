"""
Performance Analyzer.
Takes an ExecutionResult and calculates aggregate scalar metrics.
Completely stateless.
"""
import pandas as pd
import numpy as np
from typing import Dict

from ..core.models import ExecutionResult, PerformanceMetrics
from ..core.capital import Capital


class PerformanceAnalyzer:
    """
    Stateless analyzer. Delegates capital-model math to the injected Capital.
    """

    def __init__(self, capital: Capital):
        self.capital = capital

    def analyze(self, result: ExecutionResult) -> PerformanceMetrics:
        """Main entry point: calculates all scalar metrics from the ExecutionResult."""
        
        returns = result.returns
        daily_pnl = result.daily_pnl
        equity = result.equity
        drawdown = result.drawdown
        
        # Environment parameters: Capital is the single source of truth.
        capital_model = self.capital.capital_model
        initial_capital = self.capital.initial_capital
        risk_free_rate = self.capital.risk_free_rate
        trading_days = result.asset.trading_days
        # Years measured by calendar elapsed time (robust to how many rows
        # survived alignment), rather than row count / trading days.
        n_years = (equity.index[-1] - equity.index[0]).days / 365.25
        
        # --- Core Metrics (delegated to the capital model) ---
        total_return = capital_model.calculate_total_return(returns)
        cagr = capital_model.calculate_cagr(total_return, n_years)
        
        vol_decimal = returns.std() * np.sqrt(trading_days) 
        sharpe = (cagr - risk_free_rate) / (vol_decimal) if vol_decimal > 0 else 0.0
        
        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(trading_days) if len(downside_returns) > 0 else 0.0
        sortino = (cagr - risk_free_rate) / (downside_vol) if downside_vol > 0 else 0.0
        
        max_dd = drawdown.min()
        avg_dd = self._calculate_avg_drawdown(drawdown)
        
        # --- Distribution & Tail Risk ---
        win_rate = (daily_pnl > 0).sum() / len(daily_pnl) * 100 if len(daily_pnl) > 0 else 0.0
        skew_val = float(returns.skew()) if not pd.isna(returns.skew()) else 0.0
        kurtosis_val = float(returns.kurtosis()) if not pd.isna(returns.kurtosis()) else 0.0
        
        tail_metrics = self._calculate_tail_ratios(returns)
        
        # --- Fee & Turnover Analysis ---
        # Recover per-day costs from cumulative series. diff() leaves the first
        # element NaN; set it to the first cumulative value so day-0 costs/turnover
        # are never silently dropped.
        daily_fees = result.cumulative_fees.diff().fillna(result.cumulative_fees.iloc[0])
        daily_turnover = result.cumulative_turnover.diff().fillna(result.cumulative_turnover.iloc[0])
        
        raw_pnl = daily_pnl + daily_fees
        net_pnl = daily_pnl.sum()
        gross_pnl = raw_pnl.sum()
        total_fees = daily_fees.sum()
        
        fee_drag_ratio = total_fees / abs(gross_pnl) if gross_pnl != 0 else 0.0
        cost_efficiency = net_pnl / gross_pnl if gross_pnl != 0 else 1.0
        
        # Gross (pre-cost) metrics — delegated to the capital model, like net metrics.
        gross_returns = capital_model.calculate_returns(equity, raw_pnl, initial_capital)
        gross_return = capital_model.calculate_total_return(gross_returns)
        gross_cagr = capital_model.calculate_cagr(gross_return, n_years)

        gross_vol = gross_returns.std() * np.sqrt(trading_days)
        gross_sharpe = gross_cagr / gross_vol if gross_vol > 0 else 0.0
        
        sharpe_drag = gross_sharpe - sharpe
        avg_daily_turnover = daily_turnover.mean()  # currency / day
        # Turnover cost drag: annualized turnover (as a fraction of capital) times
        # the per-trade cost rate (commission + slippage). Unit-free, and only
        # meaningful now that turnover is measured in notional currency.
        cost_rate = (result.asset.commission_rate or 0.0) + (result.asset.slippage_rate or 0.0)
        annual_turnover_fraction = (daily_turnover.sum() / initial_capital / n_years) if n_years > 0 else 0.0
        turnover_penalty = annual_turnover_fraction * cost_rate
        turnover_adjusted_sharpe = max(0.0, sharpe - turnover_penalty)
        
        # --- Assemble PerformanceMetrics ---
        return PerformanceMetrics(
            total_return_pct=total_return * 100,
            cagr_pct=cagr * 100,
            annual_volatility_pct=vol_decimal*100,
            gross_return_pct=gross_return*100,
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
            annualized_fee_drag_pct=((total_fees / initial_capital) * 100 / n_years) if n_years > 0 else 0.0,
            num_years=n_years,
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