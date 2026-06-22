"""
Vectorized Backtesting Engine.
Fast, strategy-agnostic engine using pandas/numpy for continuous position strategies.
Handles P&L calculation, transaction costs, slippage, and comprehensive metric aggregation.
"""
import pandas as pd
import numpy as np
from typing import Optional
from ..strategies.base import BaseStrategy
from .models import BacktestResult
from .metrics import calculate_metrics


class VectorizedEngine:
    """
    Vectorized backtesting engine for continuous position strategies.
    Handles P&L calculation, transaction costs, slippage, and metric aggregation.
    """
    def __init__(
        self,
        initial_capital: float = 100_000.0,
        point_value: float = 1.0,               # $ per point (1.0 for stocks/spot, 50 for ES, etc.)
        commission_per_contract: Optional[float] = None,  # Fixed fee per contract/share
        commission_rate: Optional[float] = None,           # Decimal rate (0.001 = 0.1%)
        slippage_rate: Optional[float] = None              # Decimal rate (0.0005 = 0.05%)
    ):
        self.initial_capital = initial_capital
        self.point_value = point_value
        
        # Validate commission modes (mutually exclusive)
        if commission_per_contract is not None and commission_rate is not None:
            raise ValueError("Specify either commission_per_contract OR commission_rate, not both")
            
        self.commission_per_contract = commission_per_contract
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        
        # Default to zero fixed commission if neither specified
        if self.commission_per_contract is None and self.commission_rate is None:
            self.commission_per_contract = 0.0

    def run(self, strategy: BaseStrategy, data: pd.DataFrame, 
            strategy_name: str = "", ticker: str = "") -> BacktestResult:
        """
        Execute backtest. Engine handles position generation internally
        to guarantee data/strategy alignment and prevent lookahead bias.

        Args:
            strategy: Strategy instance implementing generate_positions()
            data: Price data with 'close' column
            strategy_name: Optional override for strategy name
            ticker: Optional ticker/instrument name
        """
        # 1. Generate positions from strategy (guarantees matching index)
        positions = strategy.generate_positions(data)
        
        # 2. Align & shift positions (execute at next bar to prevent lookahead)
        pos = positions.reindex(data.index).fillna(0).shift(1).fillna(0)
        
        # 3. Raw PnL calculation (before costs)
        price_change = data["close"].diff().fillna(0)
        raw_pnl = pos * price_change * self.point_value
        
        # 4. Transaction costs & slippage
        turnover = pos.diff().abs().fillna(0)  # Contracts traded each bar
        
        # Commission
        if self.commission_rate is not None:
            commission_cost = turnover * data["close"] * self.commission_rate * self.point_value
        else:
            commission_cost = turnover * (self.commission_per_contract or 0.0)
            
        # Slippage
        slippage_cost = 0.0
        if self.slippage_rate is not None:
            slippage_cost = turnover * data["close"] * self.slippage_rate * self.point_value
            
        total_costs = commission_cost + slippage_cost
        
        # 5. Equity & PnL
        daily_pnl = raw_pnl - total_costs
        equity = self.initial_capital + daily_pnl.cumsum()
        
        # 6. Cumulative series for visualization & storage
        cumulative_fees = total_costs.cumsum()
        cumulative_turnover = turnover.cumsum()
        
        # 7. Base performance metrics (return, vol, sharpe, drawdown, etc.)
        metrics = calculate_metrics(equity, daily_pnl, pos, self.initial_capital)
        
        # 🔑 8. Fee-Adjusted Metrics (Calculated EXACTLY in engine, no estimation)
        net_pnl = daily_pnl.sum()
        gross_pnl = raw_pnl.sum()
        total_fees = total_costs.sum()
        n_years = len(equity) / 252.0
        
        # Cost efficiency ratios
        fee_drag_ratio = total_fees / abs(gross_pnl) if gross_pnl != 0 else 0.0
        cost_efficiency = net_pnl / gross_pnl if gross_pnl != 0 else 1.0
        
        # Gross vs Net Sharpe & Drag
        gross_return = gross_pnl / self.initial_capital
        gross_cagr = (1 + gross_return) ** (1 / n_years) - 1 if n_years > 0 else 0.0
        
        # Exact gross returns: raw P&L / previous equity
        equity_prev = equity.shift(1).fillna(self.initial_capital)
        gross_returns = raw_pnl / equity_prev
        gross_vol = gross_returns.std() * np.sqrt(252)
        gross_sharpe = gross_cagr / gross_vol if gross_vol > 0 else 0.0
        
        net_sharpe = metrics.get('sharpe_ratio', 0.0)
        sharpe_drag = gross_sharpe - net_sharpe
        
        # Turnover-adjusted Sharpe (penalizes excessive churn)
        avg_daily_turnover = turnover.mean()
        turnover_penalty = avg_daily_turnover * 0.1  # Adjustable sensitivity
        turnover_adjusted_sharpe = max(0.0, net_sharpe - turnover_penalty)
        
        # Merge fee/turnover metrics into the main metrics dictionary
        metrics.update({
            'gross_pnl': gross_pnl,
            'gross_return': gross_return,
            'gross_return_pct': gross_return * 100,
            'net_pnl': net_pnl,
            'total_fees_currency': total_fees,
            'fee_drag_ratio': fee_drag_ratio,
            'cost_efficiency': cost_efficiency,
            'gross_sharpe_ratio': gross_sharpe,
            'sharpe_drag': sharpe_drag,
            'turnover_adjusted_sharpe': turnover_adjusted_sharpe,
            'avg_daily_turnover': avg_daily_turnover,
            'total_turnover': turnover.sum(),
            'total_fee_drag_pct': (total_fees / self.initial_capital) * 100,
            'annualized_fee_drag_pct': ((total_fees / self.initial_capital) * 100 / n_years) if n_years > 0 else 0.0
        })
        
        return BacktestResult(
            equity=equity,
            positions=pos,
            metrics=metrics,
            daily_pnl=daily_pnl,
            strategy_name=strategy_name or strategy.__class__.__name__,
            ticker=ticker,
            cumulative_fees=cumulative_fees,
            cumulative_turnover=cumulative_turnover
        )