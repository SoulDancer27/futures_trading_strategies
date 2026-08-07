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
from .capital_models import BaseCapitalModel, FixedCapitalModel  # <-- NEW IMPORTS


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
        slippage_rate: Optional[float] = None,             # Decimal rate (0.0005 = 0.05%)
        capital_model: Optional[BaseCapitalModel] = None,  # <-- NEW PARAMETER
        risk_free_rate: float = 0.0,
        trading_days: int = 252
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
            
        # Default to Fixed Capital Model (Carver's methodology) if none provided
        self.capital_model = capital_model if capital_model is not None else FixedCapitalModel()
        self.risk_free_rate = risk_free_rate
        self.trading_days = trading_days

    def run(self, strategy: BaseStrategy, data: pd.DataFrame, 
            strategy_name: str = "", ticker: str = "") -> BacktestResult:
        """
        Execute backtest. Engine handles position generation internally
        to guarantee data/strategy alignment and prevent lookahead bias.
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

        # 7. Calculate Leverage internally (Self-contained result!)
        notional = pos * data['close'] * self.point_value
        leverage = self.capital_model.calculate_leverage(notional, equity, self.initial_capital)
        
        # 8. Base performance metrics (Delegated to metrics.py and capital_model)
        metrics = calculate_metrics(
            equity=equity,
            daily_pnl=daily_pnl,
            capital_model=self.capital_model,  # <-- PASSES THE MODEL
            initial_capital=self.initial_capital,
            positions=pos,
            cumulative_fees=cumulative_fees,
            cumulative_turnover=cumulative_turnover,
            risk_free_rate=self.risk_free_rate,
            trading_days=self.trading_days
        )
        
        return BacktestResult(
            equity=equity,
            positions=pos,
            metrics=metrics,
            daily_pnl=daily_pnl,
            strategy_name=strategy_name or strategy.__class__.__name__,
            ticker=ticker,
            cumulative_fees=cumulative_fees,
            cumulative_turnover=cumulative_turnover,
            price_data=data['close'].copy(),
            point_value=self.point_value,
            leverage=leverage
        )