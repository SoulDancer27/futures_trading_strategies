import pandas as pd
from typing import Optional
from .models import BacktestResult
from .metrics import calculate_metrics
from ..strategies import BaseStrategy

class VectorizedEngine:
    """
    Vectorized backtesting engine for continuous position strategies.
    Handles PnL calculation, transaction costs, slippage, and metric aggregation.
    """
    def __init__(
        self,
        initial_capital: float = 100_000.0,
        point_value: float = 1.0,               # $ per point (1.0 for stocks/spot, 50 for ES, etc.)
        commission_per_contract: Optional[float] = None,  # Fixed fee per contract/share
        commission_pct: Optional[float] = None,           # % of trade value (0.001 = 0.1%)
        slippage_pct: Optional[float] = None              # % slippage of trade value
    ):
        self.initial_capital = initial_capital
        self.point_value = point_value
        
        # Validate commission modes (mutually exclusive)
        if commission_per_contract is not None and commission_pct is not None:
            raise ValueError("Use either commission_per_contract OR commission_pct, not both")
            
        self.commission_per_contract = commission_per_contract
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        
        # Default to zero fixed commission if neither specified
        if self.commission_per_contract is None and self.commission_pct is None:
            self.commission_per_contract = 0.0

    def run(self, strategy: BaseStrategy, data: pd.DataFrame, 
            strategy_name: str = "", ticker: str = "") -> BacktestResult:
        """
        Execute backtest. Engine handles position generation internally
        to guarantee data/strategy alignment and prevent lookahead bias.

        Args:
            strategy: Strategy instance
            data: Price data with 'close' column
            strategy_name: Optional override for strategy name
            ticker: Optional ticker/instrument name
        """
        # 1. Generate positions from strategy (guarantees matching index)
        positions = strategy.generate_positions(data)
        
        # 2. Align & shift positions (execute at next bar to prevent lookahead)
        pos = positions.reindex(data.index).fillna(0).shift(1).fillna(0)
        
        # 3. Raw PnL calculation
        price_change = data["close"].diff().fillna(0)
        raw_pnl = pos * price_change * self.point_value
        
        # 4. Transaction costs & slippage
        turnover = pos.diff().abs().fillna(0)  # Contracts traded each bar
        
        # Commission
        if self.commission_pct is not None:
            commission_cost = turnover * data["close"] * self.commission_pct * self.point_value
        else:
            commission_cost = turnover * (self.commission_per_contract or 0)
            
        # Slippage (modeled as % of trade value)
        slippage_cost = 0.0
        if self.slippage_pct is not None:
            slippage_cost = turnover * data["close"] * self.slippage_pct * self.point_value
            
        total_costs = commission_cost + slippage_cost
        
        # 5. Equity curve & metrics
        daily_pnl = raw_pnl - total_costs
        equity = self.initial_capital + daily_pnl.cumsum()
        metrics = calculate_metrics(equity, daily_pnl, pos, self.initial_capital)
        
        return BacktestResult(
            equity=equity,
            positions=pos,
            metrics=metrics,
            daily_pnl=daily_pnl,
            strategy_name=strategy_name or strategy.__class__.__name__,
            ticker=ticker
        )