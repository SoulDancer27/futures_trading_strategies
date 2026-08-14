import pandas as pd
import numpy as np
from typing import Optional
from ..core.capital import Capital
from ..core.asset import Asset
from ..core.sizers import BasePositionSizer, FixedFractionSizer
from ..core.models import ExecutionResult
from ..strategies.base import BaseStrategy

class VectorizedEngine:
    def __init__(self, capital: Capital, position_sizer: Optional[BasePositionSizer] = None):
        self.capital = capital
        self.position_sizer = position_sizer or FixedFractionSizer(max_allocation=1.0)

    def run(self, strategy: BaseStrategy, asset: Asset) -> ExecutionResult:
        data = asset.price_data.to_frame(name='close')
        raw_signal = strategy.generate_signals(data)
        pos = self.position_sizer.calculate_position(raw_signal, self.capital, asset)
        pos = pos.shift(1).fillna(0)
        
        price_change = asset.price_data.diff().fillna(0)
        raw_pnl = pos * price_change * asset.point_value
        turnover = pos.diff().abs().fillna(0)
        
        # Trades that establish today's position execute at the PREVIOUS close
        # (pos is already shifted by 1). Charge costs at that price, not today's
        # close, to avoid a one-bar look-ahead in cost accounting.
        prev_price = asset.price_data.shift(1)
        commission_cost = turnover * prev_price * (asset.commission_rate or 0.0) * asset.point_value if asset.commission_rate else turnover * (asset.commission_per_contract or 0.0)
        slippage_cost = turnover * prev_price * (asset.slippage_rate or 0.0) * asset.point_value if asset.slippage_rate else 0.0
        total_costs = (commission_cost + slippage_cost).fillna(0)
        daily_pnl = raw_pnl - total_costs
        equity = self.capital.initial_capital + daily_pnl.cumsum()
        
        notional = pos * asset.price_data * asset.point_value
        leverage = self.capital.capital_model.calculate_leverage(notional, equity, self.capital.initial_capital)
        returns = self.capital.capital_model.calculate_returns(equity, daily_pnl, self.capital.initial_capital)
        drawdown = self.capital.capital_model.calculate_drawdown(equity, self.capital.initial_capital) * 100
        realized_vol = returns.rolling(window=21, min_periods=1).std() * np.sqrt(asset.trading_days) * 100
        
        return ExecutionResult(
            equity=equity, daily_pnl=daily_pnl, positions=pos, leverage=leverage,
            drawdown=drawdown, returns=returns, realized_vol=realized_vol,
            cumulative_fees=total_costs.cumsum(), cumulative_turnover=turnover.cumsum(),
            strategy_name=strategy.name, asset=asset, risk_free_rate=self.capital.risk_free_rate
        )