"""
Vectorized Backtesting Engine.
Fast, strategy-agnostic engine using pandas/numpy.
Handles execution, transaction costs, and equity curve generation.
"""
import numpy as np
from ..core.asset import Capital, Asset
from ..core.models import BacktestResult
from ..strategies.base import BaseStrategy
from .metrics import calculate_metrics  # We will build this next


class VectorizedEngine:
    """
    Executes backtests using vectorized operations.
    It knows nothing about market signals; it only knows about execution and accounting.
    """
    
    def __init__(self, capital: Capital):
        """
        Initialize the engine with the account configuration.
        """
        self.capital = capital

    def run(self, strategy: BaseStrategy, asset: Asset) -> BacktestResult:
        """
        Execute a backtest for a given strategy and asset.
        """
        # 1. Get RAW signal from strategy (e.g., -1.0 to 1.0)
        raw_signal = strategy.generate_signals(asset.price_data)
        
        # 2. SIZE the position using Capital rules
        pos = self.capital.position_sizer.calculate_position(raw_signal, self.capital, asset)
        
        # 3. Shift for execution (prevent lookahead)
        pos = pos.shift(1).fillna(0)
        
        # 4. Raw PnL calculation (before costs)
        price_change = asset.price_data.diff().fillna(0)
        raw_pnl = pos * price_change * asset.point_value
        
        # 5. Transaction costs & slippage
        turnover = pos.diff().abs().fillna(0)  # Contracts traded each bar
        
        # Commission calculation (based on Asset rules)
        if asset.commission_rate is not None:
            commission_cost = turnover * asset.price_data * asset.commission_rate * asset.point_value
        else:
            commission_cost = turnover * (asset.commission_per_contract or 0.0)
            
        # Slippage calculation
        slippage_cost = 0.0
        if asset.slippage_rate is not None:
            slippage_cost = turnover * asset.price_data * asset.slippage_rate * asset.point_value
            
        total_costs = commission_cost + slippage_cost
        
        # 6. Net PnL & Equity Curve
        daily_pnl = raw_pnl - total_costs
        equity = self.capital.initial_capital + daily_pnl.cumsum()
        
        # 7. Leverage Calculation (Delegated to Capital Model)
        notional = pos * asset.price_data * asset.point_value
        leverage = self.capital.capital_model.calculate_leverage(
            notional, equity, self.capital.initial_capital
        )

        # 8. Drawdown Calculation (Delegated to Capital Model)
        drawdown = self.capital.capital_model.calculate_drawdown(
            equity, self.capital.initial_capital
        ) * 100  # Convert to percentage for plotting

        # 9. Returns & Realized Volatility (Delegated to Capital Model)
        returns = self.capital.capital_model.calculate_returns(
            equity, daily_pnl, self.capital.initial_capital
        )
        
        # Calculate rolling realized volatility (standard 21-day window)
        # We multiply by sqrt(trading_days) to annualize it for the plot
        realized_vol = returns.rolling(window=21, min_periods=1).std() * np.sqrt(asset.trading_days)*100
        
        # 10. Cumulative series (Mandatory fields, even if costs are 0)
        cumulative_fees = total_costs.cumsum()
        cumulative_turnover = turnover.cumsum()
        
        # 11. Calculate Metrics
        metrics = calculate_metrics(
            equity=equity,
            daily_pnl=daily_pnl,
            capital= self.capital,
            cumulative_fees=cumulative_fees,
            cumulative_turnover=cumulative_turnover,
            trading_days=asset.trading_days
        )
        
        # 10. Return strict data container
        return BacktestResult(
            equity=equity,
            positions=pos,
            daily_pnl=daily_pnl,
            metrics=metrics,
            strategy_name=strategy.name,
            asset=asset,
            cumulative_fees=cumulative_fees,
            cumulative_turnover=cumulative_turnover,
            leverage=leverage,
            drawdown=drawdown,
            returns=returns,
            realized_vol = realized_vol
        )