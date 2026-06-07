import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class BacktestResult:
    strategy_name: str
    equity: pd.Series
    positions: pd.Series
    metrics: Dict[str, float]
    daily_pnl: pd.Series

class VectorizedEngine:
    def __init__(
        self,
        initial_capital: float = 100_000.0,
        point_value: float = 1,        # $ per point (e.g., ES=50, NQ=20) a multiplier for futures pricing, for any other assets put 1 in here
        commission_per_contract: float = None,  # Fixed $ per contract (legacy)
        commission_pct: float = None  ,     # Percentage of trade value (e.g., 0.001 = 0.1%)
        slippage_pct: float = None
    ):
        self.initial_capital = initial_capital
        self.point_value = point_value
        self.commission = commission_per_contract

        # Validate commission mode (mutually exclusive)
        if commission_per_contract is not None and commission_pct is not None:
            raise ValueError("Use either commission_per_contract OR commission_pct, not both")
            
        self.commission_per_contract = commission_per_contract
        self.commission_pct = commission_pct
        
        # Default to fixed commission if neither specified (backward compatible)
        if commission_per_contract is None and commission_pct is None:
            self.commission_per_contract = 0.0
        
        # Use slippage if defined
        self.slippage_pct = slippage_pct
        
    def run(self, data: pd.DataFrame, positions: pd.Series, strategy_name: str) -> BacktestResult:
        # 1. Align & shift positions (avoid lookahead: position[t] executes at close[t+1])
        pos = positions.reindex(data.index).fillna(0).shift(1).fillna(0)
        
        # 2. Calculate price changes & raw PnL
        price_change = data["close"].diff().fillna(0)
        raw_pnl = pos * price_change * self.point_value
        
        # 3. Apply transaction costs (only when position changes)
        turnover = pos.diff().abs().fillna(0)
        if self.commission_pct is not None:
            # % commission: cost = contracts × price × commission_pct × point_value
            costs = turnover * data["close"] * self.commission_pct * self.point_value
        else:
            # Fixed commission: cost = contracts × commission_per_contract
            costs = turnover * self.commission_per_contract
        
        if self.slippage_pct is not None:
            slippage_cost = turnover * data["close"] * self.slippage_pct * self.point_value
            daily_pnl -= slippage_cost
        
        # 4. Net daily PnL & equity curve
        daily_pnl = raw_pnl - costs
        equity = self.initial_capital + daily_pnl.cumsum()
        
        # 5. Calculate metrics
        metrics = self._calculate_metrics(equity, daily_pnl, pos)
        
        return BacktestResult(
            strategy_name=strategy_name,
            equity=equity,
            positions=pos,
            metrics=metrics,
            daily_pnl=daily_pnl
        )
        
    def _calculate_metrics(self, equity: pd.Series, daily_pnl: pd.Series, positions: pd.Series) -> Dict[str, float]:
        returns = equity.pct_change().dropna()
        
        total_return = (equity.iloc[-1] / self.initial_capital - 1) * 100
        cagr = (1 + total_return/100) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe = cagr / volatility if volatility > 0 else 0.0
        
        # Max Drawdown
        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max
        max_dd = drawdown.min() * 100
        
        # Trade stats (approximate via position changes)
        flips = positions.diff().abs()
        total_flips = flips.sum()
        
        return {
            "total_return_pct": round(total_return, 2),
            "cagr_pct": round(cagr * 100, 2),
            "annual_volatility_pct": round(volatility * 100, 2),
            "sharpe_ratio": round(sharpe, 2),
            "max_drawdown_pct": round(max_dd, 2),
            "total_position_flips": int(total_flips),
            "avg_daily_pnl": round(daily_pnl.mean(), 2),
            "final_equity": round(equity.iloc[-1], 2)
        }