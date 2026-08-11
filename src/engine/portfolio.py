"""
Portfolio Builder and Analysis.
Combines multiple ExecutionResults into a unified PortfolioExecutionResult.
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Union, Optional

from ..core.models import ExecutionResult, Asset


@dataclass
class PortfolioExecutionResult(ExecutionResult):
    """
    Extends ExecutionResult with portfolio-specific data.
    Fully compatible with the standard Plotter and PerformanceAnalyzer.
    """
    weights: np.ndarray = None
    correlation_matrix: pd.DataFrame = None
    diversification_ratio: float = 0.0
    volatility_reduction_pct: float = 0.0


class PortfolioBuilder:
    """
    Aggregates multiple backtest results into a single portfolio time series.
    """
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Args:
            weights: Dictionary mapping strategy names to target weights (e.g., {'Strat A': 0.6, 'Strat B': 0.4}).
                     If None, uses equal weighting (1/N).
        """
        self.target_weights = weights

    def build(self, results: Union[List[ExecutionResult], Dict[str, ExecutionResult]]) -> PortfolioExecutionResult:
        # 1. Normalize input to a dictionary
        if isinstance(results, list):
            res_dict = {r.strategy_name: r for r in results}
        else:
            res_dict = results
            
        names = list(res_dict.keys())
        if len(names) < 2:
            raise ValueError("Portfolio requires at least 2 strategies.")

        # 2. Align Equity and Returns
        equity_df = pd.DataFrame({name: res_dict[name].equity for name in names}).ffill().dropna()
        returns_df = pd.DataFrame({name: res_dict[name].returns for name in names}).reindex(equity_df.index).fillna(0)
        
        # 3. Calculate Weights
        if self.target_weights is None:
            w = np.ones(len(names)) / len(names)
        else:
            w = np.array([self.target_weights.get(n, 0.0) for n in names])
            if w.sum() == 0:
                raise ValueError("Weights sum to zero.")
            w = w / w.sum()

        # 4. Calculate Portfolio Series
        # Weighted average of daily returns
        port_returns = (returns_df * w).sum(axis=1)
        
        # Convert returns to PnL and Equity (assuming Fixed Capital model logic)
        initial_cap = equity_df.iloc[0, 0] # Use first strategy's capital as base
        port_daily_pnl = port_returns * initial_cap
        port_equity = initial_cap + port_daily_pnl.cumsum()
        
        # Drawdown
        drawdown = ((port_equity - port_equity.cummax()) / initial_cap * 100)
        
        # Leverage (Weighted average of individual leverages)
        leverage_df = pd.DataFrame({name: res_dict[name].leverage for name in names}).reindex(port_equity.index).fillna(0)
        port_leverage = (leverage_df * w).sum(axis=1)

        # 5. Calculate Diversification Metrics
        corr_matrix = returns_df.corr()
        ind_vols = returns_df.std() * np.sqrt(res_dict[names[0]].asset.trading_days)
        port_vol = port_returns.std() * np.sqrt(res_dict[names[0]].asset.trading_days)
        weighted_sum_vols = np.sum(w * ind_vols.values)
        
        div_ratio = weighted_sum_vols / port_vol if port_vol > 0 else 1.0
        vol_reduction = (1 - port_vol / weighted_sum_vols) * 100 if weighted_sum_vols > 0 else 0.0

        # 6. Create Dummy Asset for Compatibility
        # The PerformanceAnalyzer needs an asset to get trading_days.
        dummy_asset = Asset(
            ticker="PORTFOLIO",
            price_data=port_equity, 
            trading_days=res_dict[names[0]].asset.trading_days
        )

        return PortfolioExecutionResult(
            equity=port_equity,
            daily_pnl=port_daily_pnl,
            positions=pd.Series(0.0, index=port_equity.index), # Dummy positions
            leverage=port_leverage,
            drawdown=drawdown,
            returns=port_returns,
            realized_vol=port_returns.rolling(21, min_periods=1).std() * np.sqrt(dummy_asset.trading_days) * 100,
            cumulative_fees=pd.Series(0.0, index=port_equity.index), # TODO: Aggregate fees later
            cumulative_turnover=pd.Series(0.0, index=port_equity.index),
            strategy_name="Portfolio",
            asset=dummy_asset,
            risk_free_rate=res_dict[names[0]].risk_free_rate,
            weights=w,
            correlation_matrix=corr_matrix,
            diversification_ratio=div_ratio,
            volatility_reduction_pct=vol_reduction
        )