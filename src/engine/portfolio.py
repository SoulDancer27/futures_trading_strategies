"""
Portfolio Analysis.
Combines multiple ExecutionResults into a unified Portfolio.
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Union, Optional

from ..core.models import ExecutionResult
from ..core.asset import Asset

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



class Portfolio(PortfolioExecutionResult):
    """
    A fully formed Portfolio object. 
    Inherits from ExecutionResult, so it can be passed directly to the Analyzer or Plotter.
    """
    
    def __init__(
        self, 
        results: Union[List[ExecutionResult], Dict[str, ExecutionResult]], 
        weights: Optional[Union[Dict[str, float], List[float]]] = None
    ):
        # 1. Normalize input to a dictionary (deduplicate names for list input)
        if isinstance(results, list):
            res_dict = {}
            for i, r in enumerate(results):
                base = r.strategy_name or f"Strategy {i + 1}"
                name, n = base, 2
                while name in res_dict:
                    name = f"{base} ({n})"
                    n += 1
                res_dict[name] = r
        else:
            res_dict = results
            
        names = list(res_dict.keys())
        if len(names) < 1:
            raise ValueError("Portfolio requires at least 1 strategy.")

        # 2. Align on a unified business-day calendar over the common date range.
        #    Each series is reindexed to the calendar; missing days are
        #    forward-filled (equity/positions/fees carry over) or zeroed
        #    (no return on a day the asset didn't trade).
        common_start = max(res_dict[name].equity.index.min() for name in names)
        common_end = min(res_dict[name].equity.index.max() for name in names)
        calendar = pd.bdate_range(common_start, common_end)

        equity_df = pd.DataFrame({name: res_dict[name].equity for name in names}).reindex(calendar).ffill().dropna()
        returns_df = pd.DataFrame({name: res_dict[name].returns for name in names}).reindex(equity_df.index).fillna(0)
        
        # 3. Calculate Weights
        if weights is None:
            w = np.ones(len(names)) / len(names)
        elif isinstance(weights, dict):
            w = np.array([weights.get(n, 0.0) for n in names])
            if w.sum() == 0: raise ValueError("Weights sum to zero.")
            w = w / w.sum()
        elif isinstance(weights, list):
            if len(weights) != len(names):
                raise ValueError(f"Number of weights ({len(weights)}) must match number of strategies ({len(names)}).")
            w = np.array(weights)
            if w.sum() == 0: raise ValueError("Weights sum to zero.")
            w = w / w.sum()
        else:
            raise TypeError("weights must be None, dict, or list")

        # 4. Calculate Portfolio Series
        port_returns = (returns_df * w).sum(axis=1)
        # Initial capital: equity starts flat at capital (first daily PnL is 0),
        # so the first equity value of any strategy equals the initial capital.
        initial_cap = res_dict[names[0]].equity.iloc[0]
        port_daily_pnl = port_returns * initial_cap
        port_equity = initial_cap + port_daily_pnl.cumsum()
        drawdown = ((port_equity - port_equity.cummax()) / initial_cap * 100)
        
        # Leverage (Weighted average of individual leverages)
        leverage_df = pd.DataFrame({name: res_dict[name].leverage for name in names}).reindex(port_equity.index).ffill().fillna(0)
        port_leverage = (leverage_df * w).sum(axis=1)

        # 5. Aggregate REAL Fees and Turnover (Weighted sum)
        fees_df = pd.DataFrame({name: res_dict[name].cumulative_fees for name in names}).reindex(port_equity.index).ffill().fillna(0)
        port_cumulative_fees = (fees_df * w).sum(axis=1)

        turnover_df = pd.DataFrame({name: res_dict[name].cumulative_turnover for name in names}).reindex(port_equity.index).ffill().fillna(0)
        port_cumulative_turnover = (turnover_df * w).sum(axis=1)

        # 6. Create Composite Price Index from REAL asset prices
        price_series_list = []
        for name in names:
            asset_price = res_dict[name].asset.price_data
            if asset_price is not None:
                normalized_price = (asset_price / asset_price.iloc[0]) * 100
                price_series_list.append(normalized_price)
        
        if price_series_list:
            price_df = pd.DataFrame(price_series_list).T
            composite_price = (price_df * w).sum(axis=1)
            composite_price.name = 'Portfolio_Composite_Price'
        else:
            composite_price = None

        # 7. Calculate Diversification Metrics
        if len(names) == 1:
            corr_matrix = pd.DataFrame([[1.0]], columns=names, index=names)
            div_ratio = 1.0
            vol_reduction = 0.0
        else:
            corr_matrix = returns_df.corr()
            ind_vols = returns_df.std() * np.sqrt(res_dict[names[0]].asset.trading_days)
            port_vol = port_returns.std() * np.sqrt(res_dict[names[0]].asset.trading_days)
            weighted_sum_vols = np.sum(w * ind_vols.values)
            
            div_ratio = weighted_sum_vols / port_vol if port_vol > 0 else 1.0
            vol_reduction = (1 - port_vol / weighted_sum_vols) * 100 if weighted_sum_vols > 0 else 0.0

        # 8. Create Dummy Asset for Compatibility
        dummy_asset = Asset(
            ticker="PORTFOLIO",
            price_data=composite_price, 
            trading_days=res_dict[names[0]].asset.trading_days,
            point_value=1.0
        )

        # 9. Initialize the parent ExecutionResult class with all calculated data
        super().__init__(
            equity=port_equity,
            daily_pnl=port_daily_pnl,
            positions=pd.Series(0.0, index=port_equity.index),
            leverage=port_leverage,
            drawdown=drawdown,
            returns=port_returns,
            realized_vol=port_returns.rolling(21, min_periods=1).std() * np.sqrt(dummy_asset.trading_days) * 100,
            cumulative_fees=port_cumulative_fees,
            cumulative_turnover=port_cumulative_turnover,
            strategy_name="Portfolio" if len(names) > 1 else names[0],
            asset=dummy_asset,
            risk_free_rate=res_dict[names[0]].risk_free_rate,
            weights=w,
            correlation_matrix=corr_matrix,
            diversification_ratio=div_ratio,
            volatility_reduction_pct=vol_reduction
        )