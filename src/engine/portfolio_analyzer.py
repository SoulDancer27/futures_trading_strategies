"""
Portfolio Analyzer.
Aggregates multiple backtest results into a unified portfolio and calculates 
diversification benefits and portfolio-level metrics.
Uses Capital Models to separate Fixed vs Compounding logic.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union, Any

from .models import PortfolioResult, BacktestResult
from .metrics import calculate_metrics, calculate_diversification_metrics
from .capital_models import BaseCapitalModel, FixedCapitalModel

class PortfolioAnalyzer:
    """
    Analyzes diversification benefits and portfolio statistics from multiple 
    backtest results.
    """
    
    def __init__(
        self,
        strategies: Union[List[BacktestResult], Dict[str, BacktestResult]],
        names: Optional[List[str]] = None,
        weights: Optional[List[float]] = None,
        risk_free_rate: float = 0.0,
        trading_days_per_year: int = 252,
        capital_model: Optional[BaseCapitalModel] = None,
    ):
        # 1. Parse strategies and names
        if isinstance(strategies, dict):
            if not strategies:
                raise ValueError("Strategies dictionary cannot be empty.")
            self.strategy_names = list(strategies.keys())
            self.results = list(strategies.values())
        elif isinstance(strategies, list):
            if names is None:
                raise ValueError("If 'strategies' is a list, 'names' must be provided.")
            if len(strategies) != len(names):
                raise ValueError("Length of 'strategies' must match length of 'names'.")
            self.results = strategies
            self.strategy_names = names
        else:
            raise TypeError("'strategies' must be either a list or a dictionary.")
        
        # 2. Basic validation & setup
        self.n_strategies = len(self.results)
        self.rf = risk_free_rate
        self.trading_days = trading_days_per_year
        
        # Default to Fixed Capital Model if none provided
        self.capital_model = capital_model if capital_model is not None else FixedCapitalModel()
        
        if len(set(self.strategy_names)) != len(self.strategy_names):
            raise ValueError("Strategy names must be unique.")
        
        # 3. Set weights (default to equal weight 1/N)
        if weights is None:
            self.weights = np.ones(self.n_strategies) / self.n_strategies
        else:
            if len(weights) != self.n_strategies:
                raise ValueError("Number of weights must match number of strategies")
            self.weights = np.array(weights) / np.sum(weights)
        
        # 4. Align and process data
        self.aligned_equity = self._align_equity_curves()
        self.individual_returns = self._calculate_individual_returns()
        self.portfolio_returns = self._calculate_portfolio_returns()
        self.portfolio_equity = self._calculate_portfolio_equity()
        
        # 5. Align optional series (fees and turnover)
        self.aligned_fees = self._align_optional_series('cumulative_fees')
        self.aligned_turnover = self._align_optional_series('cumulative_turnover')
        self.portfolio_cumulative_fees = self._calculate_weighted_series(self.aligned_fees)
        self.portfolio_cumulative_turnover = self._calculate_weighted_series(self.aligned_turnover)
        
        # 6. Calculate metrics using the UNIFIED metrics.py module
        self.metrics = self._calculate_portfolio_metrics()
        self.diversification_metrics = calculate_diversification_metrics(
            self.individual_returns, self.weights, self.strategy_names, self.trading_days
        )
        self.correlation_matrix = self.individual_returns[self.strategy_names].corr()
        
        # Extract individual metrics for the comparison table
        self.individual_metrics = [r.metrics for r in self.results]
        self.portfolio_leverage = self._calculate_portfolio_leverage()

    def _align_equity_curves(self) -> pd.DataFrame:
        """Align all equity curves to common date index."""
        equity_dict = {}
        for i, result in enumerate(self.results):
            name = self.strategy_names[i]
            equity = result.equity
            
            if equity is None:
                raise ValueError(f"Result '{name}' has None equity curve")
            if not isinstance(equity, pd.Series):
                raise ValueError(f"Result '{name}' equity must be pd.Series")
            
            equity_dict[name] = equity
        
        df = pd.DataFrame(equity_dict).ffill().dropna()
        
        if len(df.columns) != self.n_strategies:
            missing = set(self.strategy_names) - set(df.columns)
            raise ValueError(f"Failed to align all equity curves. Missing: {missing}")
        
        return df

    def _align_optional_series(self, attr_name: str) -> Optional[pd.DataFrame]:
        """Aligns optional Series (like cumulative_fees) from results."""
        series_dict = {}
        has_data = False
        
        for i, result in enumerate(self.results):
            series = getattr(result, attr_name, None)
            if series is not None and isinstance(series, pd.Series):
                series_dict[self.strategy_names[i]] = series
                has_data = True
                
        if not has_data:
            return None
            
        return pd.DataFrame(series_dict).ffill().fillna(0)

    def _calculate_weighted_series(self, df: Optional[pd.DataFrame]) -> Optional[pd.Series]:
        """Calculates the weighted sum of an aligned DataFrame."""
        if df is None:
            return None
        return (df * self.weights).sum(axis=1)

    def _calculate_individual_returns(self) -> pd.DataFrame:
        """Calculate daily returns for each strategy (Delegated to Capital Model)."""
        returns_dict = {}
        for i, result in enumerate(self.results):
            name = self.strategy_names[i]
            initial_cap = result.equity.iloc[0]
            
            # NO IF/ELSE! The model knows how to calculate returns.
            ret = self.capital_model.calculate_returns(result.equity, result.daily_pnl, initial_cap)
            returns_dict[name] = ret
            
        return pd.DataFrame(returns_dict).ffill().dropna()

    def _calculate_portfolio_returns(self) -> pd.Series:
        """Calculate weighted portfolio returns."""
        portfolio_ret = (
            self.individual_returns[self.strategy_names] * self.weights
        ).sum(axis=1)
        portfolio_ret.name = 'Portfolio'
        return portfolio_ret
    
    def _calculate_portfolio_equity(self) -> pd.Series:
        """Calculate cumulative portfolio equity curve (Delegated to Capital Model)."""
        initial_capital = self.results[0].equity.iloc[0]

        return self.capital_model.calculate_equity(self.portfolio_returns, initial_capital)

    def _calculate_portfolio_leverage(self) -> Optional[pd.Series]:
        """Calculate true portfolio leverage using embedded price data and point_value."""
        total_notional = None
        
        for result in self.results:
            if result.price_data is not None and result.positions is not None:
                # Calculate notional for this strategy
                notional = (result.positions * result.price_data * result.point_value)
                
                # Align to portfolio index
                notional = notional.reindex(self.portfolio_equity.index).fillna(0)
                
                # Sum all notionals
                total_notional = notional if total_notional is None else total_notional + notional
        
        if total_notional is not None:
            # Get initial capital for the portfolio (from the first strategy)
            initial_capital = self.results[0].equity.iloc[0]
            
            # Delegate to the capital model!
            return self.capital_model.calculate_leverage(
                total_notional, 
                self.portfolio_equity, 
                initial_capital
            )
        
        return None

    def _calculate_portfolio_metrics(self) -> Dict[str, float]:
        """Delegate ALL portfolio metric calculations to metrics.py."""
        initial_capital = self.results[0].equity.iloc[0]
        
        # Calculate portfolio daily PnL (needed for the metrics function signature)
        port_daily_pnl = self.portfolio_equity.diff().fillna(0)

        # Call the unified function
        metrics = calculate_metrics(
            equity=self.portfolio_equity,
            daily_pnl=port_daily_pnl,
            capital_model=self.capital_model,  # <-- PASSES THE MODEL
            initial_capital=initial_capital,
            positions=None,
            cumulative_fees=self.portfolio_cumulative_fees,
            cumulative_turnover=self.portfolio_cumulative_turnover,
            risk_free_rate=self.rf,
            trading_days=self.trading_days
        )
        
        # Add trading days/years for reference
        metrics['n_trading_days'] = len(self.portfolio_returns)
        metrics['n_years'] = len(self.portfolio_returns) / self.trading_days
        
        return metrics

    def run(self) -> PortfolioResult:
        """Execute the portfolio analysis and return the smart dataclass."""
        return PortfolioResult(
            portfolio_equity=self.portfolio_equity,
            portfolio_returns=self.portfolio_returns,
            individual_returns=self.individual_returns,
            weights=self.weights,
            strategy_names=self.strategy_names,
            metrics=self.metrics,
            correlation_matrix=self.correlation_matrix,
            individual_metrics=self.individual_metrics,
            trading_days=self.trading_days,
            portfolio_cumulative_fees=self.portfolio_cumulative_fees,
            portfolio_cumulative_turnover=self.portfolio_cumulative_turnover,
            portfolio_leverage=self.portfolio_leverage
        )