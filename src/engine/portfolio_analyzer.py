import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union, Any

# Import the dataclass from models
from .models import PortfolioResult

class PortfolioAnalyzer:
    """
    Analyzes diversification benefits and portfolio statistics from multiple 
    backtest results.
    
    Accepts inputs in two formats:
    1. Dictionary: { 'Strategy Name': BacktestResult, ... }
    2. Two Lists:  strategies=[BacktestResult, ...], names=['Name 1', ...]
    """
    
    def __init__(
        self,
        strategies: Union[List[Any], Dict[str, Any]],
        names: Optional[List[str]] = None,
        weights: Optional[List[float]] = None,
        risk_free_rate: float = 0.0,
        trading_days_per_year: int = 252
    ):
        """
        Initialize portfolio analyzer.
        """
        # 1. Parse strategies and names based on input type
        if isinstance(strategies, dict):
            if not strategies:
                raise ValueError("Strategies dictionary cannot be empty.")
            self.strategy_names = list(strategies.keys())
            self.results = list(strategies.values())
        elif isinstance(strategies, list):
            if names is None:
                raise ValueError("If 'strategies' is a list, 'names' must be provided.")
            if len(strategies) != len(names):
                raise ValueError(
                    f"Length of 'strategies' ({len(strategies)}) must match "
                    f"length of 'names' ({len(names)})."
                )
            self.results = strategies
            self.strategy_names = names
        else:
            raise TypeError("'strategies' must be either a list or a dictionary.")
        
        # 2. Basic validation
        self.n_strategies = len(self.results)
        self.rf = risk_free_rate
        self.trading_days = trading_days_per_year
        
        if len(set(self.strategy_names)) != len(self.strategy_names):
            raise ValueError("Strategy names must be unique.")
        
        # 3. Set weights (default to equal weight 1/N)
        if weights is None:
            self.weights = np.ones(self.n_strategies) / self.n_strategies
        else:
            if len(weights) != self.n_strategies:
                raise ValueError(f"Number of weights ({len(weights)}) must match "
                               f"number of strategies ({self.n_strategies})")
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
        
        # 6. Calculate metrics
        self.metrics = self._calculate_portfolio_metrics()
        self.correlation_matrix = self.individual_returns.corr()
        
        # Extract individual metrics for the comparison table
        self.individual_metrics = [r.metrics for r in self.results]

    def _align_equity_curves(self) -> pd.DataFrame:
        """Align all equity curves to common date index."""
        equity_dict = {}
        for i, result in enumerate(self.results):
            name = self.strategy_names[i]
            equity = result.equity
            
            if equity is None:
                raise ValueError(f"Result '{name}' has None equity curve")
            if not isinstance(equity, pd.Series):
                raise ValueError(f"Result '{name}' equity must be pd.Series, got {type(equity)}")
            
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
        """Calculate daily returns for each strategy."""
        return self.aligned_equity.pct_change().dropna()
    
    def _calculate_portfolio_returns(self) -> pd.Series:
        """Calculate weighted portfolio returns."""
        portfolio_ret = (
            self.individual_returns[self.strategy_names] * self.weights
        ).sum(axis=1)
        portfolio_ret.name = 'Portfolio'
        return portfolio_ret
    
    def _calculate_portfolio_equity(self) -> pd.Series:
        """Calculate cumulative portfolio equity curve."""
        portfolio_equity = (1 + self.portfolio_returns).cumprod()
        if len(self.results) > 0:
            initial_capital = self.results[0].equity.iloc[0]
            portfolio_equity = portfolio_equity * initial_capital
        return portfolio_equity
    
    def _calculate_portfolio_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive portfolio performance metrics."""
        ret = self.portfolio_returns
        
        # Basic return metrics
        total_return = (1 + ret).prod() - 1
        n_years = len(ret) / self.trading_days
        cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0.0
        
        # Risk metrics
        volatility = ret.std() * np.sqrt(self.trading_days)
        downside_returns = ret[ret < 0]
        downside_vol = downside_returns.std() * np.sqrt(self.trading_days) if len(downside_returns) > 0 else 0.0
        
        sharpe = (cagr - self.rf) / volatility if volatility > 0 else 0.0
        sortino = (cagr - self.rf) / downside_vol if downside_vol > 0 else 0.0
        
        # Drawdown calculations
        cumulative = (1 + ret).cumprod()
        rolling_max = cumulative.cummax()
        drawdowns = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()
        avg_drawdown = drawdowns[drawdowns < 0].mean() if (drawdowns < 0).any() else 0.0
        
        # Tail risk & distribution
        sorted_returns = np.sort(ret)
        n = len(sorted_returns)
        lower_tail = sorted_returns[int(0.05 * n)] if n > 0 else 0.0
        upper_tail = sorted_returns[int(0.95 * n)] if n > 0 else 0.0
        
        metrics = {
            'total_return_pct': total_return * 100,
            'cagr_pct': cagr * 100,
            'annual_volatility_pct': volatility * 100,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown_pct': max_drawdown * 100,
            'avg_drawdown_pct': avg_drawdown * 100,
            'skew': ret.skew(),
            'kurtosis': ret.kurtosis(),
            'lower_tail_pct': lower_tail * 100,
            'upper_tail_pct': upper_tail * 100,
            'n_trading_days': len(ret),
            'n_years': n_years
        }

        # --- Fee & Turnover Metrics (Matching VectorizedEngine) ---
        initial_capital = self.results[0].equity.iloc[0]
        
        daily_fees = self.portfolio_cumulative_fees.diff().fillna(0) if self.portfolio_cumulative_fees is not None else pd.Series(0, index=ret.index)
        daily_turnover = self.portfolio_cumulative_turnover.diff().fillna(0) if self.portfolio_cumulative_turnover is not None else pd.Series(0, index=ret.index)
        
        net_pnl = self.portfolio_equity.iloc[-1] - initial_capital
        total_fees = daily_fees.sum()
        gross_pnl = net_pnl + total_fees
        
        fee_drag_ratio = total_fees / abs(gross_pnl) if gross_pnl != 0 else 0.0
        cost_efficiency = net_pnl / gross_pnl if gross_pnl != 0 else 1.0
        
        gross_return = gross_pnl / initial_capital
        gross_cagr = (1 + gross_return) ** (1 / n_years) - 1 if n_years > 0 else 0.0
        
        equity_prev = self.portfolio_equity.shift(1).fillna(initial_capital)
        gross_returns = ret + (daily_fees / equity_prev)
        gross_vol = gross_returns.std() * np.sqrt(self.trading_days)
        gross_sharpe = gross_cagr / gross_vol if gross_vol > 0 else 0.0
        
        net_sharpe = metrics.get('sharpe_ratio', 0.0)
        sharpe_drag = gross_sharpe - net_sharpe
        
        avg_daily_turnover = daily_turnover.mean()
        turnover_penalty = avg_daily_turnover * 0.1
        turnover_adjusted_sharpe = max(0.0, net_sharpe - turnover_penalty)
        
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
            'total_turnover': daily_turnover.sum(),
            'total_fee_drag_pct': (total_fees / initial_capital) * 100,
            'annualized_fee_drag_pct': ((total_fees / initial_capital) * 100 / n_years) if n_years > 0 else 0.0
        })
        
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
            portfolio_cumulative_turnover=self.portfolio_cumulative_turnover
        )


# ==========================================
# Convenience Function
# ==========================================
def run_portfolio(
    strategies: Union[List[Any], Dict[str, Any]],
    names: Optional[List[str]] = None,
    weights: Optional[List[float]] = None,
    risk_free_rate: float = 0.0
) -> PortfolioResult:
    """
    Convenience function to run a portfolio analysis in one line.
    """
    analyzer = PortfolioAnalyzer(
        strategies=strategies,
        names=names,
        weights=weights,
        risk_free_rate=risk_free_rate
    )
    return analyzer.run()