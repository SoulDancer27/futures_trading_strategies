from dataclasses import dataclass
from typing import Dict, Optional, List, Union, Any
import pandas as pd
import numpy as np

@dataclass
class BacktestResult:
    """
    Immutable container for backtest execution results.
    """
    equity: pd.Series           # Daily portfolio value (currency units)
    positions: pd.Series        # Contracts/shares held at each timestamp
    metrics: Dict[str, float]   # Pre-calculated performance & risk statistics
    daily_pnl: pd.Series        # Day-over-day profit/loss in currency units
    strategy_name: str = ""     # Human-readable strategy identifier
    ticker: str = ""            # Instrument/symbol being tested
    cumulative_fees: Optional[pd.Series] = None
    cumulative_turnover: Optional[pd.Series] = None

    def print_metrics(self) -> None:
        """Print detailed performance metrics in a vertical key-value format."""
        print("📊 Backtest Metrics:")
        for key, value in self.metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")

    def print_summary(self) -> None:
        """Print a condensed, book-style summary table."""
        years = len(self.equity) / 252
        
        # Table header with strategy/instrument metadata
        print("┌" + "─" * 70 + "┐")  # <-- FIXED: Added the missing right corner
        print(f"│ Strategy: {self.strategy_name or 'Backtest':<40} │ {self.ticker or 'Instrument':<24} │")
        print("├" + "─" * 70 + "┤")  # <-- FIXED: Added the missing left corner '├'
        
        metrics_table = [
            ("Years of data", f"{years:.1f}"),
            ("Mean annual return", f"{self.metrics.get('cagr_pct', 0):+.1f}%"),
            ("Average drawdown", f"{self.metrics.get('avg_drawdown_pct', 0):.1f}%"), 
            ("Maximum drawdown", f"{self.metrics.get('max_drawdown_pct', 0):.1f}%"),
            ("Annualised standard deviation", f"{self.metrics.get('annual_volatility_pct', 0):.1f}%"),
            ("Sharpe ratio", f"{self.metrics.get('sharpe_ratio', 0):.2f}"),
            ("Skew", f"{self.metrics.get('skew', 0):.2f}"),
            ("Lower tail", f"{self.metrics.get('lower_tail', 0):.2f}"),
            ("Upper tail", f"{self.metrics.get('upper_tail', 0):.2f}"),
        ]
        
        for label, value in metrics_table:
            print(f"│ {label:<40} │ {value:>24} │")
        
        print("└" + "─" * 70 + "┘")


@dataclass
class PortfolioResult:
    """
    Smart container for portfolio-level analysis results.
    """
    portfolio_equity: pd.Series
    portfolio_returns: pd.Series
    individual_returns: pd.DataFrame
    weights: np.ndarray
    strategy_names: List[str]
    metrics: Dict[str, float]
    correlation_matrix: pd.DataFrame
    individual_metrics: List[Dict[str, float]]
    trading_days: int = 252
    
    portfolio_cumulative_fees: Optional[pd.Series] = None
    portfolio_cumulative_turnover: Optional[pd.Series] = None

    # ==========================================
    # 1. Diversification Metrics (Calculated on demand)
    # ==========================================
    def get_diversification_metrics(self) -> Dict[str, Any]:
        """Calculate diversification-specific metrics."""
        ind_vols = self.individual_returns[self.strategy_names].std() * np.sqrt(self.trading_days)
        port_vol = self.portfolio_returns.std() * np.sqrt(self.trading_days)
        weighted_sum_vols = np.sum(self.weights * ind_vols.values)
        
        div_ratio = weighted_sum_vols / port_vol if port_vol > 0 else 1.0
        vol_reduction = (1 - port_vol / weighted_sum_vols) * 100 if weighted_sum_vols > 0 else 0.0
        
        upper_tri = self.correlation_matrix.where(
            np.triu(np.ones(self.correlation_matrix.shape), k=1).astype(bool)
        )
        avg_corr = upper_tri.stack().mean()
        
        return {
            'diversification_ratio': div_ratio,
            'volatility_reduction_pct': vol_reduction,
            'avg_correlation': avg_corr,
            'individual_vols_pct': {name: vol * 100 for name, vol in zip(self.strategy_names, ind_vols)},
            'portfolio_vol_pct': port_vol * 100
        }

    # ==========================================
    # 2. Text Summary
    # ==========================================
    def print_summary(self) -> None:
        """Print comprehensive portfolio summary."""
        print("\n" + "═" * 80)
        print(" PORTFOLIO ANALYSIS SUMMARY")
        print("═" * 80)
        
        print("\nStrategy Allocation:")
        for name, weight in zip(self.strategy_names, self.weights):
            print(f"  {name:<30}: {weight*100:>6.1f}%")
        
        print("\n" + "─" * 80)
        print("Portfolio Performance Metrics:")
        print("─" * 80)
        metrics_display = [
            ("Total Return", f"{self.metrics.get('total_return_pct', 0):+.2f}%"),
            ("CAGR", f"{self.metrics.get('cagr_pct', 0):+.2f}%"),
            ("Annual Volatility", f"{self.metrics.get('annual_volatility_pct', 0):.2f}%"),
            ("Sharpe Ratio", f"{self.metrics.get('sharpe_ratio', 0):.2f}"),
            ("Sortino Ratio", f"{self.metrics.get('sortino_ratio', 0):.2f}"),
            ("Max Drawdown", f"{self.metrics.get('max_drawdown_pct', 0):.2f}%"),
            ("Gross Sharpe", f"{self.metrics.get('gross_sharpe_ratio', 0):.2f}"),
            ("Fee Drag Ratio", f"{self.metrics.get('fee_drag_ratio', 0):.2f}"),
        ]
        for label, value in metrics_display:
            print(f"  {label:<35} {value:>20}")
        
        div_metrics = self.get_diversification_metrics()
        print("\n" + "─" * 80)
        print("Diversification Benefits:")
        print("─" * 80)
        print(f"  Diversification Ratio:     {div_metrics['diversification_ratio']:>20.2f}")
        print(f"  Volatility Reduction:      {div_metrics['volatility_reduction_pct']:>19.1f}%")
        print(f"  Average Correlation:       {div_metrics['avg_correlation']:>20.2f}")
        print("\n" + "═" * 80 + "\n")

    # ==========================================
    # 3. Comparison Table
    # ==========================================
    def get_comparison_table(self) -> pd.DataFrame:
        """Create a comparison table of individual strategies vs portfolio."""
        comparison_data = {
            'Strategy': self.strategy_names + ['Portfolio'],
            'Weight': list(self.weights) + [1.0],
        }
        
        for i, ind_metrics in enumerate(self.individual_metrics):
            for metric, value in ind_metrics.items():
                if metric not in comparison_data:
                    comparison_data[metric] = [0.0] * (len(self.strategy_names) + 1)
                comparison_data[metric][i] = value
        
        for metric, value in self.metrics.items():
            if metric not in comparison_data:
                comparison_data[metric] = [0.0] * (len(self.strategy_names) + 1)
            comparison_data[metric][-1] = value
        
        return pd.DataFrame(comparison_data)

    # ==========================================
    # 4. Conversion to BacktestResult
    # ==========================================
    def to_backtest_result(self, name: str = "Portfolio") -> BacktestResult:
        """Convert portfolio analysis to BacktestResult format for visualization."""
        daily_pnl = self.portfolio_equity.diff().fillna(0)
        initial_capital = self.portfolio_equity.iloc[0]
        portfolio_positions = self.portfolio_leverage if hasattr(self, 'portfolio_leverage') and self.portfolio_leverage is not None else pd.Series(1.0, index=self.portfolio_equity.index)

        return BacktestResult(
            equity=self.portfolio_equity,
            positions=portfolio_positions,
            metrics=self.metrics,
            daily_pnl=daily_pnl,
            strategy_name=name,
            ticker="PORTFOLIO",
            cumulative_fees=self.portfolio_cumulative_fees,
            cumulative_turnover=self.portfolio_cumulative_turnover
        )