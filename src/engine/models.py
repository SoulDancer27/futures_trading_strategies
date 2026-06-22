from dataclasses import dataclass
from typing import Dict, Optional
import pandas as pd

@dataclass
class BacktestResult:
    """
    Immutable container for backtest execution results.
    
    Stores the equity curve, position sizing, performance metrics, and metadata
    required for reporting, visualization, and statistical analysis. Designed to
    be strategy-agnostic and easily serializable for database storage or CSV export.
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
        """
        Print detailed performance metrics in a vertical key-value format.
        
        Ideal for terminal inspection, logging, or Jupyter notebook output.
        Formats all floats to 2 decimal places for consistent readability.
        Non-float values (e.g., counts, strings) are printed as-is.
        """
        print("📊 Backtest Metrics:")
        for key, value in self.metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")


    def print_summary(self) -> None:
        """
        Print a condensed, book-style summary table matching industry standards.
        
        Outputs a fixed-width ASCII table with aligned columns for quick visual
        scanning. Includes core return/risk metrics, tail-risk measures, and
        strategy metadata. Matches the formatting style used in quantitative
        trading literature (e.g., Advanced Futures Trading Strategies).
        
        Calculations & Conventions:
          - Years of data: Based on 252 trading days/year (equity markets)
          - Percentages: Formatted with explicit signs (+/-) where relevant
          - Missing metrics: Safely fallback to 0 via .get() to prevent KeyError
          - Alignment: Left-aligned labels (40 chars), right-aligned values (24 chars)
        """
        # Calculate backtest duration in years (standard 252 trading days/year)
        years = len(self.equity) / 252
        
        # Table header with strategy/instrument metadata
        print("┌" + "─" * 70 + "")
        print(f"│ Strategy: {self.strategy_name or 'Backtest':<40} │ {self.ticker or 'Instrument':<24} │")
        print("├" + "─" * 70 + "┤")
        
        # Define metric rows with safe .get() fallbacks to handle missing keys
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
        
        # Render table rows with consistent spacing and right-aligned values
        for label, value in metrics_table:
            print(f"│ {label:<40} │ {value:>24} │")
        
        print("└" + "─" * 70 + "┘")