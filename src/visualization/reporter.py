"""
Console Reporting Module.
Handles formatting and printing of PerformanceMetrics to the console.
"""
from ..core.models import PerformanceMetrics
from typing import Dict, List

def print_summary(name: str, metrics: PerformanceMetrics) -> None:
    """Prints a detailed summary for a single strategy."""
    print("\n" + "═" * 60)
    print(f"  STRATEGY SUMMARY: {name}")
    print("═" * 60)
    
    # Core Performance
    print(f" Total Return:      {metrics.total_return_pct:>8.2f}%")
    print(f" CAGR:              {metrics.cagr_pct:>8.2f}%")
    print(f" Annual Volatility: {metrics.annual_volatility_pct:>8.2f}%")
    print(f" Sharpe Ratio:      {metrics.sharpe_ratio:>8.2f}")
    print(f" Sortino Ratio:     {metrics.sortino_ratio:>8.2f}")
    
    # Risk & Drawdown
    print("-" * 60)
    print(f" Max Drawdown:      {metrics.max_drawdown_pct:>8.2f}%")
    print(f" Avg Drawdown:      {metrics.avg_drawdown_pct:>8.2f}%")
    print(f" Win Rate:          {metrics.win_rate_pct:>8.2f}%")
    
    # Distribution
    print("-" * 60)
    print(f" Skew:              {metrics.skew:>8.2f}")
    print(f" Kurtosis:          {metrics.kurtosis:>8.2f}")
    print(f" Tail Risk:         {metrics.tail_risk:>8.2f}")
    
    # Fees & Turnover
    print("-" * 60)
    print(f" Gross PnL:         ${metrics.gross_pnl:>10,.2f}")
    print(f" Net PnL:           ${metrics.net_pnl:>10,.2f}")
    print(f" Total Fees:        ${metrics.total_fees_currency:>10,.2f}")
    print(f" Fee Drag Ratio:    {metrics.fee_drag_ratio:>8.2f}")
    print(f" Cost Efficiency:   {metrics.cost_efficiency:>8.2f}")
    print("═" * 60 + "\n")


def print_comparison_table(results: Dict[str, PerformanceMetrics]) -> None:
    """
    Prints a clean, aligned comparison table for multiple strategies.
    """
    if not results:
        print("No results to compare.")
        return

    # Define the columns we want to show in the comparison
    columns = [
        ("Strategy", lambda m: m.strategy_name if hasattr(m, 'strategy_name') else "Unknown", 15),
        ("Total Ret %", lambda m: f"{m.total_return_pct:.1f}", 10),
        ("CAGR %", lambda m: f"{m.cagr_pct:.1f}", 8),
        ("Sharpe", lambda m: f"{m.sharpe_ratio:.2f}", 8),
        ("Max DD %", lambda m: f"{m.max_drawdown_pct:.1f}", 8),
        ("Win Rate %", lambda m: f"{m.win_rate_pct:.1f}", 9),
        ("Fees $", lambda m: f"{m.total_fees_currency:,.0f}", 10),
    ]

    # Header
    header = "│ " + " │ ".join(f"{col[0]:<{col[2]}}" for col in columns) + " │"
    separator = "├" + "┼".join("─" * (col[2] + 2) for col in columns) + "┤"
    top_border = "┌" + "┬".join("─" * (col[2] + 2) for col in columns) + "┐"
    bottom_border = "└" + "┴".join("─" * (col[2] + 2) for col in columns) + "┘"

    print("\n" + top_border)
    print(header)
    print(separator)

    # Rows
    for name, metrics in results.items():
        row = "│ " + " │ ".join(f"{col[1](metrics):<{col[2]}}" for col in columns) + " │"
        print(row)

    print(bottom_border + "\n")