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

# src/plots/reporter.py
import pandas as pd
import numpy as np
from typing import List
from ..core.models import ExecutionResult
from ..engine.portfolio import PortfolioExecutionResult
from ..engine.analyzer import PerformanceAnalyzer

# ... (keep existing print_single_summary and print_comparison_table) ...

def print_portfolio_diversification(portfolio_result: PortfolioExecutionResult) -> None:
    """Prints the correlation matrix and diversification benefits."""
    print("\n" + "═" * 60)
    print("  PORTFOLIO DIVERSIFICATION ANALYSIS")
    print("═" * 60)
    
    print(f"\n  Diversification Ratio:    {portfolio_result.diversification_ratio:>8.2f}")
    print(f"  Volatility Reduction:     {portfolio_result.volatility_reduction_pct:>7.1f}%")
    
    print("\n  Correlation Matrix:")
    print("─" * 60)
    # Format the correlation matrix nicely
    corr_str = portfolio_result.correlation_matrix.to_string(float_format="%.2f")
    # Indent the matrix for better readability
    for line in corr_str.split('\n'):
        print(f"  {line}")
    print("═" * 60 + "\n")


def print_portfolio_comparison(
    portfolio_result: PortfolioExecutionResult, 
    individual_results: List[ExecutionResult]
) -> None:
    """
    Compares individual strategies against the portfolio using the stateless Analyzer.
    """
    analyzer = PerformanceAnalyzer()
    
    # 1. Calculate metrics for all individual strategies
    ind_metrics_list = [analyzer.analyze(res) for res in individual_results]
    
    # 2. Calculate metrics for the portfolio
    port_metrics = analyzer.analyze(portfolio_result)
    
    # 3. Build the comparison data
    rows = []
    for i, res in enumerate(individual_results):
        m = ind_metrics_list[i]
        rows.append({
            'Strategy': res.strategy_name,
            'Weight': f"{portfolio_result.weights[i]*100:.0f}%",
            'CAGR %': f"{m.cagr_pct:.1f}",
            'Sharpe': f"{m.sharpe_ratio:.2f}",
            'Max DD %': f"{m.max_drawdown_pct:.1f}",
            'Vol %': f"{m.annual_volatility_pct:.1f}"
        })
        
    # Add Portfolio row
    rows.append({
        'Strategy': 'PORTFOLIO',
        'Weight': '100%',
        'CAGR %': f"{port_metrics.cagr_pct:.1f}",
        'Sharpe': f"{port_metrics.sharpe_ratio:.2f}",
        'Max DD %': f"{port_metrics.max_drawdown_pct:.1f}",
        'Vol %': f"{port_metrics.annual_volatility_pct:.1f}"
    })
    
    df = pd.DataFrame(rows)
    
    print("\n" + "═" * 60)
    print("  STRATEGY vs PORTFOLIO COMPARISON")
    print("═" * 60)
    print(df.to_string(index=False))
    print("═" * 60 + "\n")