"""
Plots module: Visualization and reporting utilities.
"""
from .plotter import plot_backtest_results, plot_regression
from .reporter import print_summary, print_comparison_table, print_report_comparison, print_portfolio_comparison, print_portfolio_diversification, print_regression

__all__ = [
    "plot_backtest_results",
    "plot_regression",
    "print_summary",
    "print_comparison_table",
    "print_report_comparison",
    "print_portfolio_comparison", 
    "print_portfolio_diversification",
    "print_regression"
]