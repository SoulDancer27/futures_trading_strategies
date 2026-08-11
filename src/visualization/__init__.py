"""
Plots module: Visualization and reporting utilities.
"""
from .plotter import plot_backtest_results
from .reporter import print_summary, print_comparison_table

__all__ = [
    "plot_backtest_results",
    "print_summary",
    "print_comparison_table"
]