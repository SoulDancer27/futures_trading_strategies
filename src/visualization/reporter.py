"""
Modular Reporting Module for Backtest Results.
Handles console printing of summaries, comparison tables, and portfolio analysis.
Strictly "dumb": reads pre-calculated metrics from PerformanceMetrics dataclass.
"""
import pandas as pd
import numpy as np
from typing import List, Tuple, Union, Dict, Any, Optional

from ..core.models import PerformanceMetrics
from ..engine.portfolio import PortfolioExecutionResult


# ==========================================
# 1. METADATA & FORMATTING RULES
# ==========================================
METRIC_CATEGORIES = {
    " Returns & Performance": [
        "total_return_pct", "cagr_pct", "gross_return_pct",
    ],
    "⚠️ Risk & Volatility": [
        "annual_volatility_pct", "max_drawdown_pct", "avg_drawdown_pct",
    ],
    "🎯 Risk-Adjusted Returns": [
        "sharpe_ratio", "gross_sharpe_ratio", "turnover_adjusted_sharpe",
        "sharpe_drag", "sortino_ratio",  # <-- ADDED
    ],
    "💰 Fee & Cost Analysis": [
        "total_fees_currency", "fee_drag_ratio", "cost_efficiency",
        "total_fee_drag_pct", "annualized_fee_drag_pct",
    ],
    "🔄 Trade Activity": [
        "total_turnover", "avg_daily_turnover", "win_rate_pct",
    ],
    "📊 Distribution & Tail Risk": [
        "lower_tail", "upper_tail", "tail_risk",
        "skew", "kurtosis",  # <-- ADDED
    ],
    "📋 Other": [
        "gross_pnl", "net_pnl", "gross_return",
    ],
}

METRIC_FORMATTERS = {
    "total_return_pct": lambda v: f"{v:+.2f}%",
    "cagr_pct": lambda v: f"{v:+.2f}%",
    "gross_return_pct": lambda v: f"{v:+.2f}%",
    "annual_volatility_pct": lambda v: f"{v:+.2f}%",
    "max_drawdown_pct": lambda v: f"{v:+.2f}%",
    "avg_drawdown_pct": lambda v: f"{v:+.2f}%",
    "total_fee_drag_pct": lambda v: f"{v:+.2f}%",
    "annualized_fee_drag_pct": lambda v: f"{v:+.2f}%",
    "win_rate_pct": lambda v: f"{v:+.2f}%",
    "total_fees_currency": lambda v: f"${v:,.2f}",
    "gross_pnl": lambda v: f"{v:,.2f}",
    "net_pnl": lambda v: f"{v:,.2f}",
    "gross_return": lambda v: f"{v:.2f}",
    "sortino_ratio": lambda v: f"{v:.2f}",  # <-- ADDED
    "skew": lambda v: f"{v:.2f}",           # <-- ADDED
    "kurtosis": lambda v: f"{v:.2f}",       # <-- ADDED
}

METRIC_LABELS = {
    "total_return_pct": "Total Return %",
    "cagr_pct": "CAGR %",
    "gross_return_pct": "Gross Return %",
    "annual_volatility_pct": "Annual Vol %",
    "max_drawdown_pct": "Max Drawdown %",
    "avg_drawdown_pct": "Avg Drawdown %",
    "sharpe_ratio": "Sharpe Ratio",
    "gross_sharpe_ratio": "Gross Sharpe",
    "turnover_adjusted_sharpe": "Turnover-Adj Sharpe",
    "sharpe_drag": "Sharpe Drag",
    "sortino_ratio": "Sortino Ratio",     # <-- ADDED
    "total_fees_currency": "Total Fees ($)",
    "fee_drag_ratio": "Fee Drag Ratio",
    "cost_efficiency": "Cost Efficiency",
    "total_fee_drag_pct": "Total Fee Drag %",
    "annualized_fee_drag_pct": "Annual Fee Drag %",
    "total_turnover": "Total Turnover",
    "avg_daily_turnover": "Avg Daily Turnover",
    "win_rate_pct": "Win Rate %",
    "lower_tail": "Lower Tail Ratio",
    "upper_tail": "Upper Tail Ratio",
    "tail_risk": "Tail Risk (Geo Mean)",
    "skew": "Skew",                       # <-- ADDED
    "kurtosis": "Kurtosis",               # <-- ADDED
    "gross_pnl": "Gross Pnl",
    "net_pnl": "Net Pnl",
    "gross_return": "Gross Return",
}


# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def _get_metric_value(metrics: Any, key: str) -> Any:
    """Safely extract a metric value from a dataclass or dictionary."""
    if isinstance(metrics, dict):
        return metrics.get(key)
    return getattr(metrics, key, None)


def _format_metric_value(key: str, value: Any) -> str:
    """Format a metric value based on its key."""
    if value is None:
        return "N/A"
    formatter = METRIC_FORMATTERS.get(key)
    if formatter:
        return formatter(value)
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


# ==========================================
# 3. SINGLE STRATEGY SUMMARY
# ==========================================
def print_summary(strategy_name: str, metrics: PerformanceMetrics) -> None:
    """Print a clean, vertical summary for a single strategy."""
    print("\n" + "═" * 60)
    print(f"  STRATEGY SUMMARY: {strategy_name}")
    print("═" * 60)
    
    # Returns
    print(f"  Total Return:      {metrics.total_return_pct:+.2f}%")
    print(f"  CAGR:              {metrics.cagr_pct:+.2f}%")
    print(f"  Gross Return:      {metrics.gross_return_pct:+.2f}%")
    print("-" * 60)
    
    # Risk
    print(f"  Annual Volatility: {metrics.annual_volatility_pct:.2f}%")
    print(f"  Max Drawdown:      {metrics.max_drawdown_pct:+.2f}%")
    print(f"  Avg Drawdown:      {metrics.avg_drawdown_pct:+.2f}%")
    print("-" * 60)
    
    # Risk-Adjusted
    print(f"  Sharpe Ratio:      {metrics.sharpe_ratio:.2f}")
    print(f"  Gross Sharpe:      {metrics.gross_sharpe_ratio:.2f}")
    print(f"  Sortino Ratio:     {metrics.sortino_ratio:.2f}")
    print("-" * 60)
    
    # Fees
    print(f"  Total Fees:        ${metrics.total_fees_currency:,.2f}")
    print(f"  Fee Drag Ratio:    {metrics.fee_drag_ratio:.2f}")
    print("═" * 60 + "\n")


# ==========================================
# 4. BEAUTIFUL COMPARISON TABLE
# ==========================================
def print_comparison_table(
    title: str,
    strategies: List[Tuple[str, Any]],
) -> None:
    if not strategies:
        return
    
    # 1. Build the data matrix
    all_keys = set()
    for _, metrics in strategies:
        if isinstance(metrics, dict):
            all_keys.update(metrics.keys())
        else:
            for field in metrics.__dataclass_fields__:
                all_keys.add(field)
    
    # Build ordered rows based on category definitions
    ordered_rows = []
    for category, keys in METRIC_CATEGORIES.items():
        for key in keys:
            if key in all_keys:
                ordered_rows.append((category, key))
    
    # Fallback for uncategorized metrics
    categorized_keys = {k for _, keys in METRIC_CATEGORIES.items() for k in keys}
    for key in sorted(all_keys - categorized_keys):
        ordered_rows.append(("Other", key))
    
    # 2. Calculate column widths (FIXED)
    max_label_len = max(len(METRIC_LABELS.get(key, key)) for _, key in ordered_rows)
    max_label_len = max(max_label_len, len("Metric"))
    
    strategy_names = [name for name, _ in strategies]
    col_widths = []  # <-- FIX: Start with an empty list
    
    # Calculate the exact width needed for each strategy column
    for i, (_, metrics) in enumerate(strategies):
        # Start with the length of the strategy name itself
        max_val_len = len(strategy_names[i])
        for _, key in ordered_rows:
            val = _get_metric_value(metrics, key)
            formatted = _format_metric_value(key, val)
            max_val_len = max(max_val_len, len(formatted))
        col_widths.append(max_val_len)
    
    label_col = max_label_len + 2
    data_cols = [w + 2 for w in col_widths]
    total_width = label_col + sum(data_cols)
    
    # 3. Helper functions for drawing
    def draw_line(left, mid, right, fill="─"):
        parts = [left + fill * label_col]
        for w in data_cols:
            parts.append(fill * w)
        return mid.join(parts) + right

    def draw_row(cells, left="│", mid="│", right="│"):
        parts = [left + cells[0].ljust(label_col - 1) + " "]
        for i, cell in enumerate(cells[1:]):
            parts.append(cell.rjust(data_cols[i] - 1) + " ")  # <-- FIX: Use 'i' instead of 'i + 1'
        return mid.join(parts) + right
    
    # 4. Print the table
    print()
    print(draw_line("", "┬", "┐"))
    
    # Title row
    title_cell = f" {title} ".center(total_width - 2)
    print("│" + title_cell + "│")
    
    print(draw_line("├", "┼", "┤"))
    
    # Header row
    header_cells = ["Metric"] + strategy_names
    print(draw_row(header_cells, "│", "│", "│"))
    
    print(draw_line("├", "┼", "┤"))
    
    # Data rows with category headers
    current_category = None
    for category, key in ordered_rows:
        if category != current_category:
            if current_category is not None:
                print(draw_line("├", "┼", "┤"))
            
            # Category header spanning the whole row
            cat_content = f" {category} "
            print(f"│{cat_content.ljust(total_width - 2)}│")
            
            print(draw_line("├", "┼", "┤"))
            current_category = category
        
        label = METRIC_LABELS.get(key, key)
        values = []
        for _, metrics in strategies:
            val = _get_metric_value(metrics, key)
            values.append(_format_metric_value(key, val))
        
        print(draw_row([label] + values, "│", "│", "│"))
    
    print(draw_line("└", "┴", "┘"))
    print()


def print_report_comparison(
    title: str,
    reports: List[Tuple[str, Any]],
) -> None:
    """
    Wrapper that extracts metrics from BacktestReport objects.
    """
    strategies = [(name, report.metrics) for name, report in reports]
    print_comparison_table(title, strategies)


# ==========================================
# 5. PORTFOLIO REPORTS
# ==========================================
def print_portfolio_diversification(portfolio_result: PortfolioExecutionResult) -> None:
    """Prints the correlation matrix and diversification benefits."""
    print("\n" + "═" * 60)
    print("  PORTFOLIO DIVERSIFICATION ANALYSIS")
    print("═" * 60)
    
    print(f"\n  Diversification Ratio:    {portfolio_result.diversification_ratio:>8.2f}")
    print(f"  Volatility Reduction:     {portfolio_result.volatility_reduction_pct:>7.1f}%")
    
    print("\n  Correlation Matrix:")
    print("─" * 60)
    corr_str = portfolio_result.correlation_matrix.to_string(float_format="%.2f")
    for line in corr_str.split('\n'):
        print(f"  {line}")
    print("═" * 60 + "\n")


def print_portfolio_comparison(
    portfolio_metrics: PerformanceMetrics,
    individual_metrics: List[Tuple[str, PerformanceMetrics]],
) -> None:
    """
    Compares individual strategies against the portfolio.
    Accepts already-computed PerformanceMetrics (see PerformanceAnalyzer).
    """
    strategies = list(individual_metrics)
    strategies.append(("PORTFOLIO", portfolio_metrics))

    print_comparison_table("Strategy vs Portfolio Comparison", strategies)