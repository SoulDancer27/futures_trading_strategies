"""
Analysis utilities for comparing backtest results.
Metrics are grouped into logical categories for clearer comparison.
"""
import pandas as pd
from typing import Union, List, Dict, Optional
from ..engine.vectorized import BacktestResult

# 🔑 Logical metric categories (in display order)
METRIC_CATEGORIES = {
    "📈 Returns & Performance": [
        'total_return_pct',
        'cagr_pct', 
        'gross_return_pct',
    ],
    "⚠️ Risk & Volatility": [
        'annual_volatility_pct',
        'max_drawdown_pct',
        'avg_drawdown_pct',
    ],
    "🎯 Risk-Adjusted Returns": [
        'sharpe_ratio',
        'gross_sharpe_ratio',
        'turnover_adjusted_sharpe',
        'sharpe_drag',
    ],
    "💰 Fee & Cost Analysis": [
        'total_fees_currency',
        'fee_drag_ratio',
        'cost_efficiency',
        'total_fee_drag_pct',
        'annualized_fee_drag_pct',
    ],
    "🔄 Trade Activity": [
        'total_turnover',
        'avg_daily_turnover',
        'win_rate_pct',
    ],
    "📊 Distribution & Tail Risk": [
        'skew',
        'lower_tail',
        'upper_tail',
        'tail_risk',
    ],
}

# Fallback label map for clean display names
LABEL_MAP = {
    'total_return_pct': 'Total Return %',
    'cagr_pct': 'CAGR %',
    'gross_return_pct': 'Gross Return %',
    'annual_volatility_pct': 'Annual Vol %',
    'max_drawdown_pct': 'Max Drawdown %',
    'avg_drawdown_pct': 'Avg Drawdown %',
    'sharpe_ratio': 'Sharpe Ratio',
    'gross_sharpe_ratio': 'Gross Sharpe',
    'turnover_adjusted_sharpe': 'Turnover-Adj Sharpe',
    'sharpe_drag': 'Sharpe Drag',
    'total_fees_currency': 'Total Fees ($)',
    'fee_drag_ratio': 'Fee Drag Ratio',
    'cost_efficiency': 'Cost Efficiency',
    'total_fee_drag_pct': 'Total Fee Drag %',
    'annualized_fee_drag_pct': 'Annual Fee Drag %',
    'total_turnover': 'Total Turnover',
    'avg_daily_turnover': 'Avg Daily Turnover',
    'win_rate_pct': 'Win Rate %',
    'skew': 'Skew',
    'lower_tail': 'Lower Tail Ratio',
    'upper_tail': 'Upper Tail Ratio',
    'tail_risk': 'Tail Risk (Geo Mean)',
}


def compare_strategies(
    results: Union[BacktestResult, List[BacktestResult], Dict[str, BacktestResult]],
    metrics: Optional[List[str]] = None,
    round_decimals: int = 2,
    group_by_category: bool = True
) -> pd.DataFrame:
    """
    Compare performance metrics across strategy results.
    Metrics are grouped into logical categories for clearer comparison.
    
    Args:
        results: BacktestResult, List, or Dict[str, BacktestResult]
        metrics: Optional list of metric keys. If None, auto-discovers all.
        round_decimals: Decimal places for rounding
        group_by_category: If True, orders metrics by logical categories
        
    Returns:
        DataFrame with strategies as columns and metrics as rows (ordered)
    """
    # 1️⃣ Normalize input to dict {name: result}
    if isinstance(results, BacktestResult):
        results_dict = {results.strategy_name or "Strategy": results}
    elif isinstance(results, dict):
        results_dict = results
    elif hasattr(results, '__iter__') and not hasattr(results, 'metrics'):
        results_dict = {getattr(r, 'strategy_name', f"Strategy {i+1}"): r for i, r in enumerate(results)}
    else:
        results_dict = {getattr(results, 'strategy_name', "Strategy"): results}
        
    if not results_dict:
        raise ValueError("No results provided to compare.")
    
    # 2️⃣ Discover or use specified metrics
    if metrics is None:
        all_keys = []
        for res in results_dict.values():
            all_keys.extend(res.metrics.keys())
        available_metrics = list(dict.fromkeys(all_keys))  # Dedupe, preserve order
    else:
        available_metrics = metrics

    # 3️⃣ Sort metrics by category (if enabled)
    if group_by_category:
        ordered_metrics = []
        remaining = set(available_metrics)
        
        for category, preferred_order in METRIC_CATEGORIES.items():
            for m in preferred_order:
                if m in remaining:
                    ordered_metrics.append(m)
                    remaining.remove(m)
        
        # Append any uncategorized metrics at the end
        ordered_metrics.extend(sorted(remaining))
        selected_metrics = ordered_metrics
    else:
        selected_metrics = available_metrics

    # 4️⃣ Extract & format metrics
    data = {}
    for name, res in results_dict.items():
        row = {}
        for m in selected_metrics:
            val = res.metrics.get(m)
            if val is None:
                row[m] = "N/A"
            elif isinstance(val, float):
                if m.endswith('_pct'):
                    row[m] = f"{val:+.{round_decimals}f}%"
                elif m.endswith('_currency'):
                    row[m] = f"${val:,.{round_decimals}f}"
                elif any(k in m for k in ['turnover', 'contracts', 'trades', 'days']):
                    row[m] = f"{val:,.{round_decimals}f}"
                else:
                    row[m] = f"{val:.{round_decimals}f}"
            else:
                row[m] = str(val)
        data[name] = row
        
    # 5️⃣ Create DataFrame with clean labels
    df = pd.DataFrame(data)
    df.index = [LABEL_MAP.get(m, m.replace('_', ' ').replace('pct', '%').title()) for m in df.index]
    
    return df


def print_comparison_table(
    results: Union[BacktestResult, List[BacktestResult], Dict[str, BacktestResult]],
    metrics: Optional[List[str]] = None,
    title: str = "Strategy Comparison",
    group_by_category: bool = True,
    show_category_headers: bool = True
) -> None:
    """
    Print a formatted ASCII comparison table with logical metric grouping.
    
    Args:
        results: BacktestResult, List, or Dict[str, BacktestResult]
        metrics: Optional list of metric keys
        title: Table title
        group_by_category: If True, groups metrics by logical categories
        show_category_headers: If True, prints category section headers in output
    """
    df = compare_strategies(results, metrics=metrics, group_by_category=group_by_category)
    
    if df.empty:
        print("⚠️ No data to display")
        return
    
    # Calculate column widths
    col_widths = [max(len(str(col)), max(len(str(v)) for v in df[col])) for col in df.columns]
    idx_width = max(len(str(idx)) for idx in df.index)
    header_width = idx_width + sum(w + 3 for w in col_widths) + 2
    
    print("┌" + "─" * header_width + "┐")
    print(f"│ {title:^{header_width}} │")
    print("├" + "─" * (idx_width + 2) + "┼" + "┼".join(["─" * (w + 2) for w in col_widths]) + "┤")
    
    # Header row
    header = f"│ {'Metric':<{idx_width}} │"
    for i, col in enumerate(df.columns):
        header += f" {col:<{col_widths[i]}} │"
    print(header)
    print("├" + "─" * (idx_width + 2) + "┼" + "┼".join(["─" * (w + 2) for w in col_widths]) + "┤")
    
    # Data rows with optional category headers
    current_category = None
    for idx, row in df.iterrows():
        # Print category header if enabled and we crossed a boundary
        if show_category_headers and group_by_category:
            category = _get_metric_category(idx, LABEL_MAP)
            if category != current_category:
                if current_category is not None:
                    print("├" + "─" * (idx_width + 2) + "┼" + "┼".join(["─" * (w + 2) for w in col_widths]) + "┤")
                print(f"│ {category:<{idx_width}} │" + " " * (header_width - idx_width - 2) + "│")
                current_category = category
        
        line = f"│ {idx:<{idx_width}} │"
        for i, col in enumerate(df.columns):
            line += f" {str(row[col]):<{col_widths[i]}} │"
        print(line)
    
    print("└" + "─" * header_width + "┘")


def _get_metric_category(metric_label: str, label_map: Dict[str, str]) -> str:
    """Helper: map a display label back to its category."""
    # Reverse lookup: find which category contains this metric's original key
    original_key = None
    for key, label in label_map.items():
        if label == metric_label:
            original_key = key
            break
    
    if original_key is None:
        return "📋 Other"
    
    for category, keys in METRIC_CATEGORIES.items():
        if original_key in keys:
            return category
    
    return "📋 Other"