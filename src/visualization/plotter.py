"""
Modular Visualization Module for Backtest Results.
Uses a Panel Registry pattern for clean, extensible plotting.
Strictly "dumb": reads pre-calculated series from ExecutionResult.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from pathlib import Path
from datetime import datetime
from typing import Union, List, Dict, Optional, Callable, Tuple

from ..core.models import ExecutionResult, RegressionResult

# ==========================================
# 1. DATA PREPARATION
# ==========================================
def prepare_plot_data(
    results_list: List[Tuple[str, ExecutionResult]]
) -> Dict[str, Dict[str, pd.Series]]:
    """
    Extracts all necessary pre-calculated series from ExecutionResult.
    Returns a nested dictionary: { strategy_name: { 'equity': ..., 'drawdown': ... } }
    """
    plot_data = {}
    
    for name, res in results_list:
        plot_data[name] = {
            'equity': res.equity,
            'positions': res.positions,
            'drawdown': res.drawdown,
            'leverage': res.leverage,
            'cumulative_fees': res.cumulative_fees,
            'cumulative_turnover': res.cumulative_turnover,
            'realized_vol': res.realized_vol,
            'price': res.asset.price_data if res.asset else None
        }
        
    return plot_data


# ==========================================
# 2. PANEL RENDERERS (The Modules)
# ==========================================
def _render_equity(ax, data, names, colors, plot_pct=False, **kwargs):
    for name, color in zip(names, colors):
        eq = data[name]['equity']
        if plot_pct:
            eq = (eq / eq.iloc[0] - 1) * 100
            label = f"{name} ({eq.iloc[-1]:+.1f}%)"
        else:
            label = f"{name} (${eq.iloc[-1]:,.0f})"
        ax.plot(eq.index, eq, label=label, color=color, linewidth=1.5, alpha=0.85)
    
    if not plot_pct:
        ax.axhline(data[names[0]]['equity'].iloc[0], color='gray', linestyle=':', linewidth=0.5, alpha=0.3)

def _render_drawdown(ax, data, names, colors, **kwargs):
    for name, color in zip(names, colors):
        dd = data[name]['drawdown']
        ax.plot(dd.index, dd, label=name, color=color, linewidth=1.2, alpha=0.8)
        if len(names) == 1:
            ax.fill_between(dd.index, dd, 0, color='red', alpha=0.15)

def _render_leverage(ax, data, names, colors, **kwargs):
    for name, color in zip(names, colors):
        lev = data[name]['leverage']
        ax.plot(lev.index, lev, label=name, color=color, linewidth=1.5, alpha=0.85)
        ax.fill_between(lev.index, lev, 1.5, where=lev > 1.5, color=color, alpha=0.1)
    ax.axhline(1.0, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)

def _render_positions(ax, data, names, colors, **kwargs):
    for name, color in zip(names, colors):
        ax.plot(data[name]['positions'].index, data[name]['positions'], label=name, color=color, 
                linewidth=1.2, drawstyle='steps-post', alpha=0.8)

def _render_realized_vol(ax, data, names, colors, **kwargs):
    for name, color in zip(names, colors):
        ax.plot(data[name]['realized_vol'].index, data[name]['realized_vol'], label=name, color=color, linewidth=1.5, alpha=0.85)

def _render_cumulative_fees(ax, data, names, colors, **kwargs):
    for name, color in zip(names, colors):
        ax.plot(data[name]['cumulative_fees'].index, data[name]['cumulative_fees'], label=name, color=color, linewidth=1.5, alpha=0.85)

def _render_cumulative_turnover(ax, data, names, colors, **kwargs):
    for name, color in zip(names, colors):
        ax.plot(data[name]['cumulative_turnover'].index, data[name]['cumulative_turnover'], label=name, color=color, linewidth=1.5, alpha=0.85)

def _render_price(ax, data, names, colors, **kwargs):
    price = data[names[0]]['price']
    if price is not None:
        ax.plot(price.index, price, label='Price', color='black', linewidth=1.2, alpha=0.7)
    else:
        ax.text(0.5, 0.5, 'No price data available', transform=ax.transAxes, ha='center')


# Registry mapping panel names to their rendering functions
PANEL_RENDERERS: Dict[str, Callable] = {
    'equity': _render_equity,
    'drawdown': _render_drawdown,
    'leverage': _render_leverage,
    'positions': _render_positions,
    'realized_vol': _render_realized_vol,
    'cumulative_fees': _render_cumulative_fees,
    'cumulative_turnover': _render_cumulative_turnover,
    'price': _render_price
}

# Y-axis formatters for each panel
PANEL_FORMATTERS: Dict[str, Callable] = {
    'positions': lambda y, _: f'{y:.0f}',
    'drawdown': lambda y, _: f'{y:.1f}%',
    'realized_vol': lambda y, _: f'{y:.1f}%',
    'leverage': lambda y, _: f'{y:.2f}x',
    'cumulative_fees': lambda y, _: f'${y:,.0f}',
    'cumulative_turnover': lambda y, _: f'{y:.0f}',
    'equity_pct': lambda y, _: f'{y:.1f}%',
    'equity_abs': lambda y, _: f'${y:,.0f}'
}


# ==========================================
# 3. THE ORCHESTRATOR
# ==========================================
def plot_backtest_results(
    results: Union[ExecutionResult, List[ExecutionResult], Dict[str, ExecutionResult]],
    panels: List[str] = ['equity', 'drawdown', 'leverage', 'cumulative_fees', 'realized_vol'],
    save_to_file: bool = False,
    plot_pct: bool = False,
    output_path: Optional[str] = None,
    colors: Optional[List[str]] = None,
    figsize_base: tuple = (14, 3.5)
):
    """
    Main entry point for plotting backtest results.
    """
    # 1. Normalize Input
    if isinstance(results, ExecutionResult):
        results_list = [(results.strategy_name or "Strategy", results)]
    elif isinstance(results, dict):
        results_list = [(name, res) for name, res in results.items()]
    else:
        results_list = [(r.strategy_name or f"Strategy {i+1}", r) for i, r in enumerate(results)]

    # 2. Validate Panels
    valid_panels = list(PANEL_RENDERERS.keys())
    selected_panels = [p for p in panels if p in valid_panels]
    if not selected_panels:
        raise ValueError(f"No valid panels selected. Choose from: {valid_panels}")

    # 3. Setup Figure
    n_panels = len(selected_panels)
    n_strategies = len(results_list)
    fig_height = (figsize_base[0], figsize_base[1] * n_panels)

    fig, axes = plt.subplots(n_panels, 1, figsize=fig_height, sharex=True, 
                             gridspec_kw={"hspace": 0.16, "top": 0.96})
    if n_panels == 1:
        axes = [axes]

    if colors is None:
        colors = [plt.cm.tab10.colors[i % 10] for i in range(n_strategies)]
    else:
        colors = list(colors) + [plt.cm.tab10.colors[i % 10] for i in range(len(colors), n_strategies)]

    # 4. Prepare Data (Extracted once)
    plot_data = prepare_plot_data(results_list)
    strategy_names = [name for name, _ in results_list]

    # 5. Render Panels
    for i, panel in enumerate(selected_panels):
        ax = axes[i]
        
        # Dispatch to the correct renderer
        renderer = PANEL_RENDERERS[panel]
        renderer(ax, plot_data, strategy_names, colors, plot_pct=plot_pct)

        # Formatting
        ax.set_title(panel.replace('_', ' ').title(), loc='left', fontweight='bold', fontsize=11, pad=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=9, framealpha=0.9, ncol=1 if n_strategies <= 3 else 2)
        
        # Apply Y-axis formatter
        formatter_key = f"equity_pct" if panel == 'equity' and plot_pct else f"equity_abs" if panel == 'equity' else panel
        if formatter_key in PANEL_FORMATTERS:
            ax.yaxis.set_major_formatter(FuncFormatter(PANEL_FORMATTERS[formatter_key]))

    axes[-1].set_xlabel("Date")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # 6. Save or Show
    if save_to_file:
        if output_path is None:
            output_path = Path("logs") / f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"📊 Plot saved to: {output_path}")
    else:
        plt.show()
    plt.close()


def plot_regression(result: RegressionResult, ax=None, figsize=(8, 6)):
    """
    Scatter the strategy's monthly returns against the benchmark's monthly returns,
    overlaid with the fitted regression line (y = alpha + beta * x).

    Returns the matplotlib Axes for further customization.
    """
    if result.strategy_monthly is None or result.benchmark_monthly is None:
        raise ValueError(
            "RegressionResult has no monthly return series; run RegressionAnalyzer.analyze() first."
        )

    x = result.benchmark_monthly * 100.0
    y = result.strategy_monthly * 100.0

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    ax.scatter(x, y, s=25, alpha=0.6, edgecolors="none", label="Monthly returns")

    # Fitted line: y% = alpha% + beta * x%
    xline = np.linspace(x.min(), x.max(), 100)
    yline = result.alpha_monthly_pct + result.beta * xline
    ax.plot(
        xline,
        yline,
        color="red",
        linewidth=2,
        label=f"Fit: α={result.alpha_monthly_pct:+.2f}%/mo, β={result.beta:.2f}",
    )

    ax.axhline(0, color="gray", linewidth=0.5, alpha=0.5)
    ax.axvline(0, color="gray", linewidth=0.5, alpha=0.5)
    ax.set_xlabel(f"{result.benchmark_name} monthly return (%)")
    ax.set_ylabel(f"{result.strategy_name} monthly return (%)")
    ax.set_title(
        f"Benchmark Regression: {result.strategy_name} vs {result.benchmark_name}\n"
        f"R²={result.r_squared:.3f}   α t-stat={result.alpha_t_stat:.2f}"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax