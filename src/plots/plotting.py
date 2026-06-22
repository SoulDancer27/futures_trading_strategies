"""
Matplotlib visualization module for backtest results.
"""
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from datetime import datetime
from typing import Union, List, Dict, Optional
import pandas as pd
from ..engine.vectorized import BacktestResult

def plot_results(
    results: Union[BacktestResult, List[BacktestResult], Dict[str, BacktestResult]],
    save_to_file: bool = False,
    plot_pct: bool = False,
    output_path: Optional[str] = None,
    colors: Optional[Union[List[str], Dict[str, str]]] = None,
    figsize: tuple = (12, 9)
):
    """
    Plot backtest results. Supports single or multiple strategies side-by-side.
    
    Args:
        results: Single BacktestResult, list of results, or dict mapping {name: result}
        save_to_file: If True, saves plot instead of showing interactively
        plot_pct: If True, displays equity curve as % return instead of absolute value
        output_path: Optional custom file path. Defaults to outputs/backtest_<timestamp>.png
        colors: List of colors or dict mapping {strategy_name: color}. Auto-generated if None.
        figsize: Figure dimensions (width, height)
    """
    # 1. Normalize input to list of (name, result)
    if isinstance(results, BacktestResult):
        results_list = [(results.strategy_name or "Strategy 1", results)]
    elif isinstance(results, dict):
        results_list = [(name, res) for name, res in results.items()]
    else:
        results_list = [(res.strategy_name or f"Strategy {i+1}", res) for i, res in enumerate(results)]

    n_strategies = len(results_list)
    
    # 2. Resolve colors
    if colors is None:
        colors = [plt.cm.tab10.colors[i % 10] for i in range(n_strategies)]
    elif isinstance(colors, dict):
        colors = [colors.get(name, plt.cm.tab10(i % 10)) for i, (name, _) in enumerate(results_list)]
    else:
        colors = list(colors) + [plt.cm.tab10.colors[i % 10] for i in range(len(colors), n_strategies)]

    # 3. Setup figure
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True, gridspec_kw={"hspace": 0.08})
    fig.suptitle("Backtest Results" if n_strategies == 1 else "Strategy Comparison", 
                 fontsize=14, fontweight="bold")

    for i, (name, res) in enumerate(results_list):
        color = colors[i]
        initial_capital = res.equity.iloc[0]

        # Panel 1: Equity Curve
        equity_plot = (res.equity / initial_capital - 1) * 100 if plot_pct else res.equity
        axes[0].plot(equity_plot.index, equity_plot, label=name, color=color, linewidth=1.5, alpha=0.85)

        # Panel 2: Positions (step plot for clean multi-strategy overlay)
        axes[1].plot(res.positions.index, res.positions, label=name, color=color, 
                     linewidth=1.2, drawstyle='steps-post', alpha=0.8)

        # Panel 3: Drawdown
        dd = (res.equity - res.equity.cummax()) / res.equity.cummax() * 100
        axes[2].plot(dd.index, dd, label=name, color=color, linewidth=1.2, alpha=0.8)
        
        # Light red fill for drawdown region (only if single strategy to avoid overlap clutter)
        if n_strategies == 1:
            axes[2].fill_between(dd.index, dd, 0, color="red", alpha=0.15)

    # 4. Formatting & Axes Configuration
    # Equity baseline
    baseline = 0 if plot_pct else initial_capital
    axes[0].axhline(baseline, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    axes[0].set_ylabel("Return (%)" if plot_pct else "Capital")
    if plot_pct:
        axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}%"))

    # Positions baseline
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].set_ylabel("Contracts")

    # Drawdown baseline
    axes[2].axhline(0, color="black", linewidth=0.5)
    axes[2].set_ylabel("Drawdown %")
    axes[2].axhspan(-100, 0, color="red", alpha=0.03, zorder=0)  # Background shading

    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", framealpha=0.9, fontsize=9)

    axes[-1].set_xlabel("Date")
    fig.autofmt_xdate()
    plt.tight_layout()

    # 5. Save or Show
    if save_to_file:
        if output_path is None:
            output_path = Path("outputs") / f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"📊 Plot saved to: {output_path}")
    else:
        plt.show()
    plt.close()