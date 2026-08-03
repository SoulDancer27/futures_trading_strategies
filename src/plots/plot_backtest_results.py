"""
Matplotlib visualization module for backtest results.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from pathlib import Path
from datetime import datetime
from typing import Union, List, Dict, Optional
from ..engine.vectorized import BacktestResult  # Adjust import path if necessary

def plot_backtest_results(
    results: Union[BacktestResult, List[BacktestResult], Dict[str, BacktestResult]],
    panels: List[str] = ['equity', 'positions', 'drawdown'],
    data: pd.DataFrame = None,
    volatility_window: int = 252,
    realized_vol_window: int = 21,
    save_to_file: bool = False,
    plot_pct: bool = False,
    output_path: Optional[str] = None,
    colors: Optional[List[str]] = None,
    figsize_base: tuple = (14, 3.5),
    multiplier: float = 1.0,
    use_fixed_capital: bool = True  # <--- NEW PARAMETER
):
    """
    Plot backtest results with dynamic panel selection and multi-strategy support.
    Uses cumulative_fees & cumulative_turnover directly from BacktestResult.
    
    Args:
        use_fixed_capital: If True, calculates returns and leverage relative to 
                           the initial capital (Carver's method). If False, uses 
                           compounding returns relative to the growing equity curve.
    """
    valid_panels = ['price', 'equity', 'positions', 'est_vol', 'realized_vol', 
                    'drawdown', 'leverage', 'cumulative_turnover', 'cumulative_fees']
    selected_panels = [p for p in panels if p in valid_panels]
    if not selected_panels:
        raise ValueError(f"No valid panels selected. Choose from: {valid_panels}")

    # 1️⃣ Normalize input
    if isinstance(results, BacktestResult):
        results_list = [(results.strategy_name or "Strategy", results)]
    elif isinstance(results, dict):
        results_list = [(name if name else f"Strategy {i+1}", res) for i, (name, res) in enumerate(results.items())]
    else:
        results_list = [(r.strategy_name or f"Strategy {i+1}", r) for i, r in enumerate(results)]

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

    # 2️⃣ Pre-calculate & load stored series
    strat_returns = {}
    real_vol = {}
    drawdown = {}
    leverage = {}
    cum_turnover = {}
    cum_fees = {}
    
    for name, res in results_list:
        initial_capital = res.equity.iloc[0]
        
        # --- FIX: Calculate returns based on capital assumption ---
        if use_fixed_capital:
            # Fixed Capital: Daily PnL / Initial Capital
            rets = res.daily_pnl / initial_capital
        else:
            # Compounding: Percentage change of equity
            rets = res.equity.pct_change()
        # ----------------------------------------------------------
        
        strat_returns[name] = rets
        real_vol[name] = rets.rolling(window=realized_vol_window, min_periods=1).std() * np.sqrt(252)
        drawdown[name] = (res.equity - res.equity.cummax()) / res.equity.cummax() * 100
        
        # --- FIX: Calculate leverage based on capital assumption ---
        if data is not None and 'close' in data.columns:
            notional = res.positions * data['close'] * multiplier
            
            if use_fixed_capital:
                # Fixed Capital: Leverage relative to constant initial capital
                leverage[name] = (notional / initial_capital).fillna(0).clip(lower=0)
            else:
                # Compounding: Leverage relative to growing equity curve
                leverage[name] = (notional / res.equity.replace(0, np.nan)).fillna(0).clip(lower=0)
        else:
            leverage[name] = pd.Series(0, index=res.equity.index)
        # -----------------------------------------------------------
            
        cum_turnover[name] = getattr(res, 'cumulative_turnover', None)
        if cum_turnover[name] is None:
            cum_turnover[name] = pd.Series(0, index=res.equity.index)
            
        cum_fees[name] = getattr(res, 'cumulative_fees', None)
        if cum_fees[name] is None:
            cum_fees[name] = pd.Series(0, index=res.equity.index)

    price_series = None
    est_vol = None
    if any(p in selected_panels for p in ['price', 'est_vol', 'leverage']):
        if data is None or 'close' not in data.columns:
            raise ValueError("'data' DataFrame with 'close' column is required for price, vol, or leverage panels")
        price_series = data['close']
        est_vol = data['close'].pct_change().rolling(window=volatility_window, min_periods=1).std() * np.sqrt(252)

    # 3️⃣ Plot Panels
    for i, panel in enumerate(selected_panels):
        ax = axes[i]
        
        if panel == 'price' and price_series is not None:
            ax.plot(price_series.index, price_series, label='Price', color='black', linewidth=1.2, alpha=0.7)
            
        elif panel == 'est_vol' and est_vol is not None:
            ax.plot(est_vol.index, est_vol, label=f'Est. Vol ({volatility_window}d)', color='tab:orange', linewidth=1.5)

        for j, (name, res) in enumerate(results_list):
            color = colors[j % len(colors)]
            
            if panel == 'equity':
                if plot_pct:
                    eq = (res.equity / res.equity.iloc[0] - 1) * 100
                    final_val = eq.iloc[-1]
                    label = f"{name} ({final_val:+.1f}%)"
                else:
                    eq = res.equity
                    final_val = eq.iloc[-1]
                    label = f"{name} (${final_val:,.0f})"
                ax.plot(eq.index, eq, label=label, color=color, linewidth=1.5, alpha=0.85)
                
            elif panel == 'positions':
                ax.plot(res.positions.index, res.positions, label=name, color=color, 
                        linewidth=1.2, drawstyle='steps-post', alpha=0.8)
                        
            elif panel == 'realized_vol':
                ax.plot(real_vol[name].index, real_vol[name], label=name, color=color, linewidth=1.5, alpha=0.85)
                
            elif panel == 'drawdown':
                dd = drawdown[name]
                ax.plot(dd.index, dd, label=name, color=color, linewidth=1.2, alpha=0.8)
                if n_strategies == 1:
                    ax.fill_between(dd.index, dd, 0, color='red', alpha=0.15)
                    
            elif panel == 'leverage':
                lev = leverage[name]
                ax.plot(lev.index, lev, label=name, color=color, linewidth=1.5, alpha=0.85)
                ax.fill_between(lev.index, lev, 1.5, where=lev > 1.5, color=color, alpha=0.1)
                
            elif panel == 'cumulative_turnover':
                ax.plot(cum_turnover[name].index, cum_turnover[name], label=name, color=color, linewidth=1.5, alpha=0.85)
                
            elif panel == 'cumulative_fees':
                ax.plot(cum_fees[name].index, cum_fees[name], label=name, color=color, linewidth=1.5, alpha=0.85)

        # Add initial capital reference line for absolute equity view (once per panel)
        if panel == 'equity' and not plot_pct:
            ax.axhline(results_list[0][1].equity.iloc[0], color='gray', linestyle=':', linewidth=0.5, alpha=0.3)

        #  Panel Formatting & Y-Axis Tickers
        ax.set_title(panel.replace('_', ' ').title(), loc='left', fontweight='bold', fontsize=11, pad=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=9, framealpha=0.9, ncol=1 if n_strategies <= 3 else 2)
        
        if panel == 'positions':
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0f}'))
        elif panel in ['drawdown']:
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.1f}%'))
        elif panel in ['est_vol', 'realized_vol']:
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y*100:.1f}%'))
        elif panel == 'leverage':
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.2f}x'))
            ax.axhline(1.0, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
        elif panel == 'cumulative_fees':
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'${y:,.0f}'))
        elif panel == 'cumulative_turnover':
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0f}'))
        elif panel == 'equity':
            if plot_pct:
                ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.1f}%'))
            else:
                ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'${y:,.0f}'))
        else:
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:,.0f}'))

    axes[-1].set_xlabel("Date")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # 4️⃣ Save or Show
    if save_to_file:
        if output_path is None:
            output_path = Path("outputs") / f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"📊 Plot saved to: {output_path}")
    else:
        plt.show()
    plt.close()