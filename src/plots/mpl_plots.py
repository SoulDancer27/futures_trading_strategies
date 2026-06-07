"""
Matplotlib visualization module for backtest results.
"""
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from ..engine.vectorized import BacktestResult

def plot_results(
    result: BacktestResult, 
    save_to_file: bool = False, 
    output_path: str = None
):
    """
    Plot backtest results.
    
    Args:
        result: BacktestResult object
        save_to_file: If True, saves plot instead of showing interactively
        output_path: Optional custom file path. Defaults to outputs/backtest_<timestamp>.png
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, gridspec_kw={"hspace": 0.05})
    fig.suptitle("Backtest Results", fontsize=14, fontweight="bold")
    
    # 1. Equity Curve
    ax1 = axes[0]
    ax1.plot(result.equity.index, result.equity, label="Equity", color="blue", alpha=0.8)
    ax1.set_ylabel("Capital")
    ax1.grid(True, alpha=0.3)
    
    # 2. Positions
    ax2 = axes[1]
    ax2.bar(result.positions.index, result.positions, color="gray", alpha=0.7, width=1)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_ylabel("Contracts")
    ax2.grid(True, alpha=0.3)
    
    # 3. Drawdown
    ax3 = axes[2]
    drawdown = (result.equity - result.equity.cummax()) / result.equity.cummax() * 100
    ax3.fill_between(drawdown.index, drawdown, 0, color="red", alpha=0.3)
    ax3.set_ylabel("Drawdown %")
    ax3.grid(True, alpha=0.3)
    
    plt.xlabel("Date")
    plt.tight_layout()
    
    if save_to_file:
        if output_path is None:
            output_path = Path("outputs") / f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"📊 Plot saved to: {output_path}")
    else:
        plt.show()
        
    plt.close()