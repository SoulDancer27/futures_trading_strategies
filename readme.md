# 📈 Standalone Vectorized Backtester

A minimal, modular Python framework for rapid futures & cash market backtesting. Designed for quantitative research, parameter sweeps, and strategy validation without database dependencies or complex visualization overhead.

> 🔗 **Companion to** [`financial_tools`](https://github.com/...): Strips away ORM/Plotly complexity while preserving the clean 3-layer architecture for fast, barebones testing.

---

## ✨ Features
- ⚡ **Pure vectorized execution** using `pandas`/`numpy` (no event loops)
- 📊 **Continuous position sizing** (contracts/shares per time step, no trade-pair matching)
- 📁 **Headerless CSV loader** for simple `[date, price]` series with automatic index normalization
- 💰 **Flexible transaction costs**: fixed per contract **or** decimal `commission_rate`
- 🎯 **Slippage modeling** via `slippage_rate` (% of trade value)
- 📈 **Advanced risk metrics**: Average Drawdown, Tail Ratios (fat-tail analysis), Skew, Sharpe, CAGR
- 💸 **Exact fee accounting**: Gross vs. net returns, fee drag ratio, Sharpe drag, turnover-adjusted Sharpe
- 🎨 **Matplotlib reporting** (equity curve, positions, drawdown, volatility, fees, leverage) with optional file output
- 🔍 **Multi-strategy comparison** with logical metric grouping and auto-formatted tables
- 🧩 **Strict modular architecture** (`data` → `strategies` → `engine` → `plots`)
- 📦 **Zero external dependencies** beyond core scientific Python stack

---

## 📁 Project Structure
```
.
├── src/
│   ├── data/              # CSV loaders & index normalization
│   │   └── loader.py
│   ├── strategies/        # BaseStrategy + implementations
│   │   ├── base.py
│   │   ├── buy_and_hold.py
│   │   ├── fixed_risk_position.py
│   │   └── vol_scaled_bnh.py
│   ├── engine/            # Vectorized execution (split for clarity)
│   │   ├── models.py      # BacktestResult dataclass
│   │   ├── metrics.py     # Performance & risk calculations
│   │   └── vectorized.py  # Core execution loop
│   ├── plots/             # Matplotlib visualization utilities
│   │   ├── mpl_plots.py   # plot_backtest_results()
│   │   └── analysis.py    # compare_strategies(), print_comparison_table()
│   └── __init__.py        # Clean top-level exports
├── data/                  # Place your headerless CSV files here
├── outputs/               # Auto-generated backtest plots
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install pandas numpy matplotlib
```

### 2. Prepare Data
Place a headerless CSV in `data/` with two columns: `date` and `price`.
```csv
2024-01-01,65.50
2024-01-02,66.20
2024-01-03,64.80
```
*(Supports any standard datetime format via the `date_format` parameter)*

### 3. Run a Backtest (Jupyter / Script)
```python
from src import (
    load_simple_price_csv,
    VectorizedEngine,
    FixedRiskPositionStrategy,
    plot_backtest_results,
    compare_strategies
)

# 1. Load data
data = load_simple_price_csv("data/sample.csv", date_format="%Y-%m-%d")

# 2. Configure strategy & engine
strategy = FixedRiskPositionStrategy(
    risk_target=0.15,           # 15% annualized vol target
    vol_method='ewma',          # Exponentially weighted volatility
    lambda_param=0.94,          # ~20-day equivalent decay
    use_fixed_capital=False     # Compound capital mode
)

engine = VectorizedEngine(
    initial_capital=100_000,
    point_value=1.0,            # 1.0 for stocks/spot, 50 for ES futures
    commission_rate=0.001,      # 0.1% of trade value
    slippage_rate=0.0005        # 0.05% slippage
)

# 3. Execute backtest
result = engine.run(strategy, data[['close']], strategy_name="VolTarget-EWMA")

# 4. View metrics & plot
result.print_metrics()
plot_backtest_results(
    result,
    panels=['equity', 'positions', 'realized_vol', 'cumulative_fees'],
    data=data,
    plot_pct=True,
    save_to_file=True
)
```

### 4. Compare Multiple Strategies
```python
# Run multiple backtests
results = {
    "FixedRisk-SMA": result_sma,
    "FixedRisk-EWMA": result_ewma,
    "Buy & Hold": result_bh
}

# Auto-discovers & groups metrics logically
df = compare_strategies(results)
display(df)

# Console output with category headers
from src.plots.analysis import print_comparison_table
print_comparison_table(results, title="Vol-Targeting Comparison")
```

---

## ⚙️ Configuration Reference

### Vectorized Engine (`src/engine/vectorized.py`)
| Parameter | Type | Default | Description |
|---|---|---|---|
| `initial_capital` | `float` | `100_000` | Starting portfolio value |
| `point_value` | `float` | `1.0` | Currency value per 1-point price move (`1.0` = cash, `50.0` = ES) |
| `commission_per_contract` | `float` | `None` | Fixed fee per contract/share traded |
| `commission_rate` | `float` | `None` | Decimal rate of trade value (e.g., `0.001` = 0.1%) |
| `slippage_rate` | `float` | `None` | Decimal rate of slippage (e.g., `0.0005` = 0.05%) |

*⚠️ Note: Specify either `commission_per_contract` OR `commission_rate`. Not both.*

### Strategies (`src/strategies/`)
All strategies inherit `BaseStrategy` and must implement:
```python
def generate_positions(self, data: pd.DataFrame) -> pd.Series:
    """Return continuous position size (positive=long, negative=short, 0=flat)"""
    pass
```
Positions are automatically shifted by 1 bar in the engine to prevent lookahead bias.

**Built-in Strategies:**
| Strategy | File | Key Parameters | Description |
|----------|------|---------------|-------------|
| Buy & Hold | `buy_and_hold.py` | `position_size` | Static long position |
| Fixed Risk Position | `fixed_risk_position.py` | `risk_target`, `vol_method`, `lambda_param` | Volatility-targeted sizing with SMA/EWMA estimation |
| Volatility-Scaled B&H | `vol_scaled_bnh.py` | `target_annual_vol`, `vol_method` | Simple vol-scaling for buy & hold |

### Data Loader (`src/data/loader.py`)
```python
load_simple_price_csv(
    file_path: str,
    delimiter: str = ",",
    date_format: str = None  # Optional pandas format string
) -> pd.DataFrame
```
Automatically handles: timezone stripping, duplicate date removal, index sorting, and column standardization.

---

## 📊 Output Metrics

### Standard Metrics (Net of Fees)
| Metric | Description |
|--------|-------------|
| `total_return_pct` | Total portfolio return (%) |
| `cagr_pct` | Compound Annual Growth Rate (%) |
| `annual_volatility_pct` | Annualized standard deviation of returns (%) |
| `sharpe_ratio` | Risk-adjusted return (CAGR / Volatility) |
| `max_drawdown_pct` | Worst peak-to-trough decline (%) |
| `avg_drawdown_pct` | Average of max drawdowns per distinct period (%) |
| `win_rate_pct` | % of days with positive P&L |
| `skew` | Return distribution asymmetry |

### Tail Risk Metrics (Fat-Tail Analysis)
| Metric | Interpretation |
|--------|---------------|
| `lower_tail` | Lower tail ratio vs. Gaussian (1.0 = normal, >1.0 = fatter) |
| `upper_tail` | Upper tail ratio vs. Gaussian |
| `tail_risk` | Geometric mean of both tails |

### Fee-Adjusted Metrics (Calculated Exactly in Engine)
| Metric | Description |
|--------|-------------|
| `gross_return_pct` | Return before deducting fees (%) |
| `total_fees_currency` | Total commissions + slippage paid ($) |
| `fee_drag_ratio` | Fees as % of gross edge (e.g., 0.027 = 2.7%) |
| `cost_efficiency` | % of gross return that survives costs (e.g., 0.973 = 97.3%) |
| `gross_sharpe_ratio` | Sharpe ratio before fees |
| `sharpe_drag` | Reduction in Sharpe due to fees |
| `turnover_adjusted_sharpe` | Sharpe penalized for excessive churn |
| `total_turnover` | Total contracts traded over backtest |
| `avg_daily_turnover` | Average daily trading activity |

**Tail Ratio Interpretation:**
- `1.0` = Normal Gaussian distribution
- `> 1.0` = Fatter tails (more extreme outliers than normal)
- Separates left (`lower_tail`) and right (`upper_tail`) tail behavior for precise risk profiling.

**Fee Efficiency Benchmarks:**
| Metric | Excellent | Acceptable | Concerning |
|--------|-----------|------------|------------|
| `fee_drag_ratio` | < 5% | 5–15% | > 15% |
| `cost_efficiency` | > 95% | 85–95% | < 85% |
| `sharpe_drag` | < 0.10 | 0.10–0.30 | > 0.30 |

---

## 🎨 Visualization (`src/plots/mpl_plots.py`)

### Dynamic Panel Selection
```python
plot_backtest_results(
    results=...,  # Single result, list, or dict
    panels=['equity', 'positions', 'drawdown'],  # Choose any combination
    data=data,    # Required for price/vol/leverage/fee panels
    plot_pct=True,  # Show equity as % return vs. absolute $
    save_to_file=True,
    output_path="outputs/my_plot.png"
)
```

**Available Panels:**
| Panel | Description | Requires `data` |
|-------|-------------|-----------------|
| `price` | Instrument price series | ✅ |
| `equity` | Portfolio equity curve | ❌ |
| `positions` | Position size over time | ❌ |
| `est_vol` | Estimated volatility (price-based) | ✅ |
| `realized_vol` | Realized volatility (equity-based) | ❌ |
| `drawdown` | Portfolio drawdown % | ❌ |
| `leverage` | Notional exposure / equity | ✅ |
| `cumulative_turnover` | Total contracts traded over time | ❌ |
| `cumulative_fees` | Total fees paid over time ($) | ✅ |

### Multi-Strategy Comparison
```python
# Compare 3 strategies with custom colors
plot_backtest_results(
    results={"SMA": r1, "EWMA": r2, "BNH": r3},
    panels=['equity', 'realized_vol', 'cumulative_fees'],
    colors=['#1f77b4', '#ff7f0e', '#2ca02c'],
    plot_pct=True
)
```

---

## 📋 Strategy Comparison (`src/plots/analysis.py`)

### Auto-Grouped Metrics
```python
df = compare_strategies({
    "FixedRisk-SMA": result_sma,
    "FixedRisk-EWMA": result_ewma
})
display(df)
```

**Metric Groups (in display order):**
1. 📈 Returns & Performance (`total_return_pct`, `cagr_pct`, `gross_return_pct`)
2. ⚠️ Risk & Volatility (`annual_volatility_pct`, `max_drawdown_pct`, `avg_drawdown_pct`)
3. 🎯 Risk-Adjusted Returns (`sharpe_ratio`, `gross_sharpe_ratio`, `turnover_adjusted_sharpe`, `sharpe_drag`)
4. 💰 Fee & Cost Analysis (`total_fees_currency`, `fee_drag_ratio`, `cost_efficiency`)
5. 🔄 Trade Activity (`total_turnover`, `avg_daily_turnover`, `win_rate_pct`)
6. 📊 Distribution & Tail Risk (`skew`, `lower_tail`, `upper_tail`, `tail_risk`)

### Console Output
```python
print_comparison_table(results, title="Vol-Targeting Comparison")
```
```
┌─────────────────────────────────────────────────────┐
│              Vol-Targeting Comparison                │
├────────────────────────────────┼────────┼────────┤
│ Metric                         │ SMA    │ EWMA   │
├────────────────────────────────┼────────┼────────┤
│ 📈 Returns & Performance       │        │        │
│ Total Return %                 │ +45.2% │ +52.1% │
│ CAGR %                         │  +8.1% │  +9.3% │
├────────────────────────────────┼────────┼────────┤
│ ⚠️ Risk & Volatility          │        │        │
│ Annual Vol %                   │ +14.2% │ +18.9% │
│ Max Drawdown %                 │ -12.4% │ -10.2% │
├────────────────────────────────┼────────┼────────┤
│ 🎯 Risk-Adjusted Returns      │        │        │
│ Sharpe Ratio                   │   1.23 │   1.45 │
│ Sharpe Drag                    │   0.08 │   0.07 │
├────────────────────────────────┼────────┼────────┤
│ 💰 Fee & Cost Analysis        │        │        │
│ Fee Drag Ratio                 │   2.69%│   2.41%│
│ Cost Efficiency                │  97.31%│  97.59%│
└────────────────────────────────┴────────┴────────┘
```

---

## 🛠️ Adding New Strategies
1. Create `src/strategies/my_strategy.py`
2. Inherit `BaseStrategy`
3. Implement `generate_positions()`
4. Add to `src/strategies/__init__.py` and `src/__init__.py`
5. Import and run!

```python
# Example: Simple momentum strategy
from src.strategies.base import BaseStrategy
import pandas as pd

class MomentumStrategy(BaseStrategy):
    def __init__(self, lookback: int = 20, position_size: float = 1.0):
        super().__init__(name=f"Momentum-{lookback}d")
        self.lookback = lookback
        self.position_size = position_size
    
    def generate_positions(self, data: pd.DataFrame) -> pd.Series:
        returns = data['close'].pct_change()
        signal = returns.rolling(self.lookback).mean().apply(lambda x: 1 if x > 0 else -1)
        return signal * self.position_size
```

---

## 🔍 Notes
- **Designed for rapid prototyping.** For production-grade trade tracking, MAE/MFE, database persistence, and interactive Plotly dashboards, use the main [`financial_tools`](https://github.com/...) repository.
- All position series are automatically aligned to the input data index and shifted by 1 bar to prevent lookahead bias.
- Matplotlib plots are static and publication-ready. Use `save_to_file=False` for interactive Jupyter display.
- Metrics are calculated on the equity curve; `avg_drawdown_pct` isolates typical pain vs. worst-case scenarios.
- Fee-adjusted metrics are calculated **exactly** in the engine using `raw_pnl` and `total_costs` — no estimation or approximation.

---

## 📜 License
MIT

---

> 💡 **Pro Tip**: Use `compare_strategies(..., group_by_category=False)` if you prefer the original metric order from your engine. All functions accept `metrics=[...]` to filter to a custom subset.