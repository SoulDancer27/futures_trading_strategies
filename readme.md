Here’s a polished, production-ready `README.md` that reflects your current architecture, new metrics, clean API, and barebones design philosophy.

```markdown
# 📈 Standalone Vectorized Backtester

A minimal, modular Python framework for rapid futures & cash market backtesting. Designed for quantitative research, parameter sweeps, and strategy validation without database dependencies or complex visualization overhead.

> 🔗 **Companion to** [`financial_tools`](https://github.com/...): Strips away ORM/Plotly complexity while preserving the clean 3-layer architecture for fast, barebones testing.

---

## ✨ Features
- ⚡ **Pure vectorized execution** using `pandas`/`numpy` (no event loops)
- 📊 **Continuous position sizing** (contracts/shares per time step, no trade-pair matching)
-  **Headerless CSV loader** for simple `[date, price]` series with automatic index normalization
- 💰 **Flexible transaction costs**: fixed per contract **or** decimal `commission_rate`
-  **Slippage modeling** via `slippage_rate` (% of trade value)
-  **Advanced risk metrics**: Average Drawdown, Tail Ratios (fat-tail analysis), Skew, Sharpe, CAGR
- 🎨 **Matplotlib reporting** (equity curve, positions, drawdown) with optional file output
- 🧩 **Strict modular architecture** (`data` → `strategies` → `engine` → `plots`)
- 📦 **Zero external dependencies** beyond core scientific Python stack

---

## 📁 Project Structure
```
.
├── src/
│   ├── data/          # CSV loaders & index normalization
│   ├── strategies/    # BaseStrategy + implementations
│   ├── engine/        # Vectorized execution (split for clarity)
│   │   ├── models.py      # BacktestResult dataclass + reporting
│   │   ├── metrics.py     # Performance & risk calculations
│   │   └── vectorized.py  # Core execution loop
│   ├── plots/         # Matplotlib visualization utilities
│   └── __init__.py    # Clean top-level exports
├── data/              # Place your headerless CSV files here
├── outputs/           # Auto-generated backtest plots
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install pandas numpy matplotlib loguru
```

### 2. Prepare Data
Place a headerless CSV in `data/` with two columns: `date` and `price`.
```csv
20241001,65.50
20241001,66.20
20241002,64.80
```
*(Supports any standard datetime format via the `date_format` parameter)*

### 3. Run a Backtest (Jupyter / Script)
```python
from src import (
    load_simple_price_csv,
    VectorizedEngine,
    MomentumStrategy,
    plot_results
)

# 1. Load data
data = load_simple_price_csv("data/sample.csv", date_format="%Y%m%d")

# 2. Generate positions & run engine
strategy = MomentumStrategy(lookback=20, position_size=1.0)
engine = VectorizedEngine(
    initial_capital=100_000,
    point_value=1.0,           # 1.0 for stocks/spot, 50 for ES futures, etc.
    commission_rate=0.001,     # 0.1% of trade value
    slippage_rate=0.0005       # 0.05% slippage
)
result = engine.run(strategy, data, ticker="Sample Instrument")

# 3. View metrics & plot
result.print_summary()
plot_results(result, save_to_file=True, output_path="outputs/test.png")
```

---

## ⚙️ Configuration Reference

### Vectorized Engine
| Parameter | Type | Default | Description |
|---|---|---|---|
| `initial_capital` | `float` | `100_000` | Starting portfolio value |
| `point_value` | `float` | `1.0` | Currency value per 1-point price move (`1.0` = cash, `50.0` = ES) |
| `commission_per_contract` | `float` | `None` | Fixed fee per contract/share traded |
| `commission_rate` | `float` | `None` | Decimal rate of trade value (e.g., `0.001` = 0.1%) |
| `slippage_rate` | `float` | `None` | Decimal rate of slippage (e.g., `0.0005` = 0.05%) |

*⚠️ Note: Specify either `commission_per_contract` OR `commission_rate`. Not both.*

### Strategies
All strategies inherit `src.strategies.base.BaseStrategy` and must implement:
```python
def generate_positions(self, data: pd.DataFrame) -> pd.Series:
    """Return continuous position size (positive=long, negative=short, 0=flat)"""
    pass
```
Positions are automatically shifted by 1 bar in the engine to prevent lookahead bias.

### Data Loader
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
The engine calculates and returns:
| Metric | Description |
|--------|-------------|
| `total_return_pct`, `cagr_pct` | Absolute & annualized returns |
| `annual_volatility_pct` | Annualized standard deviation |
| `sharpe_ratio` | Risk-adjusted return (CAGR / Vol) |
| `max_drawdown_pct` | Worst peak-to-trough decline |
| `avg_drawdown_pct` | Average of max drawdowns per distinct period |
| `skew` | Return distribution asymmetry |
| `lower_tail`, `upper_tail`, `tail_risk` | Percentile-based fat-tail ratios (vs. Gaussian baseline of 1.0) |
| `final_equity` | Ending portfolio value |

**Tail Ratio Interpretation:**
- `1.0` = Normal Gaussian distribution
- `> 1.0` = Fatter tails (more extreme outliers than normal)
- Separates left (`lower_tail`) and right (`upper_tail`) tail behavior for precise risk profiling.

---

## 🛠️ Adding New Strategies
1. Create `src/strategies/my_strategy.py`
2. Inherit `BaseStrategy`
3. Implement `generate_positions()`
4. Add to `src/strategies/__init__.py` and `src/__init__.py`
5. Import and run!

---

##  Notes
- **Designed for rapid prototyping.** For production-grade trade tracking, MAE/MFE, database persistence, and interactive Plotly dashboards, use the main [`financial_tools`](https://github.com/...) repository.
- All position series are automatically aligned to the input data index.
- Matplotlib plots are static and publication-ready. Use `save_to_file=False` for interactive Jupyter display.
- Metrics are calculated on the equity curve; `avg_drawdown_pct` isolates typical pain vs. worst-case scenarios.

---

## 📜 License
MIT
```

### 🔑 Key Improvements Over Previous Version
- Reflects the **engine split** (`models.py`, `metrics.py`, `vectorized.py`)
- Updates commission naming to `commission_rate` (decimal, unambiguous)
- Adds `slippage_rate` configuration
- Documents **Average Drawdown** vs **Max Drawdown** distinction
- Explains **Tail Ratios** clearly with interpretation guide
- Aligns with your barebones, Jupyter-first workflow
- Maintains clean tables, code blocks, and professional formatting

Drop this into your repo root. It's ready for GitHub, local reference, or onboarding collaborators. Let me know if you want to add a `CHANGELOG.md` or strategy template examples next! 📦✨