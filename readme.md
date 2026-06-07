# 📈 Standalone Vectorized Backtester

A minimal, modular Python framework for rapid futures & cash market backtesting. Designed for quantitative research, parameter sweeps, and strategy validation without database dependencies or complex visualization overhead.

> 🔗 **Companion to** [`financial_tools`](https://github.com/...): Strips away ORM/Plotly complexity while preserving the clean 3-layer architecture for fast, barebones testing.

---

## ✨ Features
- ⚡ **Pure vectorized execution** using `pandas`/`numpy` (no event loops)
- 📊 **Continuous position sizing** (contracts/shares per time step)
- 📥 **Headerless CSV loader** for simple `[date, price]` series
- 📉 **Matplotlib reporting** (equity curve, positions, drawdown)
- 💰 **Flexible commissions** (fixed per contract OR percentage of trade value)
- 🧩 **Modular architecture** (`data` → `strategies` → `engine` → `plots`)
- 📦 **Zero external dependencies** beyond core scientific Python stack

---

## 📁 Project Structure
```
.
├── src/
│   ├── data/          # CSV loaders & index normalization
│   ├── strategies/    # BaseStrategy + implementations
│   ├── engine/        # Vectorized execution & metrics calculation
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

# 2. Generate positions
strategy = MomentumStrategy(lookback=20, position_size=1.0)
positions = strategy.generate_positions(data)

# 3. Run engine
engine = VectorizedEngine(
    initial_capital=100_000,
    point_value=1.0,        # 1.0 for stocks/spot, 50 for ES futures, etc.
    commission_pct=0.001    # OR use commission_per_contract=2.5
)
result = engine.run(data, positions)

# 4. View metrics & plot
print(result.metrics)
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
| `commission_pct` | `float` | `None` | Percentage of trade value (e.g., `0.001` = 0.1%) |

*⚠️ Note: Specify either `commission_per_contract` OR `commission_pct`. Not both.*

### Strategies
All strategies inherit `src.strategies.base.BaseStrategy` and must implement:
```python
def generate_positions(self, data: pd.DataFrame) -> pd.Series:
    # Return continuous position size (positive=long, negative=short, 0=flat)
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
- `total_return_pct`, `cagr_pct`
- `annual_volatility_pct`, `sharpe_ratio`
- `max_drawdown_pct`
- `final_equity`

---

## 🛠️ Adding New Strategies
1. Create `src/strategies/my_strategy.py`
2. Inherit `BaseStrategy`
3. Implement `generate_positions()`
4. Add to `src/strategies/__init__.py` and `src/__init__.py`
5. Import and run!

---

## 📝 Notes
- Designed for **rapid prototyping**. For production-grade trade tracking, MAE/MFE, database persistence, and interactive Plotly dashboards, use the main [`financial_tools`](https://github.com/...) repository.
- All position series are automatically aligned to the input data index.
- Matplotlib plots are static and publication-ready. Use `save_to_file=False` for interactive Jupyter display.

---

## 📜 License
MIT