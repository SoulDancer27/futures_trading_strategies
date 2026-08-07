Here is a comprehensive summary of the `trimmed-architecture` branch. Save this as `ARCHITECTURE_STATUS.md` in your project root so we can pick up exactly where we left off tomorrow.

***

# `trimmed-architecture` Branch Status & Architecture Guide

## 1. Core Philosophy
The goal of this branch is to completely refactor the backtesting infrastructure into a clean, modular, and mathematically honest system. 
* **Separation of Concerns:** Strict boundaries between Data (Core), Execution (Engine), Analytics (Analyzer), and Visualization (Plotter).
* **No "God Objects":** Eliminate massive classes/functions that do everything. 
* **Fail-Fast:** Use dataclass validation to catch errors immediately.
* **Pure Data Containers:** Results should just hold data; logic belongs in dedicated modules.

## 2. New Directory Structure
```text
src/
├── core/                   # Pure data structures, config, and math models
│   ├── asset.py            # Asset dataclass
│   ├── capital.py          # Capital dataclass + Capital Models
│   ├── sizers.py           # Position Sizers (Signal -> Position bridge)
│   └── models.py           # BacktestResult, PortfolioResult (Pure data)
│
├── engine/                 # Execution and analysis logic
│   ├── vectorized.py       # VectorizedEngine (Execution & Accounting only)
│   └── analyzer.py         # PerformanceAnalyzer (Analytics & Metrics)
│
├── strategies/             # Strategy implementations
│   ├── base.py             # BaseStrategy (generate_signals)
│   └── buy_and_hold.py     # BuyAndHoldStrategy
│
├── data/                   # Data loading
│   └── loader.py           # Simple CSV loaders
│
├── plots/                  # Visualization
│   └── plotter.py          # Modular Panel Registry plotter
│
└── __init__.py             # Clean public API exports
```

## 3. Module Breakdown & Current State

### `src/core/`
* **`asset.py`**: `Asset` dataclass. Holds `ticker`, `price_data` (pd.Series), `point_value`, `commission_rate`, `slippage_rate`. Includes `__post_init__` validation.
* **`capital.py`**: `Capital` dataclass (`initial_capital`, `risk_free_rate`, `capital_model`, `position_sizer`). Contains `BaseCapitalModel` (ABC) and `FixedCapitalModel`. 
  * *Note: `CompoundingCapitalModel` was intentionally removed from the vectorized engine because vectorized engines cannot honestly calculate iterative dynamic position sizing.*
* **`sizers.py`**: `BasePositionSizer` (ABC) and `FixedFractionSizer`. Translates raw strategy signals into actual contract sizes based on `Capital` and `Asset`.
* **`models.py`**: Pure dataclasses. 
  * `BacktestResult`: **All fields are mandatory** (no `Optional`). Includes `equity`, `positions`, `daily_pnl`, `metrics`, `drawdown`, `returns`, `realized_vol`, `leverage`, `cumulative_fees`, `cumulative_turnover`, and the `asset` object itself.
  * `PortfolioResult`: Holds portfolio-level series and metrics.

### `src/engine/`
* **`vectorized.py`**: `VectorizedEngine`. 
  * *Responsibility:* **Execution & Accounting only**. Calculates raw PnL, equity curve, transaction costs, and leverage. 
  * *Does NOT calculate:* Returns, drawdowns, or volatility. It delegates this to the Analyzer.
* **`analyzer.py`** *(To be finalized tomorrow)*: `PerformanceAnalyzer` class. 
  * *Responsibility:* **Analytics**. Takes raw execution data and calculates all analytical series (`returns`, `drawdown`, `realized_vol`) and the final scalar `metrics` dictionary. Prevents double-calculation and keeps the engine clean.

### `src/strategies/`
* **`base.py`**: `BaseStrategy`. Method renamed from `generate_positions` to `generate_signals()`. Strategies are "dumb" about money; they only output raw directional signals (e.g., -1.0 to 1.0).
* **`buy_and_hold.py`**: Simple baseline strategy outputting a constant `1.0` signal.

### `src/data/`
* **`loader.py`**: `load_simple_price_csv()`. Reads 2-column CSVs (date, price) and returns a pure `pd.Series` with a `DatetimeIndex`. No assumptions about column names.

### `src/plots/`
* **`plotter.py`**: Modular Panel Registry pattern. 
  * *Responsibility:* **Dumb drawing**. It does not calculate math. It receives pre-calculated series from `BacktestResult` and dispatches them to specific renderer functions (`_render_equity`, `_render_drawdown`, etc.).

## 4. Key Architectural Decisions Made Today
1. **Signal vs. Position:** Strategies output *signals*. The `PositionSizer` bridges the signal to the `Capital` to output *positions*. This allows the same strategy to be tested with different account sizes without changing the strategy code.
2. **Mathematical Honesty:** Removed compounding from the vectorized engine. Vectorized engines are for Fixed Capital signal testing. Compounding belongs in the Event-Driven engine (backtrader).
3. **Single Source of Truth for Math:** The `PerformanceAnalyzer` calculates returns/drawdowns/volatility *once*. The Engine doesn't calculate them, and the Plotter doesn't calculate them. This ensures the plotted drawdown perfectly matches the `max_drawdown` metric in the dictionary.
4. **Strict Contracts:** `BacktestResult` fields are mandatory. The Engine *must* provide them (even if they are zero), eliminating `if x is not None` checks downstream.

## 5. Next Steps for Tomorrow

### Immediate Code Tasks
- [ ] **Finalize `src/engine/analyzer.py`**: Implement the `PerformanceAnalyzer` class with `_calculate_series()` and `_calculate_metrics_dict()` methods.
- [ ] **Update `src/engine/vectorized.py`**: Refactor the `run()` method to use the new `PerformanceAnalyzer` instead of calculating metrics inline.
- [ ] **Finalize `src/plots/plotter.py`**: Ensure the `prepare_plot_data` function simply reads the pre-calculated series from `BacktestResult` without doing any math.
- [ ] **Create `scripts/test_architecture.py`**: Write a "Hello World" script using mock data to prove the full pipeline (Data -> Strategy -> Sizer -> Engine -> Analyzer -> Plotter) works end-to-end.

### Integration Tasks (Later)
- [ ] Connect the new `VectorizedEngine` to the existing `BacktestService` orchestration layer.
- [ ] Update `ResultStorage` to handle the new `BacktestResult` format when saving to the VPS PostgreSQL database.
- [ ] Re-implement `SMA Crossover` and `MA Crossover` strategies using the new `generate_signals()` framework.

***

**Rest up! You made massive progress today. The architecture is now incredibly clean, professional, and mathematically sound. See you tomorrow!**