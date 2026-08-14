# Architecture Status

A living snapshot of the backtesting framework's architecture and current state.

---

## 1. Core Philosophy

- **Separation of Concerns:** strict boundaries between Data (`core`), Execution & Analytics (`engine`), Strategies, and Visualization.
- **No "God Objects":** small, single-responsibility classes/functions.
- **Fail-Fast:** dataclass validation catches errors immediately.
- **Pure Data Containers:** results hold data only; logic lives in dedicated modules.
- **Single Source of Truth for Math:** capital-model math is delegated to the `Capital` abstraction, never re-implemented ad hoc.

## 2. Directory Structure

```text
src/
├── core/                   # Data structures, capital model, and math
│   ├── asset.py            # Asset dataclass
│   ├── capital.py          # BaseCapitalModel (ABC) + FixedCapitalModel + Capital
│   ├── sizers.py           # Position sizers (Signal -> Position bridge)
│   ├── models.py           # ExecutionResult, PerformanceMetrics (pure data)
│   └── __init__.py
│
├── engine/                 # Execution, analysis, portfolio, and facade
│   ├── vectorized.py       # VectorizedEngine (execution & accounting)
│   ├── analyzer.py         # PerformanceAnalyzer (analytics & metrics)
│   ├── portfolio.py        # Portfolio, PortfolioExecutionResult
│   ├── runner.py           # BacktestRunner, BacktestReport (facade)
│   └── __init__.py
│
├── strategies/             # Strategy implementations (signal generators)
│   ├── base.py             # BaseStrategy (generate_signals)
│   ├── buy_and_hold.py     # BuyAndHoldStrategy
│   ├── ma_crossover.py     # MACrossoverStrategy (sma/ema, long_only/long_short)
│   ├── ewmac.py            # EWMACStrategy (trend strength)
│   └── __init__.py
│
├── data/                   # Data loading
│   ├── loader.py           # load_simple_price_csv
│   └── __init__.py
│
├── visualization/          # Drawing & reporting (no math)
│   ├── plotter.py          # Panel-registry plotter
│   ├── reporter.py         # Console summaries & comparison tables
│   └── __init__.py
│
└── __init__.py
```

## 3. Module Breakdown & Current State

### `src/core/`
- **`asset.py`** — `Asset` dataclass: `ticker`, `price_data` (pd.Series), `point_value`, `commission_rate` / `commission_per_contract`, `slippage_rate`, `trading_days`. `__post_init__` validates non-empty data and mutual exclusivity of the two commission modes.
- **`capital.py`** — `BaseCapitalModel` (ABC) with `calculate_returns`, `calculate_total_return`, `calculate_cagr`, `calculate_drawdown`, `calculate_leverage`. `FixedCapitalModel` implements them using `initial_capital` as the constant denominator (Carver fixed-capital, non-compounding). `Capital` dataclass bundles `initial_capital`, `risk_free_rate`, `capital_model`.
- **`sizers.py`** — `BasePositionSizer` (ABC) plus:
  - `FixedFractionSizer` — fixed % of capital per unit of signal.
  - `FixedContractsSizer` — constant contract count (true buy & hold).
  - `FixedRiskSizer` — Carver volatility targeting (`contracts = capital × risk / (price × point_value × annualized_vol)`) with a half-Kelly cap (`risk ≤ 0.5 × expected_sharpe`) and ordered position caps (max leverage → margin → max contracts).
- **`models.py`** — pure dataclasses:
  - `ExecutionResult` — time series only (equity, daily_pnl, positions, leverage, drawdown, returns, realized_vol, cumulative_fees, cumulative_turnover, asset). All fields mandatory.
  - `PerformanceMetrics` — aggregate scalars derived from an `ExecutionResult`.

### `src/engine/`
- **`vectorized.py`** — `VectorizedEngine`. *Execution & accounting only.* Computes raw PnL (`pos.shift(1)` avoids signal look-ahead), turnover, costs, equity, and delegates `leverage`/`returns`/`drawdown` to the capital model. Costs are priced at the execution (previous) close, not today's close.
- **`analyzer.py`** — `PerformanceAnalyzer`. *Stateless.* Constructed with a `Capital`; delegates `total_return`/`cagr`/`initial_capital`/`risk_free_rate` to it, then computes the remaining scalar metrics.
- **`portfolio.py`** — `PortfolioExecutionResult` (adds weights, correlation, diversification metrics) and `Portfolio` (combines results; deduplicates duplicate `strategy_name`).
- **`runner.py`** — `BacktestRunner` / `BacktestReport` facade wiring Engine + Analyzer + Plotter.

### `src/strategies/`
- **`base.py`** — `BaseStrategy.generate_signals()`: strategies are money-agnostic and output raw signals in [-1, 1].
- **`buy_and_hold.py`**, **`ma_crossover.py`**, **`ewmac.py`** — concrete strategies.

### `src/data/`
- **`loader.py`** — `load_simple_price_csv()`: 2-column CSV (date, price) → `pd.Series` with a `DatetimeIndex`.

### `src/visualization/`
- **`plotter.py`** — panel-registry renderers; reads pre-computed series, does no math.
- **`reporter.py`** — `print_summary`, `print_comparison_table`, `print_report_comparison`, `print_portfolio_comparison`, `print_portfolio_diversification`.

## 4. Key Architectural Decisions

1. **Signal vs. Position:** strategies output *signals*; sizers bridge signal → capital → *positions*. The same strategy is testable across account sizes.
2. **Fixed-capital model (Carver "Advanced Futures Trading Strategies"):** all per-period math uses `initial_capital` as the denominator; no compounding in the vectorized engine (compounding belongs in an event-driven engine).
3. **Single source of truth for math:** the capital model computes `returns`/`drawdown`/`leverage`/`total_return`/`cagr`; the analyzer and engine delegate to it rather than re-implementing.
4. **Strict contracts:** `ExecutionResult` fields are mandatory; downstream code has no `if x is not None` checks.
5. **No look-ahead:** `pos.shift(1)` for signals, and costs priced at the execution close.

## 5. Recently Fixed (audit pass)

- Short-position leverage was clipped to 0 → now gross leverage `|notional| / initial_capital`.
- Analyzer implicitly hardcoded fixed-capital math → now injected with `Capital` and delegates.
- `print_portfolio_comparison` referenced non-existent `res.metrics` → now accepts `PerformanceMetrics`.
- `Portfolio` silently dropped duplicate `strategy_name` → now deduplicates list input.
- Commission/slippage priced one bar late → now priced at the execution close.
- Day-0 fees/turnover dropped by `.diff().fillna(0)` → now preserved.
- `FixedRiskSizer` mixed risk-target units with leverage-ratio constraints → now pure vol-targeting + half-Kelly cap.

## 6. Notes / Conventions

- `point_value=1.0` in the notebooks models futures as units/CFDs (no margin or contract multiplier).
- `commission_rate` is interpreted as a fraction of traded notional.
- Data lives in `data/*.csv` (USDRUB, GLDRUB_TOM, MCFTR, RGBITR, RGBITR1Y, CNYRUB_TOM).
- Full usage examples live in `my_tests/*.ipynb` and `test.ipynb`.
