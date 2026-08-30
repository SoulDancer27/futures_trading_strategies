# sysstrat

A minimal, readable, educational **systematic-trading backtester**.

It implements the core ideas from Robert Carver's *Systematic Trading* and *Advanced Futures Trading Strategies* — volatility targeting, forecast scaling, composable trading rules, and portfolio construction — in a small codebase aimed at **spot/index instruments** (no futures roll calendars or contract selection).

## Features

- **Volatility targeting** position sizing: fixed-risk, half-Kelly cap, and leverage / margin / contract limits.
- **Forecast scaling**: raw signals normalised to a common scale (`avg |forecast| = 1`), so heterogeneous rules can be combined.
- **Trading rules**: buy-and-hold, MA crossover, EWMAC (trend), normalised trend, breakout, acceleration, skew, and mean reversion.
- **Combined forecasts**: weighted averages with a forecast diversification multiplier (FDM).
- **Portfolio construction**: equal/weighted combination, diversification ratio, correlation matrix, and an expected-risk metric.
- **Analysis**: full performance metrics, benchmark regression (alpha/beta), and plotting.

## Install

```bash
pip install sysstrat
```

Or, from source (editable):

```bash
pip install -e .
```

## Quick start

```python
from sysstrat.data import load_simple_price_csv
from sysstrat.core import Asset, Capital, FixedRiskSizer
from sysstrat.engine import BacktestRunner
from sysstrat.strategies import EWMACStrategy

asset = Asset(
    ticker="MCFTR",
    price_data=load_simple_price_csv("data/MCFTR.csv"),
    commission_rate=0.0004,
    slippage_rate=0.001,
)

capital = Capital(initial_capital=100_000)
sizer = FixedRiskSizer(risk_target=0.20, max_leverage=1.0)

report = BacktestRunner(capital, asset, sizer).run(EWMACStrategy())
report.print_summary()
report.plot()
```

See the notebooks in this repo (`test.ipynb`, `breakout_test.ipynb`, `mean_reversion_test.ipynb`, `portfolio_diversification.ipynb`) for fuller walkthroughs.

## How it differs from `pysystemtrade`

Robert Carver's own `pysystemtrade` library is a full, production-oriented framework built around **futures** — roll calendars, contract selection, and carry-from-basis. `sysstrat` is intentionally the opposite:

- **Spot/index instruments**, not futures (no rolling, no basis-carry).
- **Small and readable** — a few hundred lines per concept, meant to be read top-to-bottom.
- **Educational** — each rule and sizing step is explicit, matching the book's explanations.

## Project layout

```
sysstrat/
    core/          # Capital, Asset, sizers, data models
    data/          # CSV loading
    engine/        # vectorised engine, analyzer, portfolio, regression
    strategies/    # trading rules + transforms + combined forecasts
    visualization/ # plotting and console reporting
```

## License

MIT — see `LICENSE`.
