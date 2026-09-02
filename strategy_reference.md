# Strategy reference

Classification of the strategies in `sysstrat`, with feasible parameter values for
testing. Use with `my_tests/12. Parameter sweep template.ipynb`.

## The two families

| Family | Also called | Signal logic | Works best when |
| --- | --- | --- | --- |
| **Trend** | Divergent, momentum | Buy what's rising, sell what's falling; profit from continuation | Markets trend (equities, bonds in a move, commodities) |
| **Mean reversion** | Convergent, contrarian | Fade extremes; profit from return to the mean | Markets oscillate in a range (FX pairs, calm regimes) |

Cross-sectional rules are a third family: they are market-neutral *relative* bets
(long winners / short losers) and can be trend (momentum) or mean-reversion flavoured.

---

## Trend (divergent)

| Strategy | Signal | Parameters (feasible test range) | Notes |
| --- | --- | --- | --- |
| `MACrossoverStrategy` | +1/-1/0 on MA cross | `short_window` 5–50, `long_window` 50–250, `ma_type` sma/ema, `mode` long_only/long_short | Best performer in the initial scan (10/50). Try `ema` and `long_short` on FX. |
| `EWMACStrategy` | Fast-minus-slow EWMA, vol-normalised, clip ±2 | `fast_window` 8–64, `slow_window` 32–256 (fast < slow), `vol_span` 25–100 | Faster won with blended sizing (8/32 > 16/64). |
| `NormalisedTrendStrategy` | EWMAC on vol-normalised price | `fast_span` 16–64, `slow_span` 64–256, `vol_window` 25 | Less correlated with plain EWMAC; good diversifier. |
| `BreakoutStrategy` | Position in rolling high/low channel (smoothed) | `horizon` 20–320 | Continuous signal, less whippy than Donchian. |
| `DonchianStrategy` | Break of N-day high/low, discrete ±1 with exit channel | `entry_window` 20–120, `exit_window` 10–20 (entry > exit) | Classic turtle. Lower turnover than MA cross. |
| `TimeSeriesMomentumStrategy` | 12-1 style return sign, vol-normalised | `lookback` 63–504, `skip` 5–63, `vol_span` 63 | TSMOM; weak on this short FX history, stronger on MCFTR/RGBITR. |
| `AccelerationStrategy` | Change in EWMAC trend | `fast_span` 8–32, `vol_span` 25 | Trend-of-trend; often too noisy alone. |
| `MACDStrategy` | MACD − signal line, vol-normalised | `fast` 8–24, `slow` 26–52, `signal_span` 9, `vol_span` 100 | Similar to EWMAC but laggier. |

## Mean reversion (convergent)

| Strategy | Signal | Parameters (feasible test range) | Notes |
| --- | --- | --- | --- |
| `MeanReversionStrategy` | Equilibrium − price, risk-adjusted, trend-filtered, vol-multiplied | `equil_span` 5–80, `vol_span` 25, `trend_fast`/`trend_slow` 16/64, `quantile_window` 2560 | Carver's rule; already has a trend filter — if it still loses, bare MR is dead on daily data. |
| `BollingerMeanReversionStrategy` | −z-score of price vs rolling band | `window` 20–200, `num_std` 1.5–3.0 | Wider bands = fewer, stronger signals. |
| `RSIMeanReversionStrategy` | Fade RSI extremes | `window` 2–14, `oversold` 10, `overbought` 90 | RSI(2) is the classic short-term fade; very high turnover. |
| `SkewStrategy` | −rolling skew of returns | `window` 60 | Contrarian crash-anticipation flavour; ~0 Sharpe in the scan. |

## Cross-sectional (relative)

Basket-level: operate on a wide DataFrame of prices, output per-asset signals.
Run through `FixedSignalStrategy`.

| Function | Signal | Parameters | Notes |
| --- | --- | --- | --- |
| `cross_sectional_momentum` | Z-score of past returns across assets, long winners / short losers | `lookback` 63–252, `skip` 5–63, `clip` 2 | Lost to equal-weight B&H on this basket (all assets share a RUB/common factor). |
| `cross_sectional_reversal` | −Z-score of recent (1–5 day) returns | `lookback` 1–10, `clip` 2 | Short-term reversal; also lost on this basket. |

## Combiners & wrappers

| Class | Purpose | Parameters |
| --- | --- | --- |
| `CombinedForecastStrategy` | Weighted average of rule forecasts × FDM, clipped | `rules` (list of strategies), `weights` (optional), `fdm` (from `forecast_diversification_multiplier`), `cap` 2 |
| `FixedSignalStrategy` | Wrap a precomputed signal so it runs through `BacktestRunner` | `signal` (Series), `name` |
| `BuyAndHoldStrategy` | Passive long benchmark (beta) | — |

## Quick testing notes

- All strategies emit signals; `FixedRiskSizer` turns them into positions. The scan
  showed **blended(32/2520)** sizing improves Sharpe vs SMA(252) for most rules.
- Mean-reversion rules all lost in the initial scan across all five assets — check the
  per-asset tables (FX pairs are the most plausible exceptions) before writing them off.
- Always compare against `BuyAndHoldStrategy` on the same basket — on this dataset the
  long-only beta is strong (Sharpe ~1.3–1.6 diversified).
