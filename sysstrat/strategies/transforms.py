"""
Reusable price transforms and signal primitives for strategies.

These are pure functions operating on a price Series, shared across strategies so
that rules can be composed (e.g. an EWMAC crossover computed on a normalised price).
"""
import numpy as np
import pandas as pd


def normalise_price(price: pd.Series, vol_window: int = 25, scale: float = 100.0) -> pd.Series:
    """
    Carver's 'normalised price' (Strategy 17).

    A recursively accumulated series whose daily increments are the raw price
    change divided by a rolling standard deviation of price *differences*
    (absolute changes, not percentage returns):

        pn_0   = 0
        pn_t   = pn_{t-1} + (p_t - p_{t-1}) / sigma_t * scale

    where ``sigma_t`` is the rolling std of ``(p_t - p_{t-1})`` over ``vol_window``.
    The level offset (``pn_0 = 0``) and ``scale`` are arbitrary and only affect the
    magnitude, not the shape.
    """
    diffs = price.diff()
    sigma = diffs.rolling(window=vol_window, min_periods=1).std()
    sigma = sigma.replace(0.0, np.nan)
    increments = (diffs / sigma * scale).fillna(0.0)
    return increments.cumsum()


def ewmac_crossover(price: pd.Series, fast_span: int, slow_span: int) -> pd.Series:
    """Fast-minus-slow EWMA of ``price`` (the raw EWMAC 'trend', before normalisation)."""
    fast = price.ewm(span=fast_span, adjust=False).mean()
    slow = price.ewm(span=slow_span, adjust=False).mean()
    return fast - slow


def ewmac_raw_forecast(price: pd.Series, fast_span: int, slow_span: int, vol_span: int = 25) -> pd.Series:
    """
    Raw (unscaled) EWMAC forecast: the crossover divided by the daily price
    volatility, i.e. the crossover expressed in "daily volatility units".

        raw = (EWMA(fast) - EWMA(slow)) / (price * daily_vol)

    This is the input to both the EWMAC rule and the acceleration rule.
    """
    crossover = ewmac_crossover(price, fast_span, slow_span)
    daily_returns = price.pct_change()
    annualized_vol = daily_returns.ewm(span=vol_span, adjust=False).std() * np.sqrt(252)
    annualized_vol = annualized_vol.clip(lower=0.05)
    daily_price_vol = price * annualized_vol / np.sqrt(252)
    return crossover / (daily_price_vol + 1e-9)


def skew_forecast(price: pd.Series, window: int) -> pd.Series:
    """
    Carver's skew forecast: the negative rolling skew of daily percentage returns,
    smoothed with an EWMA of span ``window/4``.

        raw       = -Skew(r_t, ..., r_{t-window+1})
        smoothed  = EWMA_{window/4}[raw]

    Negative skew -> positive forecast (long); positive skew -> negative (short).
    """
    returns = price.pct_change()
    rolling_skew = returns.rolling(window=window).skew()
    raw = -rolling_skew
    smooth_span = max(window / 4.0, 1.0)
    return raw.ewm(span=smooth_span, adjust=False).mean()


def _expanding_percentile_rank(s: pd.Series) -> pd.Series:
    """Expanding percentile rank (fraction of past values <= current), in [0, 1]."""
    vals = s.to_numpy()
    out = np.empty(len(vals))
    for t in range(len(vals)):
        out[t] = (vals[: t + 1] <= vals[t]).mean()
    return pd.Series(out, index=s.index)


def mean_reversion_forecast(
    price: pd.Series,
    equil_span: int = 5,
    vol_span: int = 25,
    trend_fast: int = 16,
    trend_slow: int = 64,
    quantile_window: int = 2560,
    quantile_smooth: int = 10,
) -> pd.Series:
    """
    Carver's fast mean-reversion forecast (trend-filtered, volatility-scaled).

    1. Equilibrium = EWMA(span=equil_span) of price; raw = equilibrium - price
       (positive -> price is below equilibrium -> long).
    2. Risk-adjust: divide by the daily price volatility.
    3. Trend overlay: zero the forecast where the EWMAC(trend_fast, trend_slow)
       sign disagrees (only mean-revert in the trend direction).
    4. Volatility multiplier: scale down when current vol is high relative to its
       ``quantile_window``-day history (mean reversion works better in calm markets).
    """
    equil = price.ewm(span=equil_span, adjust=False).mean()
    raw = equil - price

    daily_returns = price.pct_change()
    annualized_vol = daily_returns.ewm(span=vol_span, adjust=False).std() * np.sqrt(252)
    annualized_vol = annualized_vol.clip(lower=0.05)
    daily_price_vol = price * annualized_vol / np.sqrt(252)
    risk_adj = raw / (daily_price_vol + 1e-9)

    trend = ewmac_raw_forecast(price, trend_fast, trend_slow, vol_span)
    risk_adj = risk_adj.where(np.sign(trend) == np.sign(risk_adj), 0.0)

    relative_vol = (annualized_vol / annualized_vol.rolling(quantile_window, min_periods=1).mean()).fillna(1.0)
    q = _expanding_percentile_rank(relative_vol)
    multiplier = (2.0 - 1.5 * q).ewm(span=quantile_smooth, adjust=False).mean()

    return risk_adj * multiplier


def breakout_forecast(price: pd.Series, horizon: int) -> pd.Series:
    """
    Carver's rolling breakout forecast (raw, smoothed).

    The price's position within a rolling ``horizon``-day high/low channel,
    scaled to roughly [-20, 20], then smoothed with an EWMA of span ``horizon/4``:

        raw       = 40 * (p - (max + min)/2) / (max - min)
        smoothed  = EWMA_{horizon/4}[raw]

    A value near +20 means the price is at a new ``horizon``-day high (strong long);
    near -20, a new low (strong short).
    """
    roll_max = price.rolling(window=horizon, min_periods=1).max()
    roll_min = price.rolling(window=horizon, min_periods=1).min()
    mid = (roll_max + roll_min) / 2.0
    rng = (roll_max - roll_min).replace(0.0, np.nan)
    raw = 40.0 * (price - mid) / rng
    smooth_span = max(horizon / 4.0, 1.0)
    return raw.ewm(span=smooth_span, adjust=False).mean()


def calibrate_forecast_scalar(
    raw_forecasts, target: float = 1.0, clip: float = 2.0, ignore_zeros: bool = False, max_iter: int = 100
) -> float:
    """
    Calibrate a forecast scalar from a basket of raw-forecast series.

    Solves for the scalar ``s`` such that the CLIPPED signal
    ``clip(raw / s, -clip, clip)`` has an average absolute value of ``target``.
    Calibrating on the clipped signal (rather than the raw mean) accounts for the
    tail-truncation, so the realised position averages ~``target`` (= ~1x the
    volatility-targeted size). ``clip`` must exceed ``target`` (Carver: forecast
    clipped at ±20 while averaging 10, i.e. clip = 2 x target).

    ``ignore_zeros`` drops exact-zero forecasts before calibrating. Use it for sparse
    signals (e.g. a trend-filtered forecast that is flat most of the time), where the
    target would otherwise be unreachable.

    ``raw_forecasts`` is an iterable of Series (one per instrument).
    """
    pooled = pd.concat([pd.Series(s) for s in raw_forecasts]).dropna()
    if ignore_zeros:
        pooled = pooled[pooled != 0]
    if pooled.empty:
        raise ValueError("No usable raw-forecast observations to calibrate.")

    def avg_clipped(scalar: float) -> float:
        return float((pooled / scalar).clip(-clip, clip).abs().mean())

    # avg_clipped is monotonically decreasing in `scalar`.
    lo, hi = 1e-9, float(pooled.abs().max())
    if avg_clipped(hi) >= target:
        return hi
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        if avg_clipped(mid) > target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0
