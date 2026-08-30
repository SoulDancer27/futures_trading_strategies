# src/strategies/combined_forecast.py
import numpy as np
import pandas as pd

from .base import BaseStrategy


def forecast_diversification_multiplier(signals, weights=None) -> float:
    """
    Forecast Diversification Multiplier (FDM).

    Combining correlated forecasts shrinks their average magnitude. The FDM
    restores it: FDM = 1 / sqrt(w^T R w), where R is the correlation matrix of the
    (equally-scaled) rule forecasts and w their weights.

    ``signals`` is an iterable of aligned Series (one per rule, across instruments).
    """
    df = pd.concat([pd.Series(s) for s in signals], axis=1).dropna()
    corr = df.corr().values
    n = df.shape[1]
    if weights is None:
        weights = np.ones(n) / n
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    var = float(w @ corr @ w)
    return 1.0 / np.sqrt(var) if var > 1e-12 else 1.0


class CombinedForecastStrategy(BaseStrategy):
    """
    Carver's 'combined forecast' (Strategy 11 and Part Two).

    Produces a single signal by taking a weighted average of several rule forecasts,
    multiplying by the forecast diversification multiplier, and capping the result.

    ``rules`` is an iterable of ``BaseStrategy``; ``weights`` align with ``rules``
    (normalised to sum to 1). ``fdm`` is optional (default 1.0 = no scaling); compute
    it with :func:`forecast_diversification_multiplier` from the rule signals.
    """

    def __init__(self, rules, weights=None, fdm: float = 1.0, cap: float = 2.0, name: str = "Combined Forecast"):
        super().__init__(name=name)
        self.rules = list(rules)
        n = len(self.rules)
        if weights is None:
            weights = np.ones(n) / n
        weights = np.asarray(weights, dtype=float)
        if weights.sum() == 0:
            raise ValueError("weights sum to zero")
        self.weights = weights / weights.sum()
        self.fdm = fdm
        self.cap = cap

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        combined = None
        for strategy, w in zip(self.rules, self.weights):
            sig = strategy.generate_signals(data)
            combined = w * sig if combined is None else combined + w * sig

        if self.fdm is not None:
            combined = combined * self.fdm

        return combined.clip(lower=-self.cap, upper=self.cap)

    def get_parameters(self) -> dict:
        return {
            "weights": [round(float(w), 4) for w in self.weights],
            "fdm": self.fdm,
            "cap": self.cap,
            "rules": [s.name for s in self.rules],
        }
