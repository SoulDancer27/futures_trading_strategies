"""
Benchmark Regression Analysis.

Regresses a strategy's monthly returns on a benchmark's monthly returns to
estimate alpha (excess return) and beta (co-movement / exposure). This is the
standard single-factor (Jensen-style) regression used to separate skill from
the return you'd get simply by holding the benchmark.

Stateless, and dependency-free: OLS is computed with numpy.
"""
from typing import Union

import numpy as np
import pandas as pd

from ..core.models import ExecutionResult, RegressionResult


class RegressionAnalyzer:
    """
    Stateless analyzer: regress strategy monthly returns on benchmark monthly
    returns (y = strategy, x = benchmark) and report alpha/beta and significance.
    """

    def __init__(self, annualization_factor: int = 12):
        self.annualization_factor = annualization_factor

    def analyze(
        self,
        strategy: ExecutionResult,
        benchmark: Union[ExecutionResult, pd.Series],
    ) -> RegressionResult:
        """
        Regress the strategy's monthly returns on the benchmark's monthly returns.

        ``benchmark`` may be either an ``ExecutionResult`` (another strategy) or a raw
        return series — e.g. ``asset.price_data.pct_change()`` for a pure market benchmark.
        """
        y = self._monthly_returns(strategy.returns)
        strategy_name = strategy.strategy_name or "Strategy"

        if isinstance(benchmark, ExecutionResult):
            x = self._monthly_returns(benchmark.returns)
            benchmark_name = benchmark.strategy_name or "Benchmark"
        else:
            x = self._monthly_returns(benchmark)
            benchmark_name = getattr(benchmark, "name", None) or "Benchmark"

        # Align on common month-end dates.
        df = pd.concat([y, x], axis=1, join="inner").dropna()
        if len(df) < 3:
            raise ValueError("Not enough overlapping monthly observations for a regression.")

        alpha, beta, r_squared, t_alpha, t_beta = self._ols(df.iloc[:, 1].values, df.iloc[:, 0].values)

        alpha_monthly_pct = alpha * 100.0
        return RegressionResult(
            strategy_name=strategy_name,
            benchmark_name=benchmark_name,
            alpha_monthly_pct=alpha_monthly_pct,
            alpha_annualized_pct=alpha_monthly_pct * self.annualization_factor,
            beta=beta,
            r_squared=r_squared,
            alpha_t_stat=t_alpha,
            beta_t_stat=t_beta,
            n_observations=len(df),
            strategy_monthly=df.iloc[:, 0],
            benchmark_monthly=df.iloc[:, 1],
        )

    @staticmethod
    def _monthly_returns(returns: pd.Series) -> pd.Series:
        """
        Aggregate daily returns into monthly returns. Uses the arithmetic sum,
        consistent with the codebase's ``calculate_total_return = returns.sum()``.
        """
        return returns.resample("ME").sum()

    @staticmethod
    def _ols(x: np.ndarray, y: np.ndarray) -> tuple:
        """Fit y = alpha + beta * x and return (alpha, beta, r_squared, t_alpha, t_beta)."""
        n = len(x)
        if np.std(x) == 0:
            raise ValueError("Benchmark monthly returns have zero variance; cannot regress.")

        X = np.column_stack([np.ones(n), x])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        alpha, beta = float(coef[0]), float(coef[1])

        residuals = y - (alpha + beta * x)
        ss_res = float(residuals @ residuals)
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        dof = n - 2
        sigma2 = ss_res / dof if dof > 0 else 0.0
        xtx_inv = np.linalg.inv(X.T @ X)
        se = np.sqrt(np.maximum(sigma2, 0.0) * np.diag(xtx_inv))

        t_alpha = alpha / se[0] if se[0] > 0 else 0.0
        t_beta = beta / se[1] if se[1] > 0 else 0.0
        return alpha, beta, r_squared, float(t_alpha), float(t_beta)
