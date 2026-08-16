"""
High-level Facade for running backtests and generating reports.
Wraps the modular Engine, Analyzer, and Plotter for convenience.
"""
from typing import List, Optional, Dict, Union
from ..core.capital import Capital
from ..core.asset import Asset
from ..core.sizers import BasePositionSizer
from ..core.models import ExecutionResult, PerformanceMetrics
from .vectorized import VectorizedEngine
from .analyzer import PerformanceAnalyzer
from .portfolio import Portfolio
from ..visualization import plot_backtest_results, print_summary


class BacktestReport:
    """
    Presentation-layer object that bundles ExecutionResult and Metrics.
    Provides convenience methods for plotting and printing.
    """
    def __init__(self, result: ExecutionResult, metrics: PerformanceMetrics):
        self.result = result
        self.metrics = metrics

    def print_summary(self):
        """Prints the detailed console summary."""
        print_summary(self.result.strategy_name, self.metrics)

    def plot(self, panels: Optional[List[str]] = None, **kwargs):
        """Plots the backtest results."""
        if panels is None:
            panels = ['equity', 'drawdown', 'leverage']
        plot_backtest_results(results=self.result, panels=panels, **kwargs)


class BacktestRunner:
    """
    High-level facade for running backtests.
    Encapsulates Engine, Analyzer, and common configuration.
    """
    def __init__(
        self, 
        capital: Capital, 
        asset: Asset, 
        sizer: Optional[BasePositionSizer] = None
    ):
        self.capital = capital
        self.asset = asset
        self.engine = VectorizedEngine(capital=capital, position_sizer=sizer)
        self.analyzer = PerformanceAnalyzer(capital)

    def run(self, strategy) -> BacktestReport:
        """Run a single strategy and return a BacktestReport."""
        result = self.engine.run(strategy=strategy, asset=self.asset)
        metrics = self.analyzer.analyze(result)
        return BacktestReport(result, metrics)

    def run_multiple(self, strategies: Dict[str, object]) -> Dict[str, BacktestReport]:
        """Run multiple strategies and return a dict of reports."""
        reports = {}
        for name, strategy in strategies.items():
            reports[name] = self.run(strategy)
        return reports


class PortfolioRunner:
    """
    Facade for building and analyzing a portfolio of individual strategy results.
    Mirrors BacktestRunner: returns a BacktestReport with the same convenience API
    (`.metrics`, `.print_summary()`, `.plot()`).
    """

    def __init__(self, capital: Capital, analyzer: Optional[PerformanceAnalyzer] = None):
        self.capital = capital
        self.analyzer = analyzer or PerformanceAnalyzer(capital)

    def run(
        self,
        results: Union[List[ExecutionResult], Dict[str, ExecutionResult]],
        weights: Optional[Union[Dict[str, float], List[float]]] = None,
    ) -> BacktestReport:
        """
        Build a Portfolio from individual results and return a BacktestReport.

        Accepts either ExecutionResult objects or BacktestReport objects
        (list or dict); BacktestReport inputs are unwrapped to their `.result`.
        """
        if isinstance(results, dict):
            results = {name: (r.result if isinstance(r, BacktestReport) else r) for name, r in results.items()}
        else:
            results = [r.result if isinstance(r, BacktestReport) else r for r in results]

        portfolio = Portfolio(results, weights)
        metrics = self.analyzer.analyze(portfolio)
        return BacktestReport(portfolio, metrics)