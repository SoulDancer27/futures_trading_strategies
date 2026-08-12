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
        self.analyzer = PerformanceAnalyzer()

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