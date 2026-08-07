"""
Capital Models for Backtesting.

Implements the Strategy Pattern to separate Fixed Capital and Compounding 
methodologies. This eliminates conditional branching (if/else) in core 
metric calculations and ensures mathematical consistency.
"""
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod


class BaseCapitalModel(ABC):
    """
    Abstract base class for capital and return calculations.
    All capital models must implement these core mathematical operations.
    """
    
    @abstractmethod
    def calculate_returns(self, equity: pd.Series, daily_pnl: pd.Series, initial_capital: float) -> pd.Series:
        """Calculate the daily return series."""
        pass

    @abstractmethod
    def calculate_equity(self, returns: pd.Series, initial_capital: float) -> pd.Series:
        """Reconstruct the equity curve from daily returns."""
        pass

    @abstractmethod
    def calculate_total_return(self, returns: pd.Series) -> float:
        """Calculate the total cumulative return from the daily returns."""
        pass
        
    @abstractmethod
    def calculate_cagr(self, total_return: float, n_years: float) -> float:
        """Annualize the total return."""
        pass
        
    @abstractmethod
    def calculate_drawdown(self, equity: pd.Series, initial_capital: float) -> pd.Series:
        """Calculate the drawdown series."""
        pass

    @abstractmethod
    def calculate_leverage(self, notional: pd.Series, equity: pd.Series, initial_capital: float) -> pd.Series:
        """Calculate leverage based on the capital model's definition of capital."""
        pass


class FixedCapitalModel(BaseCapitalModel):
    """
    Robert Carver's Fixed Capital methodology.
    
    Assumes position sizes are based on initial capital and profits are 
    not reinvested. Returns are arithmetic, and drawdowns are measured 
    against the initial capital base.
    """
    
    def calculate_returns(self, equity: pd.Series, daily_pnl: pd.Series, initial_capital: float) -> pd.Series:
        # Daily P&L as a percentage of the constant initial capital
        return (daily_pnl / initial_capital).dropna()

    def calculate_equity(self, returns: pd.Series, initial_capital: float) -> pd.Series:
        # Arithmetic: Initial Capital + Cumulative PnL
        return initial_capital + (returns * initial_capital).cumsum()
        
    def calculate_total_return(self, returns: pd.Series) -> float:
        # Arithmetic sum of daily returns
        return returns.sum()
        
    def calculate_cagr(self, total_return: float, n_years: float) -> float:
        # Simple arithmetic annualization
        return total_return / n_years if n_years > 0 else 0.0
        
    def calculate_drawdown(self, equity: pd.Series, initial_capital: float) -> pd.Series:
        # Drawdown relative to the constant initial capital (base risk budget)
        rolling_max = equity.cummax()
        return (equity - rolling_max) / initial_capital

    def calculate_leverage(self, notional: pd.Series, equity: pd.Series, initial_capital: float) -> pd.Series:
        # Fixed Capital: Leverage relative to the constant initial capital
        return (notional / initial_capital).fillna(0).clip(lower=0)


class CompoundingCapitalModel(BaseCapitalModel):
    """
    Standard Geometric Compounding methodology.
    
    Assumes profits are reinvested. Returns are geometric (percentage 
    changes of the growing equity curve), and drawdowns are measured 
    from peak equity.
    """
    
    def calculate_returns(self, equity: pd.Series, daily_pnl: pd.Series, initial_capital: float) -> pd.Series:
        # Percentage change of the growing equity curve
        return equity.pct_change().dropna()

    def calculate_equity(self, returns: pd.Series, initial_capital: float) -> pd.Series:
        # Geometric: Initial Capital * Cumulative Product
        return initial_capital * (1 + returns).cumprod()
        
    def calculate_total_return(self, returns: pd.Series) -> float:
        # Geometric product of daily returns
        return (1 + returns).prod() - 1
        
    def calculate_cagr(self, total_return: float, n_years: float) -> float:
        # Geometric annualization
        return (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0.0
        
    def calculate_drawdown(self, equity: pd.Series, initial_capital: float) -> pd.Series:
        # Drawdown relative to the growing peak equity
        rolling_max = equity.cummax()
        return (equity - rolling_max) / rolling_max

    def calculate_leverage(self, notional: pd.Series, equity: pd.Series, initial_capital: float) -> pd.Series:
        # Compounding: Leverage relative to the growing equity curve
        return (notional / equity.replace(0, np.nan)).fillna(0).clip(lower=0)