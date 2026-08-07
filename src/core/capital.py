# src/core/capital.py
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

from .sizers import BasePositionSizer, FixedFractionSizer

# ==========================================
# 1. BEHAVIOR: The Capital Models
# ==========================================
class BaseCapitalModel(ABC):
    """Abstract base class for capital calculations. 
    Kept for future use with the Event-Driven Engine."""
    
    @abstractmethod
    def calculate_returns(self, equity: pd.Series, daily_pnl: pd.Series, initial_capital: float) -> pd.Series:
        pass
    
    @abstractmethod
    def calculate_total_return(self, returns: pd.Series) -> float:
        pass
        
    @abstractmethod
    def calculate_cagr(self, total_return: float, n_years: float) -> float:
        pass
        
    @abstractmethod
    def calculate_drawdown(self, equity: pd.Series, initial_capital: float) -> pd.Series:
        pass

    @abstractmethod
    def calculate_leverage(self, notional: pd.Series, equity: pd.Series, initial_capital: float) -> pd.Series:
        pass


class FixedCapitalModel(BaseCapitalModel):
    """Carver's Fixed Capital methodology (Standard for Vectorized Engines)."""
    
    def calculate_returns(self, equity, daily_pnl, initial_capital):
        return (daily_pnl / initial_capital).dropna()
        
    def calculate_total_return(self, returns):
        return returns.sum()
        
    def calculate_cagr(self, total_return, n_years):
        return total_return / n_years if n_years > 0 else 0.0
        
    def calculate_drawdown(self, equity, initial_capital):
        return (equity - equity.cummax()) / initial_capital

    def calculate_leverage(self, notional, equity, initial_capital):
        return (notional / initial_capital).fillna(0).clip(lower=0)


# ==========================================
# 2. STATE: The Capital Configuration
# ==========================================
@dataclass
class Capital:
    """
    Represents the account configuration.
    Pure data container with sensible defaults.
    """
    initial_capital: float = 100_000.0
    risk_free_rate: float = 0.0
    
    # Defaults to Fixed Capital and Fixed Fraction Sizer
    capital_model: BaseCapitalModel = field(default_factory=FixedCapitalModel)
    position_sizer: BasePositionSizer = field(default_factory=FixedFractionSizer)

    def __post_init__(self):
        # Only basic validation remains
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be > 0.")