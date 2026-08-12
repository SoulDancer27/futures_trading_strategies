from dataclasses import dataclass, field
import pandas as pd
from abc import ABC, abstractmethod

class BaseCapitalModel(ABC):
    @abstractmethod
    def calculate_returns(self, equity: pd.Series, daily_pnl: pd.Series, initial_capital: float) -> pd.Series: pass
    @abstractmethod
    def calculate_total_return(self, returns: pd.Series) -> float: pass
    @abstractmethod
    def calculate_cagr(self, total_return: float, n_years: float) -> float: pass
    @abstractmethod
    def calculate_drawdown(self, equity: pd.Series, initial_capital: float) -> pd.Series: pass
    @abstractmethod
    def calculate_leverage(self, notional: pd.Series, equity: pd.Series, initial_capital: float) -> pd.Series: pass

class FixedCapitalModel(BaseCapitalModel):
    def calculate_returns(self, equity, daily_pnl, initial_capital):
        return (daily_pnl / initial_capital).dropna()
    def calculate_total_return(self, returns): return returns.sum()
    def calculate_cagr(self, total_return, n_years): return total_return / n_years if n_years > 0 else 0.0
    def calculate_drawdown(self, equity, initial_capital):
        drawdown = (equity - equity.cummax()) / initial_capital
        return drawdown.clip(lower=-1.0)
    def calculate_leverage(self, notional, equity, initial_capital):
        return (notional / initial_capital).fillna(0).clip(lower=0)

@dataclass
class Capital:
    initial_capital: float = 100_000.0
    risk_free_rate: float = 0.0
    capital_model: BaseCapitalModel = field(default_factory=FixedCapitalModel)
    
    def __post_init__(self):
        if self.initial_capital <= 0: raise ValueError("initial_capital must be > 0.")