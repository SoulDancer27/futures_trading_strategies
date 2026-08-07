from abc import ABC, abstractmethod
import pandas as pd
from .asset import Capital, Asset

class BasePositionSizer(ABC):
    """
    Translates a raw strategy signal into actual position size (contracts/shares).
    This is the bridge between the Strategy (Signal) and the Capital (Money).
    """
    @abstractmethod
    def calculate_position(self, signal: pd.Series, capital: Capital, asset: Asset) -> pd.Series:
        pass

class FixedFractionSizer(BasePositionSizer):
    """
    Allocates a fixed percentage of initial capital per unit of signal.
    Example: If signal is 1.0 and max_allocation is 0.5, it uses 50% of capital.
    """

    def __init__(self, max_allocation: float = 1.0):
        self.max_allocation = max_allocation # e.g., 1.0 = 100% of capital

    def calculate_position(self, signal: pd.Series, capital: Capital, asset: Asset) -> pd.Series:
        # Target notional value = Signal * Max Allocation * Initial Capital
        target_notional = signal * self.max_allocation * capital.initial_capital
        
        # Convert notional to contracts: Notional / (Price * Point Value)
        # We use a small epsilon to avoid division by zero if price is 0
        price_safe = asset.price_data.replace(0, float('nan'))
        contracts = target_notional / (price_safe * asset.point_value)
        
        return contracts.fillna(0).round(0) # Round to whole contracts