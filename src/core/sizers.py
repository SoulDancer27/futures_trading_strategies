from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
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

class FixedContractsSizer(BasePositionSizer):
    """
    Calculates a fixed number of contracts based on initial capital and the FIRST price.
    Holds this exact number of contracts for the entire backtest (True Buy & Hold).
    """
    def __init__(self, max_allocation: float = 1.0):
        self.max_allocation = max_allocation

    def calculate_position(self, signal: pd.Series, capital: Capital, asset: Asset) -> pd.Series:
        # 1. Get the very first price to calculate initial size
        initial_price = asset.price_data.iloc[0]
        
        if initial_price <= 0 or asset.point_value <= 0:
            return pd.Series(0.0, index=signal.index)
            
        # 2. Calculate how many whole contracts we can buy with our allocation
        target_notional = capital.initial_capital * self.max_allocation
        num_contracts = np.floor(target_notional / (initial_price * asset.point_value))
        
        # 3. Return a constant series of this size
        # We multiply by 'signal' so that if the strategy says 0 (flat), we hold 0.
        return pd.Series(num_contracts, index=signal.index) * signal