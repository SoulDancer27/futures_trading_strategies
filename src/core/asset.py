# src/core/config.py
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd

@dataclass
class Asset:
    """Represents the instrument being traded and its market microstructure."""
    ticker: str
    price_data: pd.Series           # The 'close' price series
    point_value: float = 1.0        # Contract multiplier
    
    commission_rate: Optional[float] = None
    commission_per_contract: Optional[float] = None
    slippage_rate: Optional[float] = None
    trading_days: int = 252

    def __post_init__(self):
        if self.price_data is None or self.price_data.empty:
            raise ValueError(f"Asset '{self.ticker}' requires non-empty price_data.")
        if self.commission_rate is not None and self.commission_per_contract is not None:
            raise ValueError(f"Asset '{self.ticker}': Specify either commission_rate OR commission_per_contract.")
        if self.point_value <= 0:
            raise ValueError(f"Asset '{self.ticker}' point_value must be > 0.")
