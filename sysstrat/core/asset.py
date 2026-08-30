# src/core/config.py
from dataclasses import dataclass, field, replace
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
        # Name the price series by its ticker (unless already meaningfully named),
        # so downstream series (e.g. pct_change()) carry a useful name.
        if self.price_data.name in (None, "UNKNOWN"):
            self.price_data = self.price_data.rename(self.ticker)
        if self.commission_rate is not None and self.commission_per_contract is not None:
            raise ValueError(f"Asset '{self.ticker}': Specify either commission_rate OR commission_per_contract.")
        if self.point_value <= 0:
            raise ValueError(f"Asset '{self.ticker}' point_value must be > 0.")

    def slice(self, start=None, end=None) -> "Asset":
        """
        Return a copy of this asset with price_data restricted to [start, end]
        (inclusive). Other fields (point_value, commission, slippage, etc.) are
        preserved. Useful for aligning several assets to a common backtest period.
        """
        data = self.price_data
        if start is not None:
            data = data.loc[start:]
        if end is not None:
            data = data.loc[:end]
        return replace(self, price_data=data)
