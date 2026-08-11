# src/strategies/ewmac.py
import pandas as pd
import numpy as np
from .base import BaseStrategy

class EWMACStrategy(BaseStrategy):
    """
    EWMAC Trend Strength Strategy.
    Outputs a continuous signal between -1.0 and 1.0 based on trend strength.
    Stronger trends = larger signal = larger position (up to the Sizer's limits).
    """
    
    def __init__(
        self, 
        fast_window: int = 16, 
        slow_window: int = 64,
        vol_span: int = 25
    ):
        super().__init__(name=f"EWMAC ({fast_window}/{slow_window})")
        
        if fast_window >= slow_window:
            raise ValueError("fast_window must be strictly less than slow_window")
            
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.vol_span = vol_span

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close = data['close']
        
        # 1. Calculate EWMA Crossover
        fast_ewma = close.ewm(span=self.fast_window, adjust=False).mean()
        slow_ewma = close.ewm(span=self.slow_window, adjust=False).mean()
        crossover = fast_ewma - slow_ewma
        
        # 2. Calculate Volatility (to normalize the crossover)
        daily_returns = close.pct_change()
        annualized_vol = daily_returns.ewm(span=self.vol_span, adjust=False).std() * np.sqrt(252)
        
        # Prevent division by zero
        annualized_vol = annualized_vol.clip(lower=0.05) 
        
        # 3. Calculate Raw Trend Strength (Z-score-like)
        # This measures how many "volatility units" the trend is away from zero
        daily_price_vol = close * annualized_vol / np.sqrt(252)
        raw_strength = crossover / (daily_price_vol + 1e-9)
        
        # 4. Normalize to -1.0 to 1.0 range
        # We clip the raw strength. E.g., if strength > 2.0, we just output 1.0 (max long)
        # You can adjust this clip value to make the strategy more/less sensitive
        signal = raw_strength.clip(lower=-2.0, upper=2.0) / 2.0 
        
        # 5. Warm-up period
        min_warmup = max(self.slow_window, self.vol_span * 2)
        signal.iloc[:min_warmup] = 0.0
        
        return signal.fillna(0.0)