# src/strategies/ma_crossover.py
import pandas as pd
import numpy as np
from typing import Literal
from .base import BaseStrategy

class MACrossoverStrategy(BaseStrategy):
    """
    Moving Average Crossover Strategy.
    Generates a binary signal: 1 (Long), 0 (Flat), -1 (Short).
    Position sizing and volatility targeting are handled by the Engine's Sizer.
    """
    
    def __init__(
        self, 
        short_window: int = 10, 
        long_window: int = 50, 
        ma_type: Literal['sma', 'ema'] = 'sma',
        mode: Literal['long_only', 'long_short'] = 'long_only'
    ):
        super().__init__(name=f"MA Crossover ({short_window}/{long_window})")
        
        if short_window >= long_window:
            raise ValueError("short_window must be strictly less than long_window")
            
        self.short_window = short_window
        self.long_window = long_window
        self.ma_type = ma_type
        self.mode = mode

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close = data['close']
        
        # 1. Calculate Moving Averages
        if self.ma_type == 'sma':
            short_ma = close.rolling(window=self.short_window).mean()
            long_ma = close.rolling(window=self.long_window).mean()
        else:
            short_ma = close.ewm(span=self.short_window, adjust=False).mean()
            long_ma = close.ewm(span=self.long_window, adjust=False).mean()
        
        # 2. Generate binary signal
        if self.mode == 'long_short':
            signal = (short_ma > long_ma).astype(float) - (short_ma < long_ma).astype(float)
        else:
            signal = (short_ma > long_ma).astype(float)
            
        # 3. Warm-up period (avoid NaNs at the start)
        min_warmup = max(self.short_window, self.long_window)
        signal.iloc[:min_warmup] = 0.0
        
        return signal.fillna(0.0)