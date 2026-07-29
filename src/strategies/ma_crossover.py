# src/strategies/ma_crossover.py
import pandas as pd
import numpy as np
from typing import Literal
from .base import BaseStrategy

class MACrossoverStrategy(BaseStrategy):
    """
    Moving Average Crossover Strategy with Volatility Targeting.
    
    Generates a binary signal (long/flat/short) and scales the position size 
    dynamically so that the instrument's contribution to portfolio risk 
    matches a predefined annual target (e.g., 20%).
    """
    
    def __init__(
        self, 
        short_window: int = 10, 
        long_window: int = 50, 
        ma_type: Literal['sma', 'ema'] = 'sma',
        mode: Literal['long_only', 'long_short'] = 'long_only',
        capital: float = 100_000.0,
        annual_target_risk: float = 0.20,  # 20% annual risk target
        vol_span: int = 25,                # Lookback window for volatility estimation
        name: str = "MA Crossover"
    ):
        self._name = name
        
        if short_window >= long_window:
            raise ValueError("short_window must be strictly less than long_window")
        if ma_type not in ['sma', 'ema']:
            raise ValueError("ma_type must be either 'sma' or 'ema'")
            
        self.short_window = short_window
        self.long_window = long_window
        self.ma_type = ma_type
        self.mode = mode
        self.capital = capital
        self.annual_target_risk = annual_target_risk
        self.vol_span = vol_span

    @property
    def name(self) -> str:
        return self._name

    def generate_positions(self, data: pd.DataFrame) -> pd.Series:
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
            signal = (short_ma > long_ma).astype(int) - (short_ma < long_ma).astype(int)
        else:
            signal = (short_ma > long_ma).astype(int)
        
        # 3. Volatility Targeting
        daily_returns = close.pct_change()
        annualized_vol = daily_returns.ewm(span=self.vol_span, adjust=False).std() * np.sqrt(252)
        
        # 🔧 FIX 1: Minimum volatility floor
        min_vol = 0.05  # 5% annualized
        annualized_vol = annualized_vol.clip(lower=min_vol)
        
        instrument_risk = close * annualized_vol
        target_risk_currency = self.capital * self.annual_target_risk
        
        base_position = target_risk_currency / (instrument_risk + 1e-9)
        positions = signal * base_position
        
        # 🔧 FIX 2: Maximum position cap (3x leverage)
        max_position = (self.capital * 3.0) / close
        positions = positions.clip(lower=-max_position, upper=max_position)
        
        #  FIX 3: Warm-up period
        min_warmup_days = max(self.short_window, self.long_window, self.vol_span * 2)
        positions.iloc[:min_warmup_days] = 0
        
        return positions.fillna(0)