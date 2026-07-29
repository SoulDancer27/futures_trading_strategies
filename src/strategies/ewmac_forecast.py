# src/strategies/ewmac_forecast.py
import pandas as pd
import numpy as np
from .base import BaseStrategy

class EWMACForecastStrategy(BaseStrategy):
    """
    EWMAC Forecast Strategy (Robert Carver's Systematic Trading).
    
    Generates a continuous position size based on the strength of the trend.
    1. Calculates the EWMA crossover (Fast EWMA - Slow EWMA).
    2. Risk-adjusts the crossover by dividing by daily price volatility (σp).
    3. Scales the raw forecast to have an average absolute value of 10.
    4. Caps the forecast to prevent extreme leverage.
    5. Scales the final position size (number of contracts) based on the 
       capped forecast, target risk, and instrument volatility.
    """
    
    def __init__(
        self, 
        fast_window: int = 16, 
        slow_window: int = 64,
        vol_span: int = 25,
        forecast_scalar: float = 1.9,
        max_forecast: float = 20.0,
        capital: float = 100_000.0,
        annual_target_risk: float = 0.20,  # 20% annual risk target
        name: str = "EWMAC Forecast"
    ):
        # Store the name internally
        self._name = name
        
        if fast_window >= slow_window:
            raise ValueError("fast_window must be strictly less than slow_window")
            
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.vol_span = vol_span
        self.forecast_scalar = forecast_scalar
        self.max_forecast = max_forecast
        self.capital = capital
        self.annual_target_risk = annual_target_risk

    @property
    def name(self) -> str:
        """Implementation of the abstract name property."""
        return self._name

    def generate_positions(self, data: pd.DataFrame) -> pd.Series:
        close = data['close']
        
        # 1. Calculate EWMA Crossover
        fast_ewma = close.ewm(span=self.fast_window, adjust=False).mean()
        slow_ewma = close.ewm(span=self.slow_window, adjust=False).mean()
        crossover = fast_ewma - slow_ewma
        
        # 2. Calculate Volatility
        daily_returns = close.pct_change()
        annualized_vol_pct = daily_returns.ewm(span=self.vol_span, adjust=False).std() * np.sqrt(252)
        
        # 🔧 FIX 1: Apply minimum volatility floor (e.g., 5% annualized)
        min_vol = 0.05  # 5% minimum annualized volatility
        annualized_vol_pct = annualized_vol_pct.clip(lower=min_vol)
        
        # Daily price volatility
        daily_price_vol = close * annualized_vol_pct / np.sqrt(252)
        
        # 3. Calculate Raw Forecast
        raw_forecast = crossover / (daily_price_vol + 1e-9)
        
        # 4. Scale and Cap Forecast
        scaled_forecast = raw_forecast * self.forecast_scalar
        capped_forecast = scaled_forecast.clip(lower=-self.max_forecast, upper=self.max_forecast)
        
        # 5. Calculate Position Size
        instrument_value_risk = close * annualized_vol_pct
        target_risk_currency = self.capital * self.annual_target_risk
        
        base_position = target_risk_currency / (instrument_value_risk + 1e-9)
        positions = (capped_forecast / 10.0) * base_position
        
        # 🔧 FIX 2: Cap maximum position size (e.g., 3x leverage)
        max_position = (self.capital * 3.0) / close  # 3x leverage cap
        positions = positions.clip(lower=-max_position, upper=max_position)
        
        # 🔧 FIX 3: Warm-up period - no positions until we have enough data
        # Require at least (vol_span × 2) days for reliable volatility
        min_warmup_days = self.vol_span * 2
        positions.iloc[:min_warmup_days] = 0
        
        return positions.fillna(0)