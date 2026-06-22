import pandas as pd
import numpy as np
from src.strategies.base import BaseStrategy
from typing import Dict, Any

class VolatilityScaledBNH(BaseStrategy):
    """
    Buy & Hold strategy with dynamic position sizing based on estimated volatility.
    Position size = Target Annual Volatility / Current Annual Volatility
    """
    def __init__(
        self,
        vol_method: str = 'ewma',
        window: int = 20,
        lambda_param: float = 0.94,
        target_annual_vol: float = 0.15,
        max_position: float = 1.0,
        name: str = "Vol-Adjusted B&H"
    ):
        super().__init__(name=name)
        self.vol_method = vol_method.lower()
        self.window = window
        self.lambda_param = lambda_param  # λ for EWMA (typically 0.94 for daily)
        self.target_annual_vol = target_annual_vol
        self.max_position = max_position  # Cap to prevent excessive leverage in low-vol regimes

    def _calculate_volatility(self, returns: pd.Series) -> pd.Series:
        """Vectorized volatility estimation matching your mathematical description."""
        if self.vol_method == 'sma':
            # Simple Rolling Standard Deviation
            return returns.rolling(window=self.window, min_periods=1).std()
            
        elif self.vol_method == 'ewma':
            # Exponentially Weighted Moving Variance (Exact match to prompt formula)
            # μ_t = λ*r_t + (1-λ)*μ_{t-1}
            # σ²_t = λ*(r_t - μ_t)² + (1-λ)*σ²_{t-1}
            alpha = 1.0 - self.lambda_param
            
            # EWMA Mean
            ewm_mean = returns.ewm(alpha=alpha, adjust=False, min_periods=1).mean()
            
            # EWMA Variance using squared deviations from EWMA mean
            squared_diff = (returns - ewm_mean) ** 2
            ewm_var = squared_diff.ewm(alpha=alpha, adjust=False, min_periods=1).mean()
            
            return np.sqrt(ewm_var)
            
        elif self.vol_method == 'pandas_ewm_std':
            # Direct pandas EWMA std (simplified, assumes zero mean for daily returns)
            return returns.ewm(span=self.window, min_periods=1).std()
            
        else:
            raise ValueError(f"Unsupported vol_method: {self.vol_method}")

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        # 1. Calculate daily returns
        returns = data['close'].pct_change()
        
        # 2. Estimate daily volatility
        daily_vol = self._calculate_volatility(returns)
        
        # 3. Annualize (√252 trading days)
        annual_vol = daily_vol * np.sqrt(252)
        
        # 4. Avoid division by zero during initialization
        annual_vol = annual_vol.clip(lower=1e-8)
        
        # 5. Volatility Targeting: Size = Target / Current
        position = self.target_annual_vol / annual_vol
        
        # 6. Apply cap (e.g., 1.0 = full capital, >1.0 allows leverage)
        position = position.clip(upper=self.max_position)
        
        # 7. Forward-fill initial NaNs from warm-up period
        position = position.ffill().fillna(self.max_position)
        
        return position

    def get_parameters(self) -> Dict[str, Any]:
        return {
            'vol_method': self.vol_method,
            'window': self.window,
            'lambda_param': self.lambda_param,
            'target_annual_vol': self.target_annual_vol,
            'max_position': self.max_position
        }