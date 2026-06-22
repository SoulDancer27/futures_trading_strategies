"""
Fixed Risk Position Sizing Strategy.

Implements the strategy from "Advanced Futures Trading Strategies":
- Volatility-targeted position sizing
- Dynamic adjustment based on capital, price, and volatility
- Risk constraints based on margin and leverage limits

Position sizing formula:
    N = (Capital × τ) ÷ (Multiplier × Price × σ_N)

Where:
    N = Number of contracts
    τ = Risk target (annualized std dev, e.g., 0.20 for 20%)
    σ_N = Instrument volatility (annualized std dev of returns)
    Multiplier = Contract multiplier (point value)
"""
import pandas as pd
import numpy as np
from typing import Optional
from .base import BaseStrategy  # Adjust import path if using flat structure

class FixedRiskPositionStrategy(BaseStrategy):
    """
    Buy and hold with variable position calculated using fixed risk estimate.
    
    Continuously adjusts position size to maintain constant risk exposure
    regardless of market volatility. Supports SMA and EWMA volatility estimation.
    """
    
    def __init__(
        self,
        initial_capital: float = 100_000.0,    # Starting capital for position sizing
        risk_target: float = 0.20,             # τ: Target annualized volatility (20%)
        volatility_window: int = 252,          # Lookback for σ_N calculation
        vol_method: str = 'sma',               # 'sma', 'ewma', or 'pandas_ewm_std'
        lambda_param: float = 0.94,            # λ decay factor for EWMA (0.94 ≈ 20-day SMA)
        multiplier: float = 1.0,               # Contract multiplier (point value)
        use_fixed_capital: bool = True,        # True = use initial_capital, False = use current equity
        margin_per_contract: Optional[float] = None,  # For risk constraints
        max_capital_loss: float = 0.50,        # Maximum capital loss (50%)
        expected_worst_return: float = 0.10,   # Expected worst return (10%)
        expected_sharpe: float = 0.5,          # Expected Sharpe ratio for Half Kelly
        min_contracts: int = 0,                # Minimum position size
        max_contracts: Optional[int] = None,    # Maximum position size (None = unlimited)
        max_leverage: Optional[float] = None   # Max notional/capital ratio (e.g., 2.0 = 2x)
    ):
        self.initial_capital = initial_capital
        self.risk_target = risk_target
        self.volatility_window = volatility_window
        self.vol_method = vol_method.lower()
        self.lambda_param = lambda_param
        self.multiplier = multiplier
        self.use_fixed_capital = use_fixed_capital
        self.margin_per_contract = margin_per_contract
        self.max_capital_loss = max_capital_loss
        self.expected_worst_return = expected_worst_return
        self.expected_sharpe = expected_sharpe
        self.min_contracts = min_contracts
        self.max_contracts = max_contracts
        self.max_leverage = max_leverage
        self.last_position = 0.0

    @property
    def name(self) -> str:
        cap_mode = "Fixed" if self.use_fixed_capital else "Compound"
        return f"FixedRisk(τ={self.risk_target*100:.0f}%, σ_method={self.vol_method.upper()}, {cap_mode})"
        
    def calculate_volatility(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate annualized volatility (σ_N) using the specified method.
        Fully vectorized for performance.
        """
        returns = data['close'].pct_change()
        alpha = 1.0 - self.lambda_param  # pandas ewm uses alpha = 1 - λ
        
        if self.vol_method == 'sma':
            daily_vol = returns.rolling(window=self.volatility_window, min_periods=1).std()
            
        elif self.vol_method == 'ewma':
            # Exact match to your formula: 
            # μ_t = λ*r_t + (1-λ)*μ_{t-1}
            # σ²_t = λ*(r_t - μ_t)² + (1-λ)*σ²_{t-1}
            ewm_mean = returns.ewm(alpha=alpha, adjust=False, min_periods=1).mean()
            squared_diff = (returns - ewm_mean) ** 2
            daily_vol = np.sqrt(squared_diff.ewm(alpha=alpha, adjust=False, min_periods=1).mean())
            
        elif self.vol_method == 'pandas_ewm_std':
            # Simplified pandas EWMA std (assumes ~0 mean for daily returns)
            daily_vol = returns.ewm(span=self.volatility_window, min_periods=1).std()
            
        else:
            raise ValueError(f"Unsupported vol_method: {self.vol_method}. Use 'sma', 'ewma', or 'pandas_ewm_std'")
            
        return daily_vol * np.sqrt(252)  # Annualize
    
    def calculate_risk_constraints(self, price: float, volatility: float) -> float:
        """Calculate maximum allowable risk based on constraints."""
        constraints = []
        
        if self.margin_per_contract is not None and self.margin_per_contract > 0:
            margin_risk = (self.multiplier * price * volatility) / self.margin_per_contract
            constraints.append(margin_risk)
        
        if volatility > 0:
            leverage_risk = (volatility * self.max_capital_loss) / self.expected_worst_return
            constraints.append(leverage_risk)
        
        constraints.append(0.5 * self.expected_sharpe)  # Half Kelly
        constraints.append(self.risk_target)            # Personal target
        
        return min(constraints)
    
    def calculate_position_size(self, capital: float, price: float, volatility: float) -> float:
        """Calculate optimal number of contracts (N)."""
        if volatility <= 0 or price <= 0:
            return 0.0
        
        effective_risk = self.calculate_risk_constraints(price, volatility)
        denominator = self.multiplier * price * volatility
        n_contracts = (capital * effective_risk) / denominator

        # Leverage cap
        if self.max_leverage is not None and self.max_leverage > 0:
            max_notional = capital * self.max_leverage
            n_contracts = min(n_contracts, max_notional / (price * self.multiplier))
        
        n_contracts = round(n_contracts)
        if self.max_contracts is not None:
            n_contracts = min(n_contracts, self.max_contracts)
        n_contracts = max(n_contracts, self.min_contracts)
        
        return float(n_contracts)
    
    def generate_positions(self, data: pd.DataFrame) -> pd.Series:
        """Generate continuous position series based on volatility targeting."""
        # Vectorized volatility calculation (done once)
        volatility = self.calculate_volatility(data)
        
        positions = []
        for i in range(len(data)):
            price = data['close'].iloc[i]
            vol = volatility.iloc[i]
            
            capital = self.initial_capital if self.use_fixed_capital else (
                self.initial_capital if i == 0 
                else self.initial_capital * (data['close'].iloc[i] / data['close'].iloc[0])
            )
            
            if pd.isna(vol) or vol <= 0:
                positions.append(0.0)
            else:
                positions.append(self.calculate_position_size(capital, price, vol))
        
        position_series = pd.Series(positions, index=data.index, dtype=float)
        self.last_position = position_series.iloc[-1] if len(position_series) > 0 else 0.0
        return position_series
    
    def get_detailed_info(self, data: pd.DataFrame, capital: float = None) -> dict:
        if capital is None:
            capital = self.initial_capital
            
        volatility = self.calculate_volatility(data)
        current_vol = volatility.iloc[-1] if len(volatility) > 0 else 0.0
        current_price = data['close'].iloc[-1]
        
        effective_risk = self.calculate_risk_constraints(current_price, current_vol)
        n_contracts = self.calculate_position_size(capital, current_price, current_vol)
        notional_value = n_contracts * self.multiplier * current_price
        risk_contribution = (notional_value * current_vol) / capital if capital > 0 else 0.0
        
        return {
            'current_price': current_price,
            'current_volatility': current_vol,
            'vol_method': self.vol_method,
            'risk_target': self.risk_target,
            'effective_risk': effective_risk,
            'capital_used': capital,
            'num_contracts': n_contracts,
            'notional_value': notional_value,
            'risk_contribution': risk_contribution
        }