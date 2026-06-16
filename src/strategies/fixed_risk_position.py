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
from .base import BaseStrategy

class FixedRiskPositionStrategy(BaseStrategy):
    """
    Buy and hold with variable position calculated using fixed risk estimate.
    
    Continuously adjusts position size to maintain constant risk exposure
    regardless of market volatility.
    """
    
    def __init__(
        self,
        initial_capital: float = 100_000.0,    # Starting capital for position sizing
        risk_target: float = 0.20,             # τ: Target annualized volatility (20%)
        volatility_window: int = 252,          # Lookback for σ_N calculation
        multiplier: float = 1.0,               # Contract multiplier (point value)
        use_fixed_capital: bool = True,        # True = use initial_capital, False = use current equity
        margin_per_contract: Optional[float] = None,  # For risk constraints
        max_capital_loss: float = 0.50,        # Maximum capital loss (50%)
        expected_worst_return: float = 0.10,   # Expected worst return (10%)
        expected_sharpe: float = 0.5,          # Expected Sharpe ratio for Half Kelly
        min_contracts: int = 0,                # Minimum position size
        max_contracts: Optional[int] = None    # Maximum position size (None = unlimited)
    ):
        """
        Initialize strategy with risk parameters.
        
        Args:
            initial_capital: Starting capital for position sizing
            risk_target: Target annualized volatility (τ). Default 20%
            volatility_window: Number of days for rolling volatility calculation
            multiplier: Contract multiplier (point value). Default 1.0 for stocks
            use_fixed_capital: If True, always use initial_capital for sizing.
                             If False, use current equity (compounding).
            margin_per_contract: Margin requirement per contract (for risk limits)
            max_capital_loss: Maximum acceptable capital loss (e.g., 0.50 = 50%)
            expected_worst_return: Expected worst-case return for leverage calc
            expected_sharpe: Expected Sharpe ratio for Half Kelly calculation
            min_contracts: Minimum number of contracts to hold
            max_contracts: Maximum number of contracts (None = no limit)
        """
        self.initial_capital = initial_capital
        self.risk_target = risk_target
        self.volatility_window = volatility_window
        self.multiplier = multiplier
        self.use_fixed_capital = use_fixed_capital
        self.margin_per_contract = margin_per_contract
        self.max_capital_loss = max_capital_loss
        self.expected_worst_return = expected_worst_return
        self.expected_sharpe = expected_sharpe
        self.min_contracts = min_contracts
        self.max_contracts = max_contracts
        
        # Store last calculated position for tracking
        self.last_position = 0.0

    @property
    def name(self) -> str:
        """Dynamic strategy name based on key configuration parameters."""
        cap_mode = "Fixed" if self.use_fixed_capital else "Compound"
        return f"FixedRisk(risk={self.risk_target*100:.0f}%, window={self.volatility_window}d, {cap_mode})"
        
    def calculate_volatility(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate annualized volatility (σ_N) using rolling window.
        
        Args:
            data: DataFrame with 'close' price column
            
        Returns:
            Series of annualized volatility estimates
        """
        # Calculate daily returns
        returns = data['close'].pct_change()
        
        # Calculate rolling standard deviation
        rolling_std = returns.rolling(window=self.volatility_window).std()
        
        # Annualize (252 trading days)
        annualized_vol = rolling_std * np.sqrt(252)
        
        return annualized_vol
    
    def calculate_risk_constraints(self, price: float, volatility: float) -> float:
        """
        Calculate maximum allowable risk based on constraints.
        
        Returns the minimum of:
        1. Risk possible given margin levels
        2. Risk possible given prudent leverage
        3. Optimal risk given expected performance (Half Kelly)
        
        Args:
            price: Current instrument price
            volatility: Current annualized volatility (σ_N)
            
        Returns:
            Maximum allowable risk target (τ)
        """
        constraints = []
        
        # 1. Risk possible given margin levels
        if self.margin_per_contract is not None and self.margin_per_contract > 0:
            margin_risk = (self.multiplier * price * volatility) / self.margin_per_contract
            constraints.append(margin_risk)
        
        # 2. Risk possible given prudent leverage
        if volatility > 0:
            leverage_risk = (volatility * self.max_capital_loss) / self.expected_worst_return
            constraints.append(leverage_risk)
        
        # 3. Optimal risk given expected performance (Half Kelly)
        half_kelly_risk = 0.5 * self.expected_sharpe
        constraints.append(half_kelly_risk)
        
        # 4. Personal risk appetite (the risk_target parameter)
        constraints.append(self.risk_target)
        
        # Return minimum of all constraints
        return min(constraints) if constraints else self.risk_target
    
    def calculate_position_size(
        self, 
        capital: float, 
        price: float, 
        volatility: float
    ) -> float:
        """
        Calculate optimal number of contracts (N).
        
        Formula: N = (Capital × τ) ÷ (Multiplier × Price × σ_N)
        
        Args:
            capital: Current trading capital (initial or running equity)
            price: Current instrument price
            volatility: Current annualized volatility (σ_N)
            
        Returns:
            Number of contracts (rounded to nearest whole number)
        """
        if volatility <= 0 or price <= 0:
            return 0.0
        
        # Calculate effective risk target (minimum of constraints)
        effective_risk = self.calculate_risk_constraints(price, volatility)
        
        # Position sizing formula (no FX rate - using nominal values)
        denominator = self.multiplier * price * volatility
        n_contracts = (capital * effective_risk) / denominator
        
        # Round to nearest whole contract
        n_contracts = round(n_contracts)
        
        # Apply min/max constraints
        if self.max_contracts is not None:
            n_contracts = min(n_contracts, self.max_contracts)
        n_contracts = max(n_contracts, self.min_contracts)
        
        return float(n_contracts)
    
    def generate_positions(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate continuous position series based on volatility targeting.
        
        Args:
            data: DataFrame with 'close' price column and DatetimeIndex
            
        Returns:
            Series of position sizes (number of contracts) for each time point
        """
        # Calculate volatility series
        volatility = self.calculate_volatility(data)
        
        positions = []
        current_equity = self.initial_capital
        
        for i in range(len(data)):
            price = data['close'].iloc[i]
            vol = volatility.iloc[i]
            
            # Determine capital to use
            if self.use_fixed_capital:
                # Always use initial capital (no compounding)
                capital = self.initial_capital
            else:
                # Use current equity (compounding)
                # For position generation, we approximate equity based on price changes
                if i == 0:
                    capital = self.initial_capital
                else:
                    # Simple approximation: scale capital by price change
                    price_return = data['close'].iloc[i] / data['close'].iloc[0]
                    capital = self.initial_capital * price_return
            
            if pd.isna(vol) or vol <= 0:
                # Not enough data for volatility estimate
                positions.append(0.0)
            else:
                n_contracts = self.calculate_position_size(capital, price, vol)
                positions.append(n_contracts)
        
        position_series = pd.Series(positions, index=data.index, dtype=float)
        
        # Store last position for reference
        self.last_position = position_series.iloc[-1] if len(position_series) > 0 else 0.0
        
        return position_series
    
    def get_detailed_info(self, data: pd.DataFrame, capital: float = None) -> dict:
        """
        Get detailed information about current position sizing.
        
        Useful for debugging and understanding the strategy behavior.
        
        Args:
            data: DataFrame with price data
            capital: Current capital (uses initial_capital if None)
            
        Returns:
            Dictionary with position sizing details
        """
        if capital is None:
            capital = self.initial_capital
            
        volatility = self.calculate_volatility(data)
        current_vol = volatility.iloc[-1] if len(volatility) > 0 else 0.0
        current_price = data['close'].iloc[-1]
        
        effective_risk = self.calculate_risk_constraints(current_price, current_vol)
        n_contracts = self.calculate_position_size(capital, current_price, current_vol)
        
        # Calculate notional value
        notional_value = n_contracts * self.multiplier * current_price
        
        # Calculate risk contribution
        risk_contribution = (notional_value * current_vol) / capital if capital > 0 else 0.0
        
        return {
            'current_price': current_price,
            'current_volatility': current_vol,
            'risk_target': self.risk_target,
            'effective_risk': effective_risk,
            'capital_used': capital,
            'use_fixed_capital': self.use_fixed_capital,
            'num_contracts': n_contracts,
            'notional_value': notional_value,
            'risk_contribution': risk_contribution,
            'margin_per_contract': self.margin_per_contract,
            'multiplier': self.multiplier
        }
