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

class FixedRiskSizer(BasePositionSizer):
    """
    Advanced Fixed Risk Position Sizing (Carver's Methodology).
    
    Calculates position size to maintain a constant risk exposure, 
    adjusting dynamically for market volatility, margin limits, and leverage caps.
    """
    def __init__(
        self,
        risk_target: float = 0.20,          # Target annualized risk (e.g., 0.20 = 20%)
        vol_method: str = 'blended',        # 'sma', 'ewma', 'pandas_ewm_std', 'blended'
        volatility_window: int = 252,       # Lookback for SMA/EWMA
        short_span: int = 32,               # Fast EWMA span (approx 1 month)
        long_span: int = 252,               # Slow EWMA span (approx 1 year)
        margin_per_contract: float = None,  # Margin requirement per contract
        max_capital_loss: float = 0.50,     # Max capital loss constraint
        expected_worst_return: float = 0.10,# Expected worst return for leverage constraint
        expected_sharpe: float = 0.5,       # Expected Sharpe for Half Kelly constraint
        min_contracts: int = 0,
        max_contracts: int = None,
        max_leverage: float = None          # Absolute leverage cap
    ):
        self.risk_target = risk_target
        self.vol_method = vol_method.lower()
        self.volatility_window = volatility_window
        self.short_span = short_span
        self.long_span = long_span
        
        self.margin_per_contract = margin_per_contract
        self.max_capital_loss = max_capital_loss
        self.expected_worst_return = expected_worst_return
        self.expected_sharpe = expected_sharpe
        
        self.min_contracts = min_contracts
        self.max_contracts = max_contracts
        self.max_leverage = max_leverage

    def _calculate_volatility(self, price_data: pd.Series, trading_days: int) -> pd.Series:
        """Vectorized annualized volatility calculation."""
        returns = price_data.pct_change()
        
        if self.vol_method == 'sma':
            daily_vol = returns.rolling(window=self.volatility_window, min_periods=1).std()
        elif self.vol_method == 'ewma':
            # RiskMetrics EWMA (lambda = 0.94)
            daily_vol = returns.ewm(alpha=0.06, adjust=False, min_periods=1).std()
        elif self.vol_method == 'pandas_ewm_std':
            daily_vol = returns.ewm(span=self.volatility_window, min_periods=1).std()
        elif self.vol_method == 'blended':
            # Carver's Blended Vol: 70% short-run + 30% long-run
            short_vol = returns.ewm(span=self.short_span, min_periods=1).std()
            long_vol = returns.ewm(span=self.long_span, min_periods=1).std()
            daily_vol = np.sqrt(0.7 * (short_vol ** 2) + 0.3 * (long_vol ** 2))
        else:
            raise ValueError(f"Unsupported vol_method: {self.vol_method}")
            
        return daily_vol * np.sqrt(trading_days)

    def calculate_position(self, signal: pd.Series, capital: Capital, asset: Asset) -> pd.Series:
        # 1. Calculate Volatility Series
        vol = self._calculate_volatility(asset.price_data, asset.trading_days)
        vol = vol.replace(0, np.nan).fillna(self.risk_target) # Prevent div by zero
        
        price = asset.price_data
        multiplier = asset.point_value
        
        # 2. Calculate Effective Risk Constraints (Vectorized)
        constraints = [pd.Series(self.risk_target, index=price.index)] # Personal target
        
        if self.margin_per_contract and self.margin_per_contract > 0:
            margin_risk = (multiplier * price * vol) / self.margin_per_contract
            constraints.append(margin_risk)
            
        if vol.any() > 0:
            leverage_risk = (vol * self.max_capital_loss) / self.expected_worst_return
            constraints.append(leverage_risk)
            
        half_kelly = 0.5 * self.expected_sharpe
        constraints.append(pd.Series(half_kelly, index=price.index))
        
        # Take the minimum constraint for each day
        effective_risk = pd.concat(constraints, axis=1).min(axis=1)
        
        # 3. Calculate Raw Contracts: N = (Capital * effective_risk) / (Multiplier * Price * Vol)
        denominator = multiplier * price * vol
        raw_contracts = (capital.initial_capital * effective_risk) / denominator
        
        # 4. Apply Leverage Cap
        if self.max_leverage is not None and self.max_leverage > 0:
            max_notional = capital.initial_capital * self.max_leverage
            max_contracts_lev = max_notional / (price * multiplier)
            raw_contracts = raw_contracts.clip(upper=max_contracts_lev)
            
        # 5. Round and Apply Absolute Min/Max
        final_contracts = raw_contracts.round(0).clip(lower=self.min_contracts)
        if self.max_contracts is not None:
            final_contracts = final_contracts.clip(upper=self.max_contracts)
            
        # 6. Multiply by Signal (so 0 signal = 0 contracts) and fill NaNs
        return (final_contracts * signal).fillna(0)