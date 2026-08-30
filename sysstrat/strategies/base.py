# src/strategies/base.py
from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any

class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    Strategies are purely mathematical: they take data and output raw signals.
    They know nothing about capital, commissions, or position sizing.
    """
    
    def __init__(self, name: str = "BaseStrategy"):
        self.name = name

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate raw trading signals based on market data.
        
        Args:
            data: DataFrame containing at least a 'close' column.
            
        Returns:
            pd.Series: Raw signals (e.g., 1.0 for full long, 0.0 for flat, -1.0 for short).
                       The index must exactly match the input data index.
        """
        pass

    def get_parameters(self) -> Dict[str, Any]:
        """
        Return a dictionary of strategy parameters. 
        Useful for logging, database storage, and parameter optimization.
        """
        return {}