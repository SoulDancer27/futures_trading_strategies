from abc import ABC, abstractmethod
import pandas as pd

class BaseStrategy(ABC):
    @abstractmethod
    def generate_positions(self, data: pd.DataFrame) -> pd.Series:
        """
        Return continuous position size (contracts) for each bar.
        Positive = long, Negative = short, 0 = flat.
        """
        pass
        
    @property
    @abstractmethod
    def name(self) -> str:
        pass