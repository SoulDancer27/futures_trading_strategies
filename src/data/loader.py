"""
Data Loading Utilities.
"""
import pandas as pd
from pathlib import Path
from typing import Union

def load_simple_price_csv(
    file_path: Union[str, Path], 
    ticker: str = "UNKNOWN"
) -> pd.Series:
    """
    Load a simple 2-column CSV (date, price) with no headers.
    Returns a pure pandas Series of prices with a DatetimeIndex.
    """
    # Read CSV: no header, first column is date, second is price
    df = pd.read_csv(
        file_path, 
        header=None, 
        parse_dates=[0], 
        index_col=0
    )
    
    # Extract the price column (it's the only one left)
    price_series = df.iloc[:, 0]
    
    # Give it a name (optional, but good for debugging)
    price_series.name = ticker 
    
    # Clean up: sort and remove duplicates
    price_series = price_series.sort_index()
    price_series = price_series[~price_series.index.duplicated(keep='first')]
    
    return price_series