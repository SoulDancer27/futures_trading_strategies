import pandas as pd
from pathlib import Path
from loguru import logger

def load_csv(file_path: str, date_col: str = "Date", price_col: str = "Close") -> pd.DataFrame:
    """Load simple price series from CSV. Returns DataFrame with DatetimeIndex & 'close'."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
        
    df = pd.read_csv(path)
    
    # Parse datetime & set index
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    
    # Standardize column name
    if price_col != "close":
        df["close"] = df[price_col]
        
    return df[["close"]]

"""
Simple CSV data loader for 2-column (date, price) files.
No headers expected. Outputs DatetimeIndex DataFrame with 'close' column.
"""
def load_simple_price_csv(
    file_path: str, 
    delimiter: str = ",", 
    date_format: str = None
) -> pd.DataFrame:
    """
    Load a headerless CSV with [date, price] columns.
    
    Args:
        file_path: Path to CSV file
        delimiter: Column separator (default: ',')
        date_format: Optional pandas datetime format (e.g., '%Y%m%d', '%d.%m.%Y')
        
    Returns:
        DataFrame with DatetimeIndex and 'close' column, sorted & deduplicated
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
        
    logger.info(f"📥 Loading simple price data: {path.name}")
    
    # 1. Read headerless CSV
    df = pd.read_csv(
        path,
        sep=delimiter,
        header=None,
        names=["date", "price"],
        dtype={"price": float},
        skipinitialspace=True
    )
    
    # 2. Parse datetime
    if date_format:
        df["date"] = pd.to_datetime(df["date"], format=date_format)
    else:
        df["date"] = pd.to_datetime(df["date"])
        
    # 3. Clean & standardize
    df = df.dropna(subset=["date", "price"])
    df = df[df["price"] > 0]  # Filter invalid prices
    
    # 4. Set index & sort
    df = df.set_index("date").sort_index()
    
    # 5. Normalize index (matches your project's timezone/duplicate handling)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
        
    df = df[~df.index.duplicated(keep="first")]
    
    # 6. Standardize column name for engine compatibility
    df.rename(columns={"price": "close"}, inplace=True)
    
    logger.info(f"✅ Loaded {len(df)} bars | Range: {df.index[0].date()} → {df.index[-1].date()}")
    return df[["close"]]