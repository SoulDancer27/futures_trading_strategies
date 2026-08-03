# scripts/convert_csv_format.py
import pandas as pd
import sys
from pathlib import Path

def convert_csv_format(input_file: str, output_file: str) -> None:
    """
    Convert CSV from format: YYYYMMDD;PRICE
    To format: YYYY-MM-DD,PRICE
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
    """
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    if not input_path.exists():
        print(f"❌ Error: Input file '{input_file}' not found!")
        sys.exit(1)
    
    try:
        # Read the incorrect format (semicolon separator, no header)
        df = pd.read_csv(input_path, sep=';', header=None, names=['date', 'price'])
        
        # Convert date from YYYYMMDD to YYYY-MM-DD
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
        
        # Format date as string
        df['date'] = df['date'].dt.strftime('%Y-%m-%d')
        
        # Write in correct format (comma separator, no header, no index)
        df.to_csv(output_path, sep=',', header=False, index=False)
        
        print(f"✅ Successfully converted:")
        print(f"   Input:  {input_file}")
        print(f"   Output: {output_file}")
        print(f"   Rows: {len(df)}")
        print(f"   Date range: {df['date'].iloc[0]} to {df['date'].iloc[-1]}")
        
    except Exception as e:
        print(f"❌ Error converting file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_csv_format.py <input_file> <output_file>")
        print("\nExample:")
        print("  python convert_csv_format.py data/MCFTR_old.csv data/MCFTR.csv")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    convert_csv_format(input_file, output_file)