# scripts/convert_csv_format.py
import pandas as pd
import sys
from pathlib import Path

def convert_csv_format(input_file: str, output_file: str) -> None:
    """
    Convert CSV from format: DD.MM.YYYY;PRICE
    To format: YYYY-MM-DD,PRICE
    
    The price column will be converted to a proper float (number).
    """
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    if not input_path.exists():
        print(f"❌ Error: Input file '{input_file}' not found!")
        sys.exit(1)
    
    try:
        # Read the file (semicolon separator, no header)
        # decimal=',' handles European decimal format (100,34 -> 100.34)
        df = pd.read_csv(
            input_path, 
            sep=';', 
            header=None, 
            names=['date', 'price'], 
            decimal=',',
            skipinitialspace=True
        )
        
        # Convert date from DD.MM.YYYY to datetime
        df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y')
        
        # Ensure price is float (numeric)
        df['price'] = df['price'].astype(float)
        
        # Format date as string in YYYY-MM-DD format
        df['date'] = df['date'].dt.strftime('%Y-%m-%d')
        
        # Write in correct format (comma separator, no header, no index)
        # float_format ensures numbers are written correctly
        df.to_csv(output_path, sep=',', header=False, index=False, float_format='%.2f')
        
        print(f"✅ Successfully converted:")
        print(f"   Input:  {input_file}")
        print(f"   Output: {output_file}")
        print(f"   Rows: {len(df)}")
        print(f"   Date range: {df['date'].iloc[0]} to {df['date'].iloc[-1]}")
        print(f"   Price column type: {df['price'].dtype}")
        
    except Exception as e:
        print(f"❌ Error converting file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_csv_format.py <input_file> <output_file>")
        print("\nExample:")
        print("  python convert_csv_format.py data/RGBITR1Y.csv data/RGBITR1Y_converted.csv")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    convert_csv_format(input_file, output_file)