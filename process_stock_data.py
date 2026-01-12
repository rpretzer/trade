import yfinance as yf
import pandas as pd
import numpy as np
import time

# Define the stock symbols
symbols = ['AAPL', 'MSFT']

# Define the date range
start_date = '2020-01-01'
end_date = '2023-01-01'

# Download historical data
print(f"Downloading stock data for {', '.join(symbols)} from {start_date} to {end_date}...")

# Try downloading both together first, if that fails, download separately
data = None
max_retries = 3

for attempt in range(max_retries):
    try:
        data = yf.download(symbols, start=start_date, end=end_date, progress=False)
        if data is not None and not data.empty:
            break
    except Exception as e:
        print(f"Download attempt {attempt + 1} failed: {e}")
        if attempt < max_retries - 1:
            time.sleep(2)

# If batch download failed, try downloading stocks separately
if data is None or data.empty:
    print("\nBatch download failed, trying individual downloads...")
    aapl_data = None
    msft_data = None
    
    # Download AAPL
    for attempt in range(max_retries):
        try:
            aapl_data = yf.download('AAPL', start=start_date, end=end_date, progress=False)
            if aapl_data is not None and not aapl_data.empty:
                break
        except Exception as e:
            print(f"AAPL download attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
    
    # Download MSFT
    for attempt in range(max_retries):
        try:
            msft_data = yf.download('MSFT', start=start_date, end=end_date, progress=False)
            if msft_data is not None and not msft_data.empty:
                break
        except Exception as e:
            print(f"MSFT download attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
    
    # Combine individual downloads
    if aapl_data is not None and msft_data is not None:
        # Create a combined MultiIndex structure
        combined_data = {}
        for col in ['Close', 'Volume']:
            combined_data[(col, 'AAPL')] = aapl_data[col] if col in aapl_data.columns else aapl_data
            combined_data[(col, 'MSFT')] = msft_data[col] if col in msft_data.columns else msft_data
        data = pd.DataFrame(combined_data)
    elif aapl_data is not None:
        print("\nWarning: Only AAPL data downloaded successfully")
        data = aapl_data
    elif msft_data is not None:
        print("\nWarning: Only MSFT data downloaded successfully")
        data = msft_data

if data is None or data.empty:
    raise ValueError("Failed to download stock data after multiple attempts. Please check your internet connection and try again.")

print("\nInitial data shape:", data.shape)
print("Initial data columns:", data.columns.tolist())

# Extract Close prices, Volume for each stock
# yfinance returns a MultiIndex DataFrame when downloading multiple symbols
if isinstance(data.columns, pd.MultiIndex):
    # Get Close prices and Volume for each symbol
    aapl_close = data[('Close', 'AAPL')] if ('Close', 'AAPL') in data.columns else None
    msft_close = data[('Close', 'MSFT')] if ('Close', 'MSFT') in data.columns else None
    aapl_volume = data[('Volume', 'AAPL')] if ('Volume', 'AAPL') in data.columns else None
    msft_volume = data[('Volume', 'MSFT')] if ('Volume', 'MSFT') in data.columns else None
else:
    # Fallback if structure is different
    if 'Close' in data.columns and len(symbols) == 1:
        # Single stock download
        aapl_close = data['Close'] if 'AAPL' in str(symbols[0]).upper() else None
        msft_close = data['Close'] if 'MSFT' in str(symbols[0]).upper() else None
        aapl_volume = data['Volume'] if 'AAPL' in str(symbols[0]).upper() else None
        msft_volume = data['Volume'] if 'MSFT' in str(symbols[0]).upper() else None
    else:
        aapl_close = data['AAPL']['Close'] if 'AAPL' in str(data.columns) else None
        msft_close = data['MSFT']['Close'] if 'MSFT' in str(data.columns) else None
        aapl_volume = data['AAPL']['Volume'] if 'AAPL' in str(data.columns) else None
        msft_volume = data['MSFT']['Volume'] if 'MSFT' in str(data.columns) else None

# Validate we have data
if aapl_close is None and msft_close is None:
    raise ValueError("Failed to extract stock price data. Please check the data structure.")

# Create a DataFrame with the close prices and volumes
price_df_data = {}
if aapl_close is not None:
    price_df_data['AAPL'] = aapl_close
    price_df_data['AAPL_Volume'] = aapl_volume if aapl_volume is not None else pd.Series([np.nan] * len(aapl_close), index=aapl_close.index)
if msft_close is not None:
    price_df_data['MSFT'] = msft_close
    price_df_data['MSFT_Volume'] = msft_volume if msft_volume is not None else pd.Series([np.nan] * len(msft_close), index=msft_close.index)

price_df = pd.DataFrame(price_df_data)

# Align indices if we have both stocks
if 'AAPL' in price_df.columns and 'MSFT' in price_df.columns:
    price_df = price_df.sort_index()
    price_df = price_df.loc[price_df.index.intersection(price_df.index)]

print("\nPrice DataFrame shape:", price_df.shape)
print("\nFirst few rows of price DataFrame:")
print(price_df.head())

# Step 1: Calculate the daily price difference (AAPL - MSFT)
if 'AAPL' in price_df.columns and 'MSFT' in price_df.columns:
    price_df['Price_Difference'] = price_df['AAPL'] - price_df['MSFT']
else:
    raise ValueError("Need both AAPL and MSFT data to calculate price difference. Download failed for one or both stocks.")

# Step 2: Calculate moving averages
# 5-day moving average
if 'AAPL' in price_df.columns:
    price_df['AAPL_MA5'] = price_df['AAPL'].rolling(window=5).mean()
if 'MSFT' in price_df.columns:
    price_df['MSFT_MA5'] = price_df['MSFT'].rolling(window=5).mean()

# 20-day moving average
if 'AAPL' in price_df.columns:
    price_df['AAPL_MA20'] = price_df['AAPL'].rolling(window=20).mean()
if 'MSFT' in price_df.columns:
    price_df['MSFT_MA20'] = price_df['MSFT'].rolling(window=20).mean()

# Volume moving averages
if 'AAPL_Volume' in price_df.columns:
    price_df['AAPL_Volume_MA5'] = price_df['AAPL_Volume'].rolling(window=5).mean()
if 'MSFT_Volume' in price_df.columns:
    price_df['MSFT_Volume_MA5'] = price_df['MSFT_Volume'].rolling(window=5).mean()

print("\nAfter adding moving averages:")
print(price_df.head())

# Step 3: Remove any rows with missing values
initial_rows = len(price_df)
price_df_clean = price_df.dropna()

# Check if we have any data left
if len(price_df_clean) == 0:
    print("\n" + "="*70)
    print("ERROR: No valid data remaining after removing missing values!")
    print("="*70)
    print(f"\nTotal rows: {initial_rows}")
    print(f"AAPL data: {price_df['AAPL'].notna().sum()} valid rows" if 'AAPL' in price_df.columns else "AAPL: Not available")
    print(f"MSFT data: {price_df['MSFT'].notna().sum()} valid rows" if 'MSFT' in price_df.columns else "MSFT: Not available")
    raise ValueError("No valid data available. Please check your internet connection and try again. The yfinance library may be experiencing issues.")

rows_removed = initial_rows - len(price_df_clean)

print(f"\nRemoved {rows_removed} rows with missing values")
print(f"Clean data shape: {price_df_clean.shape}")
print("\nFirst few rows after removing missing values:")
print(price_df_clean.head())

# Step 4: Normalize the data so it's ready for a machine learning model
# We'll use StandardScaler for normalization (z-score normalization)
from sklearn.preprocessing import StandardScaler

# Select columns to normalize
# Features: AAPL, MSFT, Price_Difference, Volumes, Moving Averages
columns_to_normalize = [
    'AAPL', 'MSFT', 'Price_Difference',
    'AAPL_Volume', 'MSFT_Volume',
    'AAPL_MA5', 'MSFT_MA5', 'AAPL_MA20', 'MSFT_MA20',
    'AAPL_Volume_MA5', 'MSFT_Volume_MA5'
]

# Only normalize columns that exist
columns_to_normalize = [col for col in columns_to_normalize if col in price_df_clean.columns]

# Create scaler
scaler = StandardScaler()

# Fit and transform the data
normalized_data = scaler.fit_transform(price_df_clean[columns_to_normalize])

# Create a new DataFrame with normalized data
price_df_normalized = pd.DataFrame(
    normalized_data,
    columns=[col + '_normalized' for col in columns_to_normalize],
    index=price_df_clean.index
)

# Combine original data with normalized data
final_df = pd.concat([price_df_clean, price_df_normalized], axis=1)

print("\n" + "="*50)
print("FINAL PROCESSED DATA")
print("="*50)
print(f"\nFinal data shape: {final_df.shape}")
print(f"\nColumns: {final_df.columns.tolist()}")
print("\nFirst few rows of final processed data:")
print(final_df.head())
print("\nLast few rows of final processed data:")
print(final_df.tail())
print("\nSummary statistics:")
print(final_df.describe())

# Save the processed data to CSV
output_file = 'processed_stock_data.csv'
final_df.to_csv(output_file)
print(f"\nProcessed data saved to: {output_file}")
print(f"\nNumber of features for model: {len(columns_to_normalize)}")
