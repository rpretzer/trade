import yfinance as yf
import pandas as pd
import numpy as np
import time
import os
from datetime import datetime, timedelta
import pytz
from data_validation import (
    validate_price_data,
    validate_technical_indicators,
    print_validation_report,
    DataQualityError
)

# Try to import market calendar library
try:
    import pandas_market_calendars as mcal
    MARKET_CALENDAR_AVAILABLE = True
except ImportError:
    MARKET_CALENDAR_AVAILABLE = False
    print("Warning: pandas_market_calendars not available. Install with: pip install pandas_market_calendars")
    print("Will use simple date calculations instead.")

# Import our custom modules
try:
    from technical_indicators import add_technical_indicators
except ImportError:
    print("Warning: technical_indicators module not found. Technical indicators will be calculated inline.")
    add_technical_indicators = None

try:
    from sentiment_analysis import fetch_reddit_sentiment_for_symbols
    REDDIT_AVAILABLE = True
except ImportError:
    print("Warning: sentiment_analysis module not found. Reddit sentiment will be skipped.")
    REDDIT_AVAILABLE = False

# Function to get selected stocks from config file
def get_selected_stocks():
    """Get the currently selected stocks from config file."""
    config_file = 'selected_stocks.txt'
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                line = f.read().strip()
                if ',' in line:
                    stocks = [s.strip().upper() for s in line.split(',')]
                    if len(stocks) >= 2:
                        return stocks[0], stocks[1]
        except Exception:
            pass
    # Default stocks
    from constants import TICKER_LONG, TICKER_SHORT
    return TICKER_LONG, TICKER_SHORT

# Function to get last trading day for a market
def get_last_trading_day(market='NYSE', timezone_str='America/New_York'):
    """
    Get the last trading day for a given market.
    
    Args:
        market: Market identifier ('NYSE', 'NASDAQ', 'NIKKEI', 'LSE', etc.)
        timezone_str: Timezone string (e.g., 'America/New_York', 'Asia/Tokyo', 'Europe/London')
        
    Returns:
        datetime object of the last trading day
    """
    if MARKET_CALENDAR_AVAILABLE:
        try:
            # Map market names to pandas_market_calendars identifiers
            market_map = {
                'NYSE': 'NYSE',
                'NASDAQ': 'NASDAQ',
                'NIKKEI': 'OSE',  # Osaka Stock Exchange (Japan)
                'LSE': 'LSE',     # London Stock Exchange
                'TSX': 'TSX',     # Toronto Stock Exchange
                'ASX': 'ASX',     # Australian Stock Exchange
            }
            
            market_id = market_map.get(market.upper(), 'NYSE')
            calendar = mcal.get_calendar(market_id)
            
            # Get timezone
            tz = pytz.timezone(timezone_str)
            now = datetime.now(tz)
            
            # Get trading schedule for recent period
            # Look back up to 10 days to find last trading day
            end = now
            start = end - timedelta(days=10)
            
            schedule = calendar.schedule(start_date=start.strftime('%Y-%m-%d'),
                                        end_date=end.strftime('%Y-%m-%d'))
            
            if not schedule.empty:
                # Get the last trading day
                last_trading_day = schedule.index[-1]
                # Convert to timezone-aware datetime, then to naive for comparison
                if hasattr(last_trading_day, 'tz_localize'):
                    last_trading_day = last_trading_day.tz_localize(None)
                return last_trading_day.to_pydatetime() if hasattr(last_trading_day, 'to_pydatetime') else last_trading_day
            else:
                # Fallback: go back a few days
                return (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        except Exception as e:
            print(f"Warning: Could not get market calendar for {market}: {e}")
            # Fallback to simple calculation
            pass
    
    # Fallback: simple calculation (yesterday, accounting for weekends)
    tz = pytz.timezone(timezone_str)
    now = datetime.now(tz)
    yesterday = now - timedelta(days=1)
    
    # If yesterday was Saturday or Sunday, go back to Friday
    while yesterday.weekday() >= 5:  # Saturday = 5, Sunday = 6
        yesterday = yesterday - timedelta(days=1)
    
    return yesterday.replace(hour=0, minute=0, second=0, microsecond=0)

# Function to get market configuration
def get_market_config():
    """Get market configuration from config file or use defaults."""
    config_file = 'market_config.txt'
    
    # Defaults
    market = 'NYSE'
    timezone_str = 'America/New_York'
    
    # Try to read from config file
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('MARKET='):
                        market = line.split('=', 1)[1].strip()
                    elif line.startswith('TIMEZONE='):
                        timezone_str = line.split('=', 1)[1].strip()
        except Exception as e:
            print(f"Warning: Could not read market config: {e}. Using defaults.")
    
    return market, timezone_str

# Function to get date range from config or use defaults
def get_date_range():
    """
    Get date range from config file or use defaults.
    Default: Last trading day - 3 years to last trading day (yesterday).
    """
    config_file = 'date_range_config.txt'
    
    # Get market configuration
    market, timezone_str = get_market_config()
    
    # Get last trading day
    last_trading_day = get_last_trading_day(market, timezone_str)
    
    # Default: 3 years back from last trading day
    end_date = last_trading_day
    start_date = end_date - timedelta(days=3*365)  # Approximately 3 years
    
    # Try to read from config file
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('START_DATE='):
                        start_date_str = line.split('=', 1)[1].strip()
                        try:
                            start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
                        except ValueError:
                            print(f"Warning: Invalid START_DATE format in config. Using default.")
                    elif line.startswith('END_DATE='):
                        end_date_str = line.split('=', 1)[1].strip()
                        try:
                            end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
                        except ValueError:
                            print(f"Warning: Invalid END_DATE format in config. Using default.")
        except Exception as e:
            print(f"Warning: Could not read date range config: {e}. Using defaults.")
    
    # Format dates for yfinance
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    
    return start_date_str, end_date_str, market, timezone_str

# Get selected stocks dynamically
stock1, stock2 = get_selected_stocks()
symbols = [stock1, stock2]

print(f"Processing stocks: {stock1} and {stock2}")

# Get date range (configurable, defaults to last trading day - 3 years)
start_date, end_date, market, timezone_str = get_date_range()
print(f"Market: {market} (Timezone: {timezone_str})")
print(f"Date range: {start_date} to {end_date} (last trading day: {end_date})")

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
    stock1_data = None
    stock2_data = None
    
    # Download stock1
    for attempt in range(max_retries):
        try:
            stock1_data = yf.download(stock1, start=start_date, end=end_date, progress=False)
            if stock1_data is not None and not stock1_data.empty:
                break
        except Exception as e:
            print(f"{stock1} download attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
    
    # Download stock2
    for attempt in range(max_retries):
        try:
            stock2_data = yf.download(stock2, start=start_date, end=end_date, progress=False)
            if stock2_data is not None and not stock2_data.empty:
                break
        except Exception as e:
            print(f"{stock2} download attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
    
    # Add delay between downloads to prevent rate limiting
    time.sleep(5)
    
    # Combine individual downloads
    if stock1_data is not None and stock2_data is not None:
        # Create a combined MultiIndex structure
        combined_data = {}
        for col in ['Close', 'Volume']:
            combined_data[(col, stock1)] = stock1_data[col] if col in stock1_data.columns else stock1_data
            combined_data[(col, stock2)] = stock2_data[col] if col in stock2_data.columns else stock2_data
        data = pd.DataFrame(combined_data)
    elif stock1_data is not None:
        print(f"\nWarning: Only {stock1} data downloaded successfully")
        data = stock1_data
    elif stock2_data is not None:
        print(f"\nWarning: Only {stock2} data downloaded successfully")
        data = stock2_data

if data is None or data.empty:
    raise ValueError("Failed to download stock data after multiple attempts. Please check your internet connection and try again.")

print("\nInitial data shape:", data.shape)
print("Initial data columns:", data.columns.tolist())

# Extract Close prices, Volume for each stock
# yfinance returns a MultiIndex DataFrame when downloading multiple symbols
if isinstance(data.columns, pd.MultiIndex):
    # Get Close prices and Volume for each symbol
    stock1_close = data[('Close', stock1)] if ('Close', stock1) in data.columns else None
    stock2_close = data[('Close', stock2)] if ('Close', stock2) in data.columns else None
    stock1_volume = data[('Volume', stock1)] if ('Volume', stock1) in data.columns else None
    stock2_volume = data[('Volume', stock2)] if ('Volume', stock2) in data.columns else None
else:
    # Fallback if structure is different
    if 'Close' in data.columns and len(symbols) == 1:
        # Single stock download
        stock1_close = data['Close'] if stock1 in str(symbols[0]).upper() else None
        stock2_close = data['Close'] if stock2 in str(symbols[0]).upper() else None
        stock1_volume = data['Volume'] if stock1 in str(symbols[0]).upper() else None
        stock2_volume = data['Volume'] if stock2 in str(symbols[0]).upper() else None
    else:
        stock1_close = data[stock1]['Close'] if stock1 in str(data.columns) else None
        stock2_close = data[stock2]['Close'] if stock2 in str(data.columns) else None
        stock1_volume = data[stock1]['Volume'] if stock1 in str(data.columns) else None
        stock2_volume = data[stock2]['Volume'] if stock2 in str(data.columns) else None

# Validate we have data
if stock1_close is None and stock2_close is None:
    raise ValueError("Failed to extract stock price data. Please check the data structure.")

# Create a DataFrame with the close prices and volumes
price_df_data = {}
if stock1_close is not None:
    price_df_data[stock1] = stock1_close
    price_df_data[f'{stock1}_Volume'] = stock1_volume if stock1_volume is not None else pd.Series([np.nan] * len(stock1_close), index=stock1_close.index)
if stock2_close is not None:
    price_df_data[stock2] = stock2_close
    price_df_data[f'{stock2}_Volume'] = stock2_volume if stock2_volume is not None else pd.Series([np.nan] * len(stock2_close), index=stock2_close.index)

price_df = pd.DataFrame(price_df_data)

# Align indices if we have both stocks
if stock1 in price_df.columns and stock2 in price_df.columns:
    price_df = price_df.sort_index()
    price_df = price_df.loc[price_df.index.intersection(price_df.index)]

print("\nPrice DataFrame shape:", price_df.shape)
print("\nFirst few rows of price DataFrame:")
print(price_df.head())

# VALIDATION: Check initial price data quality
print("\n" + "="*70)
print("VALIDATING PRICE DATA QUALITY")
print("="*70)
try:
    is_valid, issues = validate_price_data(
        price_df,
        symbols=[stock1, stock2],
        check_gaps=True,
        max_pct_change=0.20,
        check_volume=True
    )
    print_validation_report(issues)

    if not is_valid:
        print("\n⚠️  WARNING: Data quality issues detected!")
        print("Consider reviewing the data before proceeding.")
        # Don't raise error yet, just warn
except Exception as e:
    print(f"Warning: Data validation failed: {e}")
    print("Proceeding with caution...")

# Step 1: Calculate the daily price difference (stock1 - stock2)
if stock1 in price_df.columns and stock2 in price_df.columns:
    price_df['Price_Difference'] = price_df[stock1] - price_df[stock2]
else:
    raise ValueError(f"Need both {stock1} and {stock2} data to calculate price difference. Download failed for one or both stocks.")

# Step 2: Calculate moving averages
# 5-day moving average
if stock1 in price_df.columns:
    price_df[f'{stock1}_MA5'] = price_df[stock1].rolling(window=5).mean()
if stock2 in price_df.columns:
    price_df[f'{stock2}_MA5'] = price_df[stock2].rolling(window=5).mean()

# 20-day moving average
if stock1 in price_df.columns:
    price_df[f'{stock1}_MA20'] = price_df[stock1].rolling(window=20).mean()
if stock2 in price_df.columns:
    price_df[f'{stock2}_MA20'] = price_df[stock2].rolling(window=20).mean()

# Volume moving averages
if f'{stock1}_Volume' in price_df.columns:
    price_df[f'{stock1}_Volume_MA5'] = price_df[f'{stock1}_Volume'].rolling(window=5).mean()
if f'{stock2}_Volume' in price_df.columns:
    price_df[f'{stock2}_Volume_MA5'] = price_df[f'{stock2}_Volume'].rolling(window=5).mean()

print("\nAfter adding moving averages:")
print(price_df.head())

# Step 3: Add Technical Indicators (RSI, MACD)
print("\nCalculating technical indicators (RSI, MACD)...")

def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index (RSI)."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, fast_period=12, slow_period=26, signal_period=9):
    """Calculate MACD."""
    ema_fast = prices.ewm(span=fast_period, adjust=False).mean()
    ema_slow = prices.ewm(span=slow_period, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

# Add RSI for each stock
if stock1 in price_df.columns:
    price_df[f'{stock1}_RSI'] = calculate_rsi(price_df[stock1])
if stock2 in price_df.columns:
    price_df[f'{stock2}_RSI'] = calculate_rsi(price_df[stock2])

# Add MACD for each stock
if stock1 in price_df.columns:
    macd, signal, hist = calculate_macd(price_df[stock1])
    price_df[f'{stock1}_MACD'] = macd
    price_df[f'{stock1}_MACD_Signal'] = signal
    price_df[f'{stock1}_MACD_Histogram'] = hist

if stock2 in price_df.columns:
    macd, signal, hist = calculate_macd(price_df[stock2])
    price_df[f'{stock2}_MACD'] = macd
    price_df[f'{stock2}_MACD_Signal'] = signal
    price_df[f'{stock2}_MACD_Histogram'] = hist

print("Technical indicators calculated.")

# VALIDATION: Check technical indicators
print("\n" + "="*70)
print("VALIDATING TECHNICAL INDICATORS")
print("="*70)
try:
    is_valid_ti, issues_ti = validate_technical_indicators(price_df, symbols=[stock1, stock2])
    print_validation_report(issues_ti)

    if not is_valid_ti:
        print("\n⚠️  WARNING: Technical indicator validation issues detected!")
except Exception as e:
    print(f"Warning: Technical indicator validation failed: {e}")

# Step 4: Add Reddit Sentiment (Historical)
print("\nFetching Reddit sentiment (historical)...")
if REDDIT_AVAILABLE:
    try:
        from sentiment_analysis import get_historical_sentiment, get_sentiment_for_date, update_sentiment_cache
        
        # Get date range from DataFrame index
        date_range_start = price_df.index.min()
        date_range_end = price_df.index.max()
        
        print(f"Date range: {date_range_start.date()} to {date_range_end.date()}")
        
        # For each symbol, get historical sentiment
        for symbol in symbols:
            print(f"Loading sentiment for {symbol}...")
            
            # Try to get historical sentiment from cache
            historical_sentiment = get_historical_sentiment(
                symbol, 
                date_range_start, 
                date_range_end
            )
            
            if not historical_sentiment.empty:
                # Align with price_df dates
                sentiment_series = pd.Series(index=price_df.index, dtype=float)
                
                for date in price_df.index:
                    date_normalized = pd.Timestamp(date).normalize()
                    
                    # Try exact match first
                    if date_normalized in historical_sentiment.index:
                        sentiment_series[date] = historical_sentiment[date_normalized]
                    else:
                        # Use closest available date (forward fill from past)
                        past_sentiment = historical_sentiment[historical_sentiment.index <= date_normalized]
                        if not past_sentiment.empty:
                            sentiment_series[date] = past_sentiment.iloc[-1]
                        else:
                            # Try to fetch for recent dates
                            sentiment_series[date] = get_sentiment_for_date(
                                symbol, date, fetch_if_missing=True, days_tolerance=7
                            )
                
                price_df[f'{symbol}_Reddit_Sentiment'] = sentiment_series
                print(f"  {symbol}: Loaded {len(historical_sentiment)} cached sentiment values")
            else:
                # No cache available - fetch current sentiment and use for all dates
                print(f"  {symbol}: No cache found, fetching current sentiment...")
                try:
                    from sentiment_analysis import fetch_reddit_sentiment
                    sentiment_data = fetch_reddit_sentiment(symbol, limit=100)
                    
                    if sentiment_data['count'] > 0:
                        mean_sentiment = sentiment_data['mean_sentiment']
                        price_df[f'{symbol}_Reddit_Sentiment'] = mean_sentiment
                        
                        # Cache the sentiment for the latest date
                        update_sentiment_cache(symbol, date_range_end, sentiment_data)
                        print(f"  {symbol}: Current sentiment = {mean_sentiment:.4f} (cached for {date_range_end.date()})")
                    else:
                        price_df[f'{symbol}_Reddit_Sentiment'] = 0.0
                        print(f"  {symbol}: No sentiment data found, using 0.0")
                except Exception as e:
                    print(f"  Warning: Could not fetch sentiment for {symbol}: {e}")
                    price_df[f'{symbol}_Reddit_Sentiment'] = 0.0
    except Exception as e:
        print(f"Warning: Could not process Reddit sentiment: {e}")
        print("Continuing without sentiment data...")
        for symbol in symbols:
            price_df[f'{symbol}_Reddit_Sentiment'] = 0.0
else:
    print("Reddit sentiment module not available. Skipping...")
    for symbol in symbols:
        price_df[f'{symbol}_Reddit_Sentiment'] = 0.0

# Step 5: Add Options Volume (Historical)
print("\nFetching options volume data (historical)...")
try:
    from options_data import get_historical_options_volume, get_options_volume_for_date, update_options_cache
    
    # Get date range from DataFrame index
    date_range_start = price_df.index.min()
    date_range_end = price_df.index.max()
    
    print(f"Date range: {date_range_start.date()} to {date_range_end.date()}")
    
    # For each symbol, get historical options volume
    for symbol in symbols:
        print(f"Loading options volume for {symbol}...")
        
        # Try to get historical options volume from cache
        historical_options = get_historical_options_volume(
            symbol, 
            date_range_start, 
            date_range_end
        )
        
        if not historical_options.empty:
            # Align with price_df dates
            options_series = pd.Series(index=price_df.index, dtype=float)
            
            for date in price_df.index:
                date_normalized = pd.Timestamp(date).normalize()
                
                # Try exact match first
                if date_normalized in historical_options.index:
                    options_series[date] = historical_options[date_normalized]
                else:
                    # Use closest available date (forward fill from past)
                    past_options = historical_options[historical_options.index <= date_normalized]
                    if not past_options.empty:
                        options_series[date] = past_options.iloc[-1]
                    else:
                        # Try to fetch for recent dates
                        options_series[date] = get_options_volume_for_date(
                            symbol, date, fetch_if_missing=True, days_tolerance=7
                        )
            
            price_df[f'{symbol}_Options_Volume'] = options_series
            print(f"  {symbol}: Loaded {len(historical_options)} cached options volume values")
        else:
            # No cache available - fetch current options volume and use for all dates
            print(f"  {symbol}: No cache found, fetching current options volume...")
            try:
                from options_data import fetch_current_options_volume
                options_data = fetch_current_options_volume(symbol)
                
                if options_data['total_volume'] > 0:
                    total_volume = options_data['total_volume']
                    price_df[f'{symbol}_Options_Volume'] = total_volume
                    
                    # Cache the options volume for the latest date
                    update_options_cache(symbol, date_range_end, options_data)
                    print(f"  {symbol}: Current options volume = {total_volume:,} "
                          f"(Calls: {options_data['calls_volume']:,}, Puts: {options_data['puts_volume']:,})")
                    print(f"  Cached for {date_range_end.date()}")
                else:
                    price_df[f'{symbol}_Options_Volume'] = 0
                    print(f"  {symbol}: No options data found, using 0")
            except Exception as e:
                print(f"  Warning: Could not fetch options volume for {symbol}: {e}")
                price_df[f'{symbol}_Options_Volume'] = 0
except Exception as e:
    print(f"Warning: Could not process options volume: {e}")
    print("Continuing without options volume data...")
    for symbol in symbols:
        price_df[f'{symbol}_Options_Volume'] = 0

# Fill NaN values for options volume and sentiment (they're same for all dates)
for symbol in symbols:
    for col in [f'{symbol}_Reddit_Sentiment', f'{symbol}_Options_Volume']:
        if col in price_df.columns:
            price_df[col] = price_df[col].ffill().fillna(0)  # FIXED: Updated for pandas 3.0 compatibility

print("\nAfter adding sentiment and options data:")
print(price_df.head())

# Step 6: Remove any rows with missing values (but keep rows with sentiment/options)
initial_rows = len(price_df)

# Only drop rows where core price data is missing
core_columns = [stock1, stock2, 'Price_Difference']
price_df_clean = price_df.dropna(subset=core_columns)

# Check if we have any data left
if len(price_df_clean) == 0:
    print("\n" + "="*70)
    print("ERROR: No valid data remaining after removing missing values!")
    print("="*70)
    print(f"\nTotal rows: {initial_rows}")
    print(f"{stock1} data: {price_df[stock1].notna().sum()} valid rows" if stock1 in price_df.columns else f"{stock1}: Not available")
    print(f"{stock2} data: {price_df[stock2].notna().sum()} valid rows" if stock2 in price_df.columns else f"{stock2}: Not available")
    raise ValueError("No valid data available. Please check your internet connection and try again. The yfinance library may be experiencing issues.")

rows_removed = initial_rows - len(price_df_clean)

print(f"\nRemoved {rows_removed} rows with missing core values")
print(f"Clean data shape: {price_df_clean.shape}")
print("\nFirst few rows after removing missing values:")
print(price_df_clean.head())

# Step 7: Normalize the data so it's ready for a machine learning model
# We'll use StandardScaler for normalization (z-score normalization)
from sklearn.preprocessing import StandardScaler

# Select columns to normalize - build dynamically based on available stocks
columns_to_normalize = [
    stock1, stock2, 'Price_Difference',
    f'{stock1}_Volume', f'{stock2}_Volume',
    f'{stock1}_MA5', f'{stock2}_MA5', f'{stock1}_MA20', f'{stock2}_MA20',
    f'{stock1}_Volume_MA5', f'{stock2}_Volume_MA5',
    f'{stock1}_RSI', f'{stock2}_RSI',
    f'{stock1}_MACD', f'{stock2}_MACD', f'{stock1}_MACD_Signal', f'{stock2}_MACD_Signal',
    f'{stock1}_MACD_Histogram', f'{stock2}_MACD_Histogram',
    f'{stock1}_Reddit_Sentiment', f'{stock2}_Reddit_Sentiment',
    f'{stock1}_Options_Volume', f'{stock2}_Options_Volume'
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
print(f"New features added:")
print(f"  • RSI indicators for both stocks")
print(f"  • MACD indicators for both stocks")
print(f"  • Reddit sentiment scores")
print(f"  • Options volume data")
print(f"\nStocks processed: {stock1} and {stock2}")
