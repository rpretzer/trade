"""
Options Data Module for Stock Data
Fetches and caches historical options volume data
"""

import logging
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os

logger = logging.getLogger(__name__)

# Options cache file
OPTIONS_CACHE_FILE = 'options_volume_cache.csv'

def load_options_cache(cache_file=OPTIONS_CACHE_FILE):
    """
    Load historical options volume cache from CSV file.
    
    Args:
        cache_file: Path to cache file
        
    Returns:
        DataFrame with columns: Date, Symbol, Options_Volume, Calls_Volume, Puts_Volume, Last_Updated
    """
    if os.path.exists(cache_file):
        try:
            df = pd.read_csv(cache_file, parse_dates=['Date'], index_col=['Date', 'Symbol'])
            return df
        except Exception as e:
            logger.warning("Could not load options cache: %s", e)
            return pd.DataFrame()
    return pd.DataFrame()

def save_options_cache(cache_df, cache_file=OPTIONS_CACHE_FILE):
    """
    Save options cache to CSV file.
    
    Args:
        cache_df: DataFrame with options data
        cache_file: Path to cache file
    """
    try:
        cache_df.to_csv(cache_file)
    except Exception as e:
        logger.warning("Could not save options cache: %s", e)

def fetch_current_options_volume(symbol):
    """
    Fetch current options volume for a symbol using yfinance.
    
    Args:
        symbol: Stock symbol
        
    Returns:
        Dictionary with options volume metrics
    """
    try:
        ticker = yf.Ticker(symbol)
        
        # Get available options expiration dates
        options_dates = ticker.options
        
        if not options_dates:
            return {
                'total_volume': 0,
                'calls_volume': 0,
                'puts_volume': 0,
                'count': 0
            }
        
        # Get the most recent options chain
        try:
            opt = ticker.option_chain(options_dates[0])
        except Exception as e:
            logger.warning("Could not fetch options chain for %s: %s", symbol, e)
            return {
                'total_volume': 0,
                'calls_volume': 0,
                'puts_volume': 0,
                'count': 0
            }
        
        # Calculate volumes
        calls_volume = 0
        puts_volume = 0
        
        if not opt.calls.empty:
            calls_volume = opt.calls['volume'].fillna(0).sum()
        
        if not opt.puts.empty:
            puts_volume = opt.puts['volume'].fillna(0).sum()
        
        total_volume = calls_volume + puts_volume
        
        # Count of contracts
        count = len(opt.calls) + len(opt.puts)
        
        return {
            'total_volume': int(total_volume),
            'calls_volume': int(calls_volume),
            'puts_volume': int(puts_volume),
            'count': count
        }
        
    except Exception as e:
        logger.error("Error fetching options volume for %s: %s", symbol, e)
        return {
            'total_volume': 0,
            'calls_volume': 0,
            'puts_volume': 0,
            'count': 0
        }

def update_options_cache(symbol, date, options_data, cache_file=OPTIONS_CACHE_FILE):
    """
    Update options cache with new data for a specific symbol and date.
    
    Args:
        symbol: Stock symbol
        date: Date (datetime or date object)
        options_data: Dictionary with options volume metrics
        cache_file: Path to cache file
    """
    cache_df = load_options_cache(cache_file)
    
    # Convert date to datetime if needed
    if isinstance(date, str):
        date = pd.to_datetime(date)
    elif not isinstance(date, pd.Timestamp):
        date = pd.Timestamp(date)
    
    # Normalize date to just the date part (remove time)
    date = date.normalize()
    
    # Create new row
    new_row = pd.DataFrame({
        'Options_Volume': [options_data.get('total_volume', 0)],
        'Calls_Volume': [options_data.get('calls_volume', 0)],
        'Puts_Volume': [options_data.get('puts_volume', 0)],
        'Contract_Count': [options_data.get('count', 0)],
        'Last_Updated': [datetime.now().isoformat()]
    }, index=pd.MultiIndex.from_tuples([(date, symbol)], names=['Date', 'Symbol']))
    
    # Append or update
    if cache_df.empty:
        cache_df = new_row
    else:
        # Remove existing entry for this date/symbol if it exists
        cache_df = cache_df.drop((date, symbol), errors='ignore')
        cache_df = pd.concat([cache_df, new_row])
        cache_df = cache_df.sort_index()
    
    save_options_cache(cache_df, cache_file)
    return cache_df

def get_options_volume_for_date(symbol, target_date, cache_file=OPTIONS_CACHE_FILE, 
                               fetch_if_missing=False, days_tolerance=7):
    """
    Get options volume for a specific date from cache or fetch if needed.
    
    Args:
        symbol: Stock symbol
        target_date: Target date (datetime, date, or string)
        cache_file: Path to options cache file
        fetch_if_missing: If True, fetch fresh data if not in cache (for recent dates)
        days_tolerance: Number of days to look back if exact date not found
        
    Returns:
        Options volume (int)
    """
    cache_df = load_options_cache(cache_file)
    
    # Convert target_date to datetime
    if isinstance(target_date, str):
        target_date = pd.to_datetime(target_date)
    elif not isinstance(target_date, pd.Timestamp):
        target_date = pd.Timestamp(target_date)
    
    # Normalize to date only
    target_date = target_date.normalize()
    
    # Try to get exact date
    try:
        if not cache_df.empty and (target_date, symbol) in cache_df.index:
            return int(cache_df.loc[(target_date, symbol), 'Options_Volume'])
    except (KeyError, IndexError):
        pass
    
    # If exact date not found, look for nearby dates (within tolerance)
    if not cache_df.empty:
        symbol_data = cache_df.xs(symbol, level='Symbol', drop_level=False) if symbol in cache_df.index.get_level_values('Symbol') else pd.DataFrame()
        
        if not symbol_data.empty:
            # Find closest date
            date_diff = (symbol_data.index.get_level_values('Date') - target_date).abs()
            closest_idx = date_diff.idxmin()
            closest_date = symbol_data.index.get_level_values('Date')[date_diff.idxmin()]
            days_diff = abs((closest_date - target_date).days)
            
            if days_diff <= days_tolerance:
                return int(symbol_data.loc[closest_idx, 'Options_Volume'])
    
    # If fetch_if_missing and date is recent (within last 7 days), try fetching
    if fetch_if_missing:
        days_ago = (datetime.now().date() - target_date.date()).days
        if 0 <= days_ago <= 7:
            try:
                logger.info("Fetching fresh options volume for %s on %s", symbol, target_date.date())
                options_data = fetch_current_options_volume(symbol)
                if options_data['total_volume'] > 0:
                    update_options_cache(symbol, target_date, options_data, cache_file)
                    return options_data['total_volume']
            except Exception as e:
                logger.warning("Could not fetch fresh options volume: %s", e)
    
    # Default: return 0 if no data available
    return 0

def get_historical_options_volume(symbol, start_date, end_date, cache_file=OPTIONS_CACHE_FILE):
    """
    Get historical options volume for a date range.
    
    Args:
        symbol: Stock symbol
        start_date: Start date (datetime, date, or string)
        end_date: End date (datetime, date, or string)
        cache_file: Path to options cache file
        
    Returns:
        Series with dates as index and options volume as values
    """
    cache_df = load_options_cache(cache_file)
    
    # Convert dates
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)
    
    start_date = pd.Timestamp(start_date).normalize()
    end_date = pd.Timestamp(end_date).normalize()
    
    # Filter cache for symbol and date range
    if cache_df.empty:
        return pd.Series(dtype=int)
    
    try:
        symbol_data = cache_df.xs(symbol, level='Symbol', drop_level=False)
        if symbol_data.empty:
            return pd.Series(dtype=int)
        
        # Filter by date range
        mask = (symbol_data.index.get_level_values('Date') >= start_date) & \
               (symbol_data.index.get_level_values('Date') <= end_date)
        filtered = symbol_data[mask]
        
        if not filtered.empty:
            # Return as Series with Date index
            return filtered['Options_Volume'].astype(int)
    except (KeyError, IndexError):
        pass
    
    return pd.Series(dtype=int)

def build_historical_options_cache(symbols, start_date, end_date, 
                                   cache_file=OPTIONS_CACHE_FILE,
                                   delay_between_symbols=2):
    """
    Build historical options cache by fetching data for a date range.
    Note: yfinance only provides current options data, so this fetches current
    data and assigns it to dates. For true historical data, run this daily.
    
    Args:
        symbols: List of stock symbols
        start_date: Start date
        end_date: End date
        cache_file: Path to cache file
        delay_between_symbols: Seconds to wait between symbols
    """
    logger.info("Building options volume cache for %d symbols from %s to %s", len(symbols), start_date, end_date)
    
    cache_df = load_options_cache(cache_file)
    
    for symbol in symbols:
        logger.info("Processing %s", symbol)
        try:
            # Fetch current options volume
            options_data = fetch_current_options_volume(symbol)
            
            if options_data['total_volume'] > 0:
                # For each date in range, use the current options volume
                # (In production, you'd want to fetch daily and store)
                current_date = pd.Timestamp(end_date).normalize()
                update_options_cache(symbol, current_date, options_data, cache_file)
                logger.info("%s: Options volume = %s (Calls: %s, Puts: %s)",
                            symbol, f"{options_data['total_volume']:,}",
                            f"{options_data['calls_volume']:,}", f"{options_data['puts_volume']:,}")
            else:
                logger.info("%s: No options data found", symbol)
            
            time.sleep(delay_between_symbols)  # Rate limiting
        except Exception as e:
            logger.error("Error processing %s: %s", symbol, e)
    
    logger.info("Options volume cache updated: %s", cache_file)
