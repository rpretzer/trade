"""
Technical Indicators Module
Calculates RSI, MACD, and other technical indicators
"""

import pandas as pd
import numpy as np

def calculate_rsi(prices, period=14):
    """
    Calculate Relative Strength Index (RSI).
    
    Args:
        prices: Series of closing prices
        period: RSI period (default: 14)
        
    Returns:
        Series of RSI values
    """
    delta = prices.diff()
    
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_macd(prices, fast_period=12, slow_period=26, signal_period=9):
    """
    Calculate MACD (Moving Average Convergence Divergence).
    
    Args:
        prices: Series of closing prices
        fast_period: Fast EMA period (default: 12)
        slow_period: Slow EMA period (default: 26)
        signal_period: Signal line EMA period (default: 9)
        
    Returns:
        Dictionary with 'macd', 'signal', and 'histogram' Series
    """
    ema_fast = prices.ewm(span=fast_period, adjust=False).mean()
    ema_slow = prices.ewm(span=slow_period, adjust=False).mean()
    
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line
    
    return {
        'macd': macd_line,
        'signal': signal_line,
        'histogram': histogram
    }

def add_technical_indicators(df, price_column, prefix=''):
    """
    Add RSI and MACD indicators to a DataFrame.
    
    Args:
        df: DataFrame with price data
        price_column: Name of the price column
        prefix: Prefix for indicator column names (e.g., 'AAPL_')
        
    Returns:
        DataFrame with added indicator columns
    """
    df = df.copy()
    
    # Calculate RSI
    rsi = calculate_rsi(df[price_column])
    df[f'{prefix}RSI'] = rsi
    
    # Calculate MACD
    macd_data = calculate_macd(df[price_column])
    df[f'{prefix}MACD'] = macd_data['macd']
    df[f'{prefix}MACD_Signal'] = macd_data['signal']
    df[f'{prefix}MACD_Histogram'] = macd_data['histogram']
    
    return df
