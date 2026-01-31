"""
Sentiment Analysis Module for Stock Data
Fetches Reddit sentiment and analyzes it for stock symbols
"""

import praw
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import re
import os
import json
import logging

logger = logging.getLogger(__name__)

def analyze_sentiment_simple(text):
    """
    Simple sentiment analyzer based on keywords.
    Returns a score between -1 (very negative) and 1 (very positive).
    
    Args:
        text: Text to analyze
        
    Returns:
        Sentiment score between -1 and 1
    """
    if not text or not isinstance(text, str):
        return 0.0
    
    text_lower = text.lower()
    
    # Positive keywords
    positive_words = ['bull', 'moon', 'rocket', 'buy', 'pump', 'gains', 'profit', 
                     'win', 'hold', 'diamond', 'hands', 'squeeze', 'rise', 'up',
                     'good', 'great', 'amazing', 'best', 'love', 'like', 'strong']
    
    # Negative keywords
    negative_words = ['bear', 'crash', 'drop', 'sell', 'dump', 'loss', 'lose',
                     'fall', 'down', 'bad', 'worst', 'hate', 'weak', 'dead',
                     'scam', 'fraud', 'terrible', 'awful']
    
    # Count positive and negative words
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)
    
    # Calculate score
    total_words = len(text_lower.split())
    if total_words == 0:
        return 0.0
    
    # Normalize score
    score = (positive_count - negative_count) / max(total_words, 1)
    
    # Clip to [-1, 1]
    return max(-1.0, min(1.0, score * 10))

def fetch_reddit_sentiment(symbol, subreddit_name='wallstreetbets', limit=100, 
                          reddit_client_id=None, reddit_client_secret=None, 
                          user_agent='stock-sentiment:1.0'):
    """
    Fetch Reddit posts about a stock symbol and calculate sentiment scores.
    
    Args:
        symbol: Stock symbol (e.g., 'AAPL', 'MSFT')
        subreddit_name: Subreddit to search (default: 'wallstreetbets')
        limit: Number of posts to fetch (default: 100)
        reddit_client_id: Reddit API client ID (optional)
        reddit_client_secret: Reddit API client secret (optional)
        user_agent: User agent string for Reddit API
        
    Returns:
        Dictionary with sentiment statistics
    """
    try:
        # SECURITY: Require valid Reddit API credentials
        if not reddit_client_id or not reddit_client_secret:
            logger.error(
                "Reddit API credentials are required for sentiment analysis. "
                "Dummy credentials have been removed for security. "
                "Obtain credentials from https://www.reddit.com/prefs/apps"
            )
            return {
                'mean_sentiment': 0.0,
                'median_sentiment': 0.0,
                'positive_ratio': 0.0,
                'post_count': 0,
                'error': 'No API credentials provided'
            }

        # Validate credentials are not dummy values
        if reddit_client_id == 'dummy' or reddit_client_secret == 'dummy':
            logger.error("Dummy credentials detected and rejected")
            return {
                'mean_sentiment': 0.0,
                'median_sentiment': 0.0,
                'positive_ratio': 0.0,
                'post_count': 0,
                'error': 'Invalid API credentials'
            }

        # Initialize Reddit client with valid credentials
        reddit = praw.Reddit(
            client_id=reddit_client_id,
            client_secret=reddit_client_secret,
            user_agent=user_agent
        )
        
        # Search for posts mentioning the symbol
        subreddit = reddit.subreddit(subreddit_name)
        
        # Search in title and selftext
        search_query = f"${symbol} OR {symbol}"
        
        sentiment_scores = []
        post_count = 0
        
        # Get new posts
        try:
            for post in subreddit.new(limit=limit):
                if post_count >= limit:
                    break
                
                # Check if post mentions the symbol
                post_text = f"{post.title} {post.selftext}"
                if symbol.upper() in post_text.upper() or f"${symbol.upper()}" in post_text.upper():
                    score = analyze_sentiment_simple(post_text)
                    sentiment_scores.append(score)
                    post_count += 1
        except Exception as e:
            print(f"Warning: Error fetching Reddit posts: {e}")
        
        # Also try searching directly
        try:
            for post in subreddit.search(symbol, limit=limit, sort='new'):
                if post_count >= limit:
                    break
                post_text = f"{post.title} {post.selftext}"
                score = analyze_sentiment_simple(post_text)
                if score not in sentiment_scores[-10:]:  # Avoid duplicates
                    sentiment_scores.append(score)
                    post_count += 1
        except Exception:
            pass  # Search might not be available
        
        if sentiment_scores:
            return {
                'mean_sentiment': np.mean(sentiment_scores),
                'median_sentiment': np.median(sentiment_scores),
                'std_sentiment': np.std(sentiment_scores),
                'count': len(sentiment_scores)
            }
        else:
            return {
                'mean_sentiment': 0.0,
                'median_sentiment': 0.0,
                'std_sentiment': 0.0,
                'count': 0
            }
            
    except Exception as e:
        print(f"Error fetching Reddit sentiment for {symbol}: {e}")
        return {
            'mean_sentiment': 0.0,
            'median_sentiment': 0.0,
            'std_sentiment': 0.0,
            'count': 0
        }

def fetch_reddit_sentiment_for_symbols(symbols, subreddit_name='wallstreetbets', limit=100):
    """
    Fetch Reddit sentiment for multiple symbols.
    
    Args:
        symbols: List of stock symbols
        subreddit_name: Subreddit to search
        limit: Number of posts per symbol
        
    Returns:
        Dictionary mapping symbols to sentiment data
    """
    results = {}
    
    for symbol in symbols:
        print(f"Fetching Reddit sentiment for {symbol}...")
        results[symbol] = fetch_reddit_sentiment(symbol, subreddit_name, limit)
        time.sleep(1)  # Rate limiting
    
    return results

# Historical sentiment cache management
SENTIMENT_CACHE_FILE = 'reddit_sentiment_cache.csv'

def load_sentiment_cache(cache_file=SENTIMENT_CACHE_FILE):
    """
    Load historical sentiment cache from CSV file.
    
    Args:
        cache_file: Path to cache file
        
    Returns:
        DataFrame with columns: Date, Symbol, Mean_Sentiment, Count, Last_Updated
    """
    if os.path.exists(cache_file):
        try:
            df = pd.read_csv(cache_file, parse_dates=['Date'], index_col=['Date', 'Symbol'])
            return df
        except Exception as e:
            print(f"Warning: Could not load sentiment cache: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

def save_sentiment_cache(cache_df, cache_file=SENTIMENT_CACHE_FILE):
    """
    Save sentiment cache to CSV file.
    
    Args:
        cache_df: DataFrame with sentiment data
        cache_file: Path to cache file
    """
    try:
        cache_df.to_csv(cache_file)
    except Exception as e:
        print(f"Warning: Could not save sentiment cache: {e}")

def update_sentiment_cache(symbol, date, sentiment_data, cache_file=SENTIMENT_CACHE_FILE):
    """
    Update sentiment cache with new data for a specific symbol and date.
    
    Args:
        symbol: Stock symbol
        date: Date (datetime or date object)
        sentiment_data: Dictionary with sentiment metrics
        cache_file: Path to cache file
    """
    cache_df = load_sentiment_cache(cache_file)
    
    # Convert date to datetime if needed
    if isinstance(date, str):
        date = pd.to_datetime(date)
    elif not isinstance(date, pd.Timestamp):
        date = pd.Timestamp(date)
    
    # Normalize date to just the date part (remove time)
    date = date.normalize()
    
    # Create new row
    new_row = pd.DataFrame({
        'Mean_Sentiment': [sentiment_data.get('mean_sentiment', 0.0)],
        'Median_Sentiment': [sentiment_data.get('median_sentiment', 0.0)],
        'Std_Sentiment': [sentiment_data.get('std_sentiment', 0.0)],
        'Count': [sentiment_data.get('count', 0)],
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
    
    save_sentiment_cache(cache_df, cache_file)
    return cache_df

def get_sentiment_for_date(symbol, target_date, cache_file=SENTIMENT_CACHE_FILE, 
                          fetch_if_missing=False, days_tolerance=7):
    """
    Get sentiment score for a specific date from cache or fetch if needed.
    
    Args:
        symbol: Stock symbol
        target_date: Target date (datetime, date, or string)
        cache_file: Path to sentiment cache file
        fetch_if_missing: If True, fetch fresh data if not in cache (for recent dates)
        days_tolerance: Number of days to look back if exact date not found
        
    Returns:
        Sentiment score (float)
    """
    cache_df = load_sentiment_cache(cache_file)
    
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
            return float(cache_df.loc[(target_date, symbol), 'Mean_Sentiment'])
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
                return float(symbol_data.loc[closest_idx, 'Mean_Sentiment'])
    
    # If fetch_if_missing and date is recent (within last 7 days), try fetching
    if fetch_if_missing:
        days_ago = (datetime.now().date() - target_date.date()).days
        if 0 <= days_ago <= 7:
            try:
                print(f"Fetching fresh sentiment for {symbol} on {target_date.date()}...")
                sentiment_data = fetch_reddit_sentiment(symbol, limit=50)
                if sentiment_data['count'] > 0:
                    update_sentiment_cache(symbol, target_date, sentiment_data, cache_file)
                    return sentiment_data['mean_sentiment']
            except Exception as e:
                print(f"Warning: Could not fetch fresh sentiment: {e}")
    
    # Default: return 0 (neutral) if no data available
    return 0.0

def get_historical_sentiment(symbol, start_date, end_date, cache_file=SENTIMENT_CACHE_FILE):
    """
    Get historical sentiment scores for a date range.
    
    Args:
        symbol: Stock symbol
        start_date: Start date (datetime, date, or string)
        end_date: End date (datetime, date, or string)
        cache_file: Path to sentiment cache file
        
    Returns:
        Series with dates as index and sentiment scores as values
    """
    cache_df = load_sentiment_cache(cache_file)
    
    # Convert dates
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)
    
    start_date = pd.Timestamp(start_date).normalize()
    end_date = pd.Timestamp(end_date).normalize()
    
    # Filter cache for symbol and date range
    if cache_df.empty:
        return pd.Series(dtype=float)
    
    try:
        symbol_data = cache_df.xs(symbol, level='Symbol', drop_level=False)
        if symbol_data.empty:
            return pd.Series(dtype=float)
        
        # Filter by date range
        mask = (symbol_data.index.get_level_values('Date') >= start_date) & \
               (symbol_data.index.get_level_values('Date') <= end_date)
        filtered = symbol_data[mask]
        
        if not filtered.empty:
            # Return as Series with Date index
            return filtered['Mean_Sentiment']
    except (KeyError, IndexError):
        pass
    
    return pd.Series(dtype=float)

def build_historical_sentiment_cache(symbols, start_date, end_date, 
                                     cache_file=SENTIMENT_CACHE_FILE,
                                     subreddit_name='wallstreetbets',
                                     limit=50, delay_between_symbols=2):
    """
    Build historical sentiment cache by fetching data for a date range.
    Note: Reddit doesn't provide historical API, so this fetches current sentiment
    and assigns it to dates. For true historical data, you'd need to run this daily.
    
    Args:
        symbols: List of stock symbols
        start_date: Start date
        end_date: End date
        cache_file: Path to cache file
        subreddit_name: Subreddit to search
        limit: Number of posts to fetch per symbol
        delay_between_symbols: Seconds to wait between symbols
    """
    print(f"Building sentiment cache for {len(symbols)} symbols from {start_date} to {end_date}...")
    
    cache_df = load_sentiment_cache(cache_file)
    
    for symbol in symbols:
        print(f"\nProcessing {symbol}...")
        try:
            # Fetch current sentiment
            sentiment_data = fetch_reddit_sentiment(symbol, subreddit_name, limit)
            
            if sentiment_data['count'] > 0:
                # For each date in range, use the current sentiment
                # (In production, you'd want to fetch daily and store)
                current_date = pd.Timestamp(end_date).normalize()
                update_sentiment_cache(symbol, current_date, sentiment_data, cache_file)
                print(f"  {symbol}: Mean sentiment = {sentiment_data['mean_sentiment']:.4f} (from {sentiment_data['count']} posts)")
            else:
                print(f"  {symbol}: No sentiment data found")
            
            time.sleep(delay_between_symbols)  # Rate limiting
        except Exception as e:
            print(f"  Error processing {symbol}: {e}")
    
    print(f"\nSentiment cache updated: {cache_file}")
