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
        # Initialize Reddit client
        if reddit_client_id and reddit_client_secret:
            reddit = praw.Reddit(
                client_id=reddit_client_id,
                client_secret=reddit_client_secret,
                user_agent=user_agent
            )
        else:
            # Use read-only mode (rate-limited)
            reddit = praw.Reddit(
                client_id='dummy',
                client_secret='dummy',
                user_agent=user_agent
            )
            reddit.read_only = True
        
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

# For historical data, we'll use a simplified approach since Reddit doesn't provide historical API easily
def get_sentiment_for_date(symbol, target_date, sentiment_cache=None):
    """
    Get sentiment score for a specific date.
    For now, this uses a simplified approach - in production, you'd want to
    cache sentiment data daily or use a service.
    
    Args:
        symbol: Stock symbol
        target_date: Target date (datetime)
        sentiment_cache: Optional cache of sentiment data
        
    Returns:
        Sentiment score (float)
    """
    # This is a placeholder - in a real implementation, you'd want to:
    # 1. Fetch sentiment daily and store it
    # 2. Use a sentiment API service
    # 3. Cache historical sentiment data
    
    if sentiment_cache and symbol in sentiment_cache:
        return sentiment_cache[symbol].get('mean_sentiment', 0.0)
    
    # For now, return 0 (neutral) as we don't have historical sentiment
    return 0.0
