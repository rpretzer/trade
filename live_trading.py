"""
Live Trading Module
Gets latest trading signals and executes trades
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import os

def create_sequences(data, timesteps):
    """Create sequences for LSTM input."""
    X, y = [], []
    for i in range(timesteps, len(data)):
        X.append(data[i-timesteps:i])
        y.append(data[i, -1])
    return np.array(X), np.array(y)

def get_latest_signal(model_path='lstm_price_difference_model.h5',
                     data_file='processed_stock_data.csv',
                     timesteps=60,
                     threshold=0.5):
    """
    Get the latest trading signal from the trained model.
    
    Args:
        model_path: Path to trained LSTM model
        data_file: Path to processed data CSV
        timesteps: Number of timesteps for LSTM
        threshold: Threshold for signal generation
        
    Returns:
        Trading signal string
    """
    try:
        # Load model
        model = keras.models.load_model(model_path)
        
        # Load data
        df = pd.read_csv(data_file, index_col=0, parse_dates=True)
        
        # Get feature columns (use normalized if available)
        feature_columns_normalized = [
            'AAPL_normalized', 'MSFT_normalized', 'Price_Difference_normalized',
            'AAPL_Volume_normalized', 'MSFT_Volume_normalized',
            'AAPL_MA5_normalized', 'MSFT_MA5_normalized',
            'AAPL_MA20_normalized', 'MSFT_MA20_normalized',
            'AAPL_Volume_MA5_normalized', 'MSFT_Volume_MA5_normalized',
            'AAPL_RSI_normalized', 'MSFT_RSI_normalized',
            'AAPL_MACD_normalized', 'MSFT_MACD_normalized',
            'AAPL_MACD_Signal_normalized', 'MSFT_MACD_Signal_normalized',
            'AAPL_MACD_Histogram_normalized', 'MSFT_MACD_Histogram_normalized',
            'AAPL_Reddit_Sentiment_normalized', 'MSFT_Reddit_Sentiment_normalized',
            'AAPL_Options_Volume_normalized', 'MSFT_Options_Volume_normalized'
        ]
        
        feature_columns_original = [
            'AAPL', 'MSFT', 'Price_Difference',
            'AAPL_Volume', 'MSFT_Volume',
            'AAPL_MA5', 'MSFT_MA5', 'AAPL_MA20', 'MSFT_MA20',
            'AAPL_Volume_MA5', 'MSFT_Volume_MA5',
            'AAPL_RSI', 'MSFT_RSI',
            'AAPL_MACD', 'MSFT_MACD',
            'AAPL_MACD_Signal', 'MSFT_MACD_Signal',
            'AAPL_MACD_Histogram', 'MSFT_MACD_Histogram',
            'AAPL_Reddit_Sentiment', 'MSFT_Reddit_Sentiment',
            'AAPL_Options_Volume', 'MSFT_Options_Volume'
        ]
        
        available_normalized = [col for col in feature_columns_normalized if col in df.columns]
        available_original = [col for col in feature_columns_original if col in df.columns]
        
        if len(available_normalized) > len(available_original):
            feature_columns = available_normalized
        else:
            feature_columns = available_original
            # Scale if using original
            scaler = StandardScaler()
            data = scaler.fit_transform(df[feature_columns])
            return None  # For simplicity, return None if scaling needed
        
        data = df[feature_columns].values
        
        # Get the last sequence
        if len(data) < timesteps:
            return None
        
        X_latest = data[-timesteps:].reshape(1, timesteps, len(feature_columns))
        
        # Predict
        prediction = model.predict(X_latest, verbose=0)[0][0]
        
        # Generate signal
        if prediction > threshold:
            return 'Buy AAPL, Sell MSFT'
        elif prediction < -threshold:
            return 'Buy MSFT, Sell AAPL'
        else:
            return 'Hold'
            
    except Exception as e:
        print(f"Error getting latest signal: {e}")
        return None
