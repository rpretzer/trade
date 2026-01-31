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
import logging
from exceptions import (
    ModelLoadError, DataNotFoundError, InsufficientDataError,
    ModelPredictionError
)
from error_handling import retry, RetryConfig

logger = logging.getLogger(__name__)

def create_sequences(data, timesteps):
    """Create sequences for LSTM input."""
    X, y = [], []
    for i in range(timesteps, len(data)):
        X.append(data[i-timesteps:i])
        y.append(data[i, -1])
    return np.array(X), np.array(y)

@retry(
    exceptions=(ModelLoadError, DataNotFoundError),
    config=RetryConfig(max_attempts=3, initial_delay=1.0)
)
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

    Raises:
        ModelLoadError: If model cannot be loaded
        DataNotFoundError: If data file is not found
        InsufficientDataError: If not enough data for prediction
        ModelPredictionError: If prediction fails
    """
    # Load model
    try:
        model = keras.models.load_model(model_path)
    except (FileNotFoundError, OSError) as e:
        raise ModelLoadError(f"Failed to load model from {model_path}: {e}")

    # Load data
    try:
        df = pd.read_csv(data_file, index_col=0, parse_dates=True)
    except FileNotFoundError as e:
        raise DataNotFoundError(f"Data file not found: {data_file}")
    except Exception as e:
        raise DataNotFoundError(f"Failed to load data from {data_file}: {e}")

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

    # Use normalized features if available, otherwise use original with scaling
    if len(available_normalized) > len(available_original):
        feature_columns = available_normalized
        data = df[feature_columns].values
    else:
        feature_columns = available_original
        # Scale original features
        logger.info("Using original features with StandardScaler normalization")
        scaler = StandardScaler()
        try:
            data = scaler.fit_transform(df[feature_columns])
        except Exception as e:
            raise ModelPredictionError(f"Failed to scale features: {e}")

    # Check if we have enough data
    if len(data) < timesteps:
        raise InsufficientDataError(
            f"Not enough data for prediction. Need {timesteps} timesteps, "
            f"have {len(data)}"
        )

    # Get the last sequence
    X_latest = data[-timesteps:].reshape(1, timesteps, len(feature_columns))

    # Predict
    try:
        prediction = model.predict(X_latest, verbose=0)[0][0]
    except Exception as e:
        raise ModelPredictionError(f"Model prediction failed: {e}")

    # Generate signal
    if prediction > threshold:
        logger.info(f"Signal: Buy AAPL, Sell MSFT (prediction: {prediction:.4f})")
        return 'Buy AAPL, Sell MSFT'
    elif prediction < -threshold:
        logger.info(f"Signal: Buy MSFT, Sell AAPL (prediction: {prediction:.4f})")
        return 'Buy MSFT, Sell AAPL'
    else:
        logger.info(f"Signal: Hold (prediction: {prediction:.4f})")
        return 'Hold'
