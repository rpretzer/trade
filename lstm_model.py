import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import pandas as pd
import numpy as np

def create_lstm_model(input_shape, dropout_rate=0.2):
    """
    Create an improved LSTM model for predicting the next day's price difference.
    
    Architecture:
    - First LSTM layer: 50 units with tanh activation
    - Dropout layer to prevent overfitting
    - Second LSTM layer: 30 units with tanh activation
    - Dropout layer to prevent overfitting
    - Dense output layer: 1 unit
    
    Args:
        input_shape: Tuple of (timesteps, features) for input data
        dropout_rate: Dropout rate (default 0.2)
        
    Returns:
        Compiled Keras model
    """
    model = Sequential([
        # First LSTM layer with 50 units and tanh activation
        LSTM(50, activation='tanh', return_sequences=True, input_shape=input_shape),
        
        # Dropout layer to prevent overfitting
        Dropout(dropout_rate),
        
        # Second LSTM layer with 30 units and tanh activation
        LSTM(30, activation='tanh'),
        
        # Dropout layer to prevent overfitting
        Dropout(dropout_rate),
        
        # Dense output layer with 1 unit
        Dense(1)
    ])
    
    # Compile the model with Adam optimizer and MSE loss
    model.compile(
        optimizer='adam',
        loss='mean_squared_error',
        metrics=['mae']  # Mean Absolute Error for additional metrics
    )
    
    return model

# Example usage (commented out since TensorFlow may not be available)
if __name__ == "__main__":
    try:
        # Load processed data (assuming you've run process_stock_data.py first)
        # df = pd.read_csv('processed_stock_data.csv', index_col=0, parse_dates=True)
        
        # For demonstration, create model with example input shape
        # Input shape: (timesteps, features)
        # timesteps: number of previous days to look at (e.g., 60 days)
        # features: number of input features (now includes volume and moving averages)
        timesteps = 60  # Look back 60 days
        n_features = 11  # Updated: AAPL, MSFT, Price_Difference, Volumes (2), MAs (5)
        
        input_shape = (timesteps, n_features)
        
        # Create the model
        model = create_lstm_model(input_shape, dropout_rate=0.2)
        
        # Display model summary
        print("Improved LSTM Model Architecture:")
        print("=" * 50)
        model.summary()
        
        print("\nModel compiled successfully!")
        print(f"Input shape: {input_shape}")
        print("Optimizer: Adam")
        print("Loss function: Mean Squared Error (MSE)")
        print("Features: Includes price, volume, and moving averages")
        print("Regularization: Dropout layers with rate 0.2")
        
    except ImportError as e:
        print(f"TensorFlow/Keras not available: {e}")
        print("\nTo use this model:")
        print("1. Install TensorFlow: pip install tensorflow")
        print("2. Note: TensorFlow requires Python 3.11 or 3.12 (not 3.14)")
        print("3. Consider using a virtual environment with Python 3.11/3.12")
    except Exception as e:
        print(f"Error: {e}")
