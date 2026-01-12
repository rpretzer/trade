import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Try importing XGBoost
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    print("Warning: XGBoost not available. Install with: pip install xgboost")
    XGBOOST_AVAILABLE = False

def create_sequences(data, timesteps):
    """
    Create sequences for LSTM input.
    
    Args:
        data: Array with features (shape: samples, features)
        timesteps: Number of previous timesteps to use
        
    Returns:
        X: Array of shape (samples, timesteps, features)
        y: Array of shape (samples,) - target values (next day's price difference)
    """
    X, y = [], []
    
    for i in range(timesteps, len(data)):
        # Input sequence: previous timesteps
        X.append(data[i-timesteps:i])
        # Target: next day's price difference
        # Find the index of Price_Difference in the feature set
        y.append(data[i, -1])  # Assuming Price_Difference is last column
    
    return np.array(X), np.array(y)

def prepare_data(csv_file='processed_stock_data.csv', timesteps=60, test_size=0.2, use_lstm=True):
    """
    Load and prepare data for model training.
    
    Args:
        csv_file: Path to processed data CSV file
        timesteps: Number of previous days to look at (for LSTM)
        test_size: Proportion of data for testing (default 0.2 = 20%)
        use_lstm: If True, prepare 3D sequences for LSTM. If False, prepare 2D data for XGBoost
        
    Returns:
        X_train, X_test, y_train, y_test: Training and testing sets
        scaler: Fitted scaler for inverse transformation if needed
        feature_names: List of feature names
    """
    # Load processed data
    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
    
    # Use normalized features for training
    # Include ALL new features: prices, volumes, moving averages, RSI, MACD, sentiment, options
    feature_columns_normalized = [
        # Core price data
        'AAPL_normalized', 'MSFT_normalized', 'Price_Difference_normalized',
        # Volume data
        'AAPL_Volume_normalized', 'MSFT_Volume_normalized',
        # Moving averages
        'AAPL_MA5_normalized', 'MSFT_MA5_normalized',
        'AAPL_MA20_normalized', 'MSFT_MA20_normalized',
        'AAPL_Volume_MA5_normalized', 'MSFT_Volume_MA5_normalized',
        # Technical indicators - RSI
        'AAPL_RSI_normalized', 'MSFT_RSI_normalized',
        # Technical indicators - MACD
        'AAPL_MACD_normalized', 'MSFT_MACD_normalized',
        'AAPL_MACD_Signal_normalized', 'MSFT_MACD_Signal_normalized',
        'AAPL_MACD_Histogram_normalized', 'MSFT_MACD_Histogram_normalized',
        # Sentiment
        'AAPL_Reddit_Sentiment_normalized', 'MSFT_Reddit_Sentiment_normalized',
        # Options volume
        'AAPL_Options_Volume_normalized', 'MSFT_Options_Volume_normalized'
    ]
    
    # Fallback to original columns if normalized don't exist
    feature_columns_original = [
        # Core price data
        'AAPL', 'MSFT', 'Price_Difference',
        # Volume data
        'AAPL_Volume', 'MSFT_Volume',
        # Moving averages
        'AAPL_MA5', 'MSFT_MA5',
        'AAPL_MA20', 'MSFT_MA20',
        'AAPL_Volume_MA5', 'MSFT_Volume_MA5',
        # Technical indicators - RSI
        'AAPL_RSI', 'MSFT_RSI',
        # Technical indicators - MACD
        'AAPL_MACD', 'MSFT_MACD',
        'AAPL_MACD_Signal', 'MSFT_MACD_Signal',
        'AAPL_MACD_Histogram', 'MSFT_MACD_Histogram',
        # Sentiment
        'AAPL_Reddit_Sentiment', 'MSFT_Reddit_Sentiment',
        # Options volume
        'AAPL_Options_Volume', 'MSFT_Options_Volume'
    ]
    
    # Try to use normalized columns first
    available_normalized = [col for col in feature_columns_normalized if col in df.columns]
    available_original = [col for col in feature_columns_original if col in df.columns]
    
    if len(available_normalized) > len(available_original):
        feature_columns = available_normalized
        print(f"Using normalized features ({len(feature_columns)} features)")
        data = df[feature_columns].values
        scaler = None  # Data already normalized
    elif available_original:
        feature_columns = available_original
        print(f"Using original features and scaling ({len(feature_columns)} features)")
        scaler = StandardScaler()
        data = scaler.fit_transform(df[feature_columns])
    else:
        # Fallback to basic features if new ones aren't available
        print("Warning: New features not found. Using basic features only.")
        basic_features = [
            'AAPL_normalized', 'MSFT_normalized', 'Price_Difference_normalized',
            'AAPL_Volume_normalized', 'MSFT_Volume_normalized',
            'AAPL_MA5_normalized', 'MSFT_MA5_normalized',
            'AAPL_MA20_normalized', 'MSFT_MA20_normalized',
            'AAPL_Volume_MA5_normalized', 'MSFT_Volume_MA5_normalized'
        ]
        feature_columns = [col for col in basic_features if col in df.columns]
        if not feature_columns:
            feature_columns = ['AAPL', 'MSFT', 'Price_Difference', 'AAPL_Volume', 'MSFT_Volume']
            scaler = StandardScaler()
            data = scaler.fit_transform(df[feature_columns])
        else:
            data = df[feature_columns].values
            scaler = None
    
    print(f"Features included: {len(feature_columns)}")
    print(f"Feature list: {feature_columns}")
    
    if use_lstm:
        # Create sequences for LSTM
        print(f"Creating sequences with {timesteps} timesteps...")
        X, y = create_sequences(data, timesteps)
        print(f"Sequences created: X shape {X.shape}, y shape {y.shape}")
    else:
        # For XGBoost, flatten the data (use current day's features only)
        # We'll create sequences but flatten them
        print(f"Preparing data for XGBoost...")
        print(f"Note: XGBoost uses current day features. Historical context is not used.")
        # For XGBoost, we use each day's features directly
        X = data[:-1]  # All rows except last
        y = data[1:, data.shape[1]-1]  # Price_Difference shifted by 1 day
        
        # Find Price_Difference index
        price_diff_idx = None
        for i, col in enumerate(feature_columns):
            if 'Price_Difference' in col:
                price_diff_idx = i
                break
        
        if price_diff_idx is not None:
            y = data[1:, price_diff_idx]
        else:
            y = data[1:, -1]  # Fallback to last column
        
        print(f"Data prepared: X shape {X.shape}, y shape {y.shape}")
    
    # Split into training and testing sets (80% train, 20% test)
    print(f"Splitting data: {int((1-test_size)*100)}% train, {int(test_size*100)}% test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False  # shuffle=False for time series
    )
    
    print(f"Training set: X_train {X_train.shape}, y_train {y_train.shape}")
    print(f"Testing set: X_test {X_test.shape}, y_test {y_test.shape}")
    
    return X_train, X_test, y_train, y_test, scaler, feature_columns

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

def train_xgboost_model(X_train, y_train, X_test, y_test, n_estimators=100, max_depth=6, learning_rate=0.1):
    """
    Train an XGBoost model as a baseline for comparison.
    
    Args:
        X_train, y_train: Training data (2D arrays)
        X_test, y_test: Testing data (2D arrays)
        n_estimators: Number of boosting rounds
        max_depth: Maximum tree depth
        learning_rate: Learning rate
        
    Returns:
        Trained XGBoost model and evaluation metrics
    """
    if not XGBOOST_AVAILABLE:
        raise ImportError("XGBoost is not available. Install with: pip install xgboost")
    
    print(f"\nTraining XGBoost model...")
    print(f"Parameters: n_estimators={n_estimators}, max_depth={max_depth}, learning_rate={learning_rate}")
    
    # Create and train XGBoost model
    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train, verbose=False)
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print("\n" + "=" * 50)
    print("XGBoost Model Evaluation:")
    print("=" * 50)
    print(f"Training - MSE: {train_mse:.6f}, MAE: {train_mae:.6f}, R²: {train_r2:.6f}")
    print(f"Testing  - MSE: {test_mse:.6f}, MAE: {test_mae:.6f}, R²: {test_r2:.6f}")
    
    return model, {
        'train_mse': train_mse,
        'test_mse': test_mse,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_r2': train_r2,
        'test_r2': test_r2
    }

def train_model(X_train, y_train, X_test, y_test, epochs=10, batch_size=32, dropout_rate=0.2):
    """
    Train the improved LSTM model.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Testing data
        epochs: Number of training epochs
        batch_size: Batch size for training
        dropout_rate: Dropout rate for regularization (default 0.2)
        
    Returns:
        Trained model and training history
    """
    # Get input shape from training data
    timesteps, n_features = X_train.shape[1], X_train.shape[2]
    input_shape = (timesteps, n_features)
    
    print(f"\nCreating improved LSTM model with input shape: {input_shape}")
    print(f"Features: {n_features} (includes price, volume, moving averages, RSI, MACD, sentiment, options)")
    print(f"Dropout rate: {dropout_rate}")
    model = create_lstm_model(input_shape, dropout_rate=dropout_rate)
    
    print("\nModel Architecture:")
    print("=" * 50)
    model.summary()
    
    # Train the model
    print(f"\nTraining model for {epochs} epochs with batch size {batch_size}...")
    print("=" * 50)
    
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    # Evaluate the model
    print("\n" + "=" * 50)
    print("LSTM Model Evaluation:")
    print("=" * 50)
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    
    # Calculate additional metrics
    y_test_pred = model.predict(X_test, verbose=0).flatten()
    test_r2 = r2_score(y_test, y_test_pred)
    
    print(f"Test Loss (MSE): {test_loss:.6f}")
    print(f"Test MAE: {test_mae:.6f}")
    print(f"Test R²: {test_r2:.6f}")
    
    return model, history

if __name__ == "__main__":
    try:
        # Configuration
        TIMESTEPS = 60  # Look back 60 days (for LSTM)
        TEST_SIZE = 0.2  # 20% for testing, 80% for training
        EPOCHS = 10
        BATCH_SIZE = 32
        
        print("=" * 70)
        print("STOCK ARBITRAGE MODEL TRAINING")
        print("=" * 70)
        print("\nThis script will:")
        print("1. Train an LSTM model with enhanced features (RSI, MACD, sentiment, options)")
        print("2. Train an XGBoost model for comparison")
        
        # Prepare data for LSTM
        print("\n" + "=" * 70)
        print("STEP 1: Preparing data for LSTM")
        print("=" * 70)
        X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm, scaler, feature_names = prepare_data(
            timesteps=TIMESTEPS,
            test_size=TEST_SIZE,
            use_lstm=True
        )
        
        # Train LSTM model
        print("\n" + "=" * 70)
        print("STEP 2: Training LSTM Model")
        print("=" * 70)
        lstm_model, lstm_history = train_model(
            X_train_lstm, y_train_lstm, X_test_lstm, y_test_lstm,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE
        )
        
        # Save the trained LSTM model
        lstm_model_path = 'lstm_price_difference_model.h5'
        lstm_model.save(lstm_model_path)
        print(f"\nLSTM model saved to: {lstm_model_path}")
        
        # Prepare data for XGBoost
        if XGBOOST_AVAILABLE:
            print("\n" + "=" * 70)
            print("STEP 3: Training XGBoost Baseline Model")
            print("=" * 70)
            X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb, _, _ = prepare_data(
                timesteps=TIMESTEPS,
                test_size=TEST_SIZE,
                use_lstm=False
            )
            
            # Train XGBoost model
            xgb_model, xgb_metrics = train_xgboost_model(
                X_train_xgb, y_train_xgb, X_test_xgb, y_test_xgb,
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1
            )
            
            # Save XGBoost model
            xgb_model_path = 'xgb_price_difference_model.json'
            xgb_model.save_model(xgb_model_path)
            print(f"\nXGBoost model saved to: {xgb_model_path}")
        
        print("\n" + "=" * 70)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"\nModels trained with {len(feature_names)} features:")
        print(f"  • Price data (AAPL, MSFT, Price_Difference)")
        print(f"  • Volume data")
        print(f"  • Moving averages (MA5, MA20)")
        print(f"  • Technical indicators (RSI, MACD)")
        print(f"  • Reddit sentiment")
        print(f"  • Options volume")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease run process_stock_data.py first to generate the processed data file.")
    except ImportError as e:
        print(f"Error: {e}")
        print("\nTo use this script:")
        print("1. Install TensorFlow: pip install tensorflow")
        print("2. Install XGBoost (optional): pip install xgboost")
        print("3. Note: TensorFlow requires Python 3.11 or 3.12 (not 3.14)")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
