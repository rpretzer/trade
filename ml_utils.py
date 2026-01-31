"""
Machine Learning Utilities
Shared utility functions for ML model training and prediction
"""

import numpy as np
from typing import Tuple


# Common constants
DEFAULT_TIMESTEPS = 60  # Default number of timesteps for LSTM sequences
MIN_SAMPLES_FOR_TRAINING = 1000  # Minimum samples needed for reliable training


def create_sequences(data: np.ndarray, timesteps: int = DEFAULT_TIMESTEPS) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for LSTM input from time series data.

    Transforms flat time series data into sequences of length `timesteps` for LSTM models.
    The target value is the last column of each sequence.

    Args:
        data: Array with features (shape: samples, features)
              Last column should be the target variable
        timesteps: Number of previous timesteps to use for each sequence

    Returns:
        Tuple of (X, y):
        - X: Array of shape (samples, timesteps, features)
        - y: Array of shape (samples,) containing target values

    Example:
        >>> data = np.random.randn(100, 5)  # 100 samples, 5 features
        >>> X, y = create_sequences(data, timesteps=10)
        >>> X.shape
        (90, 10, 5)
        >>> y.shape
        (90,)

    Note:
        Output has (len(data) - timesteps) samples since we need `timesteps`
        previous values to create each sequence.
    """
    if len(data) < timesteps:
        raise ValueError(
            f"Data length ({len(data)}) must be at least timesteps ({timesteps})"
        )

    X, y = [], []

    for i in range(timesteps, len(data)):
        X.append(data[i - timesteps:i])
        y.append(data[i, -1])  # Price difference is the last column

    return np.array(X), np.array(y)


def validate_training_data(
    X: np.ndarray,
    y: np.ndarray,
    min_samples: int = MIN_SAMPLES_FOR_TRAINING
) -> None:
    """
    Validate that training data meets minimum requirements.

    Args:
        X: Feature array
        y: Target array
        min_samples: Minimum number of samples required

    Raises:
        ValueError: If data doesn't meet requirements
    """
    if len(X) != len(y):
        raise ValueError(f"X and y must have same length: X={len(X)}, y={len(y)}")

    if len(X) < min_samples:
        raise ValueError(
            f"Insufficient training data: {len(X)} samples < {min_samples} required"
        )

    if np.isnan(X).any():
        raise ValueError("Training data contains NaN values")

    if np.isinf(X).any():
        raise ValueError("Training data contains infinite values")


def split_time_series(
    X: np.ndarray,
    y: np.ndarray,
    train_size: float = 0.8,
    validation_size: float = 0.1,
    test_size: float = 0.1
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Split time series data into train/validation/test sets.

    Maintains temporal order (no shuffling) for time series data.

    Args:
        X: Feature array
        y: Target array
        train_size: Fraction for training (default 0.8)
        validation_size: Fraction for validation (default 0.1)
        test_size: Fraction for testing (default 0.1)

    Returns:
        Tuple of ((X_train, y_train), (X_val, y_val), (X_test, y_test))

    Raises:
        ValueError: If sizes don't sum to 1.0
    """
    if not np.isclose(train_size + validation_size + test_size, 1.0):
        raise ValueError(f"Sizes must sum to 1.0: got {train_size + validation_size + test_size}")

    n_samples = len(X)
    n_train = int(n_samples * train_size)
    n_val = int(n_samples * validation_size)

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
    X_test, y_test = X[n_train + n_val:], y[n_train + n_val:]

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
