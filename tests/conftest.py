"""
Shared pytest fixtures and module mocks.
"""

import sys
from unittest.mock import MagicMock

# Mock TensorFlow/Keras if not installed — required before any test module
# imports backtest_strategy.py (which does `import tensorflow as tf` at top level).
if 'tensorflow' not in sys.modules:
    try:
        import tensorflow  # noqa: F401
    except ImportError:
        sys.modules['tensorflow'] = MagicMock()
        sys.modules['tensorflow.keras'] = MagicMock()
