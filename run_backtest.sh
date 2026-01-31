#!/bin/bash
# Script to run backtest using Python 3.12 virtual environment (for TensorFlow support)

# Get script directory (no hardcoded path)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate Python 3.12 virtual environment if it exists
if [ -f "venv_py312/bin/activate" ]; then
    source venv_py312/bin/activate
    echo "Using Python 3.12 virtual environment (TensorFlow support)"
    python --version
else
    echo "Warning: Python 3.12 virtual environment not found. Using system Python."
    echo "Note: TensorFlow requires Python 3.11 or 3.12"
fi

# Run backtest
python backtest_strategy.py
