#!/bin/bash
# Daily stock data update script
# This script runs process_stock_data.py daily via cron

# Set the working directory to the script's directory (no hardcoded path)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Set Python path
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# Log file location
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/data_update_$(date +%Y%m%d).log"

# Activate Python 3.12 virtual environment if it exists (for TensorFlow support)
if [ -f "venv_py312/bin/activate" ]; then
    source venv_py312/bin/activate
    PYTHON_CMD="python"
else
    PYTHON_CMD="python3"
fi

# Run the data processing script
$PYTHON_CMD process_stock_data.py >> "$LOG_FILE" 2>&1

# Exit with the Python script's exit code
exit $?
