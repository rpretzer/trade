#!/bin/bash
# Daily stock data update script
# This script runs process_stock_data.py daily via cron

# Set the working directory to the script's directory
cd /home/rpretzer/stock_arbitrage_model

# Set Python path
export PYTHONPATH="/home/rpretzer/stock_arbitrage_model:$PYTHONPATH"

# Log file location
LOG_DIR="/home/rpretzer/stock_arbitrage_model/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/data_update_$(date +%Y%m%d).log"

# Run the data processing script
python3 process_stock_data.py >> "$LOG_FILE" 2>&1

# Exit with the Python script's exit code
exit $?
