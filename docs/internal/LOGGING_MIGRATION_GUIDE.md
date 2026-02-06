# Logging Migration Guide

## Overview

Migrate from print() statements to production-grade structured logging.

## Quick Start

```python
# At the top of each file
from logging_config import get_logger

logger = get_logger(__name__)

# Replace print() with logger methods
logger.info("Message")      # Instead of print()
logger.debug("Debug info")  # For detailed debugging
logger.warning("Warning")   # For warnings
logger.error("Error")       # For errors
logger.exception("Error")   # For errors with stack traces
```

## Initialization

### In main.py or entry point:
```python
from logging_config import init_logging

# Initialize logging at application start
init_logging(
    log_dir='logs',
    log_level='INFO',  # or 'DEBUG' for development
    enable_console=True,
    enable_file=True,
    enable_json=True
)
```

## Migration Examples

### Before (Print Statements):
```python
print("Processing stock data...")
print(f"Loaded {len(df)} rows")
print(f"WARNING: Missing data for {symbol}")
print(f"ERROR: Failed to connect to API: {error}")
```

### After (Structured Logging):
```python
from logging_config import get_logger

logger = get_logger(__name__)

logger.info("Processing stock data...")
logger.info(f"Loaded {len(df)} rows")
logger.warning(f"Missing data for {symbol}")
logger.error(f"Failed to connect to API: {error}")
```

## Advanced Features

### 1. Correlation IDs (Request Tracking)
```python
from logging_config import set_correlation_id, get_logger

logger = get_logger(__name__)

# Set correlation ID at start of operation
correlation_id = set_correlation_id()  # Auto-generates UUID

# All subsequent logs will include this ID
logger.info("Started trade execution")
logger.info("Fetching prices")
logger.info("Placing order")

# Logs can be traced across operations using correlation_id
```

### 2. Performance Logging
```python
from logging_config import PerformanceLogger, get_logger

logger = get_logger(__name__)

# Automatically log execution time
with PerformanceLogger("backtest_execution", logger):
    # Run backtest
    results = run_backtest()

# Logs: "Completed: backtest_execution (took 5.23s)"
```

### 3. Structured Trade Logging
```python
from logging_config import log_trade, get_logger

logger = get_logger(__name__)

log_trade(logger, {
    'symbol': 'AAPL',
    'side': 'BUY',
    'quantity': 100,
    'price': 150.00,
    'commission': 0.50
})
```

### 4. Exception Logging
```python
logger = get_logger(__name__)

try:
    risky_operation()
except Exception as e:
    # Logs exception with full stack trace
    logger.exception("Operation failed")
    raise
```

## Log Levels

| Level | When to Use |
|-------|------------|
| `DEBUG` | Detailed information for diagnosing problems |
| `INFO` | General informational messages |
| `WARNING` | Warning messages (something unexpected but not an error) |
| `ERROR` | Error messages (operation failed) |
| `CRITICAL` | Critical errors (system is unusable) |

## File-by-File Migration Priority

### High Priority (User-facing):
1. `main.py` - Replace all print() with logging
2. `dashboard.py` - Replace Streamlit st.write() where appropriate
3. `live_trading.py` - Critical for production monitoring

### Medium Priority (Core Logic):
4. `backtest_strategy.py` - Add performance logging
5. `trading_api.py` - Log all API calls
6. `process_stock_data.py` - Log data processing steps

### Low Priority (Utilities):
7. `sentiment_analysis.py` - Log API calls
8. `options_data.py` - Log data fetching
9. `credential_manager.py` - Log security events

## Log Output

### Console Output (Human-Readable):
```
2026-01-29 10:15:30 INFO     [backtest_strategy] Starting backtest...
2026-01-29 10:15:31 INFO     [backtest_strategy] Loaded 500 rows
2026-01-29 10:15:35 INFO     [backtest_strategy] Completed: backtest_execution (took 5.23s)
2026-01-29 10:15:35 WARNING  [backtest_strategy] High drawdown detected: -12.5%
```

### JSON Log File (Machine-Readable):
```json
{
  "timestamp": "2026-01-29T10:15:30Z",
  "level": "INFO",
  "logger": "backtest_strategy",
  "message": "Starting backtest...",
  "module": "backtest_strategy",
  "function": "run_backtest",
  "line": 125,
  "correlation_id": "a3b4c5d6-e7f8-9012-3456-7890abcdef12"
}
```

## Configuration via Environment Variables

```bash
# Set log level
export LOG_LEVEL=DEBUG

# Enable debug mode
export DEBUG=true

# Database URL for audit logging
export DATABASE_URL="postgresql://localhost:5432/trading_db"
```

## Integration with Monitoring

JSON logs can be sent to:
- **ELK Stack** (Elasticsearch, Logstash, Kibana)
- **Splunk**
- **DataDog**
- **CloudWatch Logs**
- **Grafana Loki**

Just configure syslog or use log shippers to forward:
```python
init_logging(
    enable_syslog=True,
    syslog_address=('logserver.example.com', 514)
)
```

## Testing

```bash
# Test logging configuration
python logging_config.py

# Check log files
ls -l logs/
tail -f logs/trading_system.log
tail -f logs/trading_system.json.log
```

## Rollout Strategy

1. Initialize logging in main.py
2. Convert one file at a time
3. Test each file after conversion
4. Monitor log file sizes and rotation
5. Configure log aggregation (optional)

## Benefits

✅ Structured logs for easy parsing
✅ Automatic log rotation (prevents disk fill)
✅ Correlation IDs for request tracing
✅ Performance metrics built-in
✅ Exception stack traces
✅ JSON format for log aggregation
✅ Thread-safe
✅ Production-ready
