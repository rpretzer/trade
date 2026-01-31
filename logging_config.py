"""
Production Logging Configuration

Structured logging with rotation, levels, and centralization support.
Replaces basic print() statements with proper logging.
"""

import logging
import logging.handlers
import json
import sys
import os
import uuid
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any
import threading

# Thread-local storage for correlation IDs
_thread_local = threading.local()


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging"""

    def format(self, record):
        """Format log record as JSON"""
        log_data = {
            'timestamp': datetime.now(timezone.utc).isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'thread': record.thread,
            'thread_name': record.threadName
        }

        # Add correlation ID if present
        correlation_id = getattr(_thread_local, 'correlation_id', None)
        if correlation_id:
            log_data['correlation_id'] = correlation_id

        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }

        # Add custom fields from extra dict
        if hasattr(record, 'extra_fields'):
            log_data.update(record.extra_fields)

        return json.dumps(log_data)


class HumanReadableFormatter(logging.Formatter):
    """Human-readable formatter for console output"""

    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'
    }

    def format(self, record):
        """Format log record for human reading"""
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']

        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S')

        # Add correlation ID if present
        correlation_id = getattr(_thread_local, 'correlation_id', None)
        corr_str = f" [{correlation_id[:8]}]" if correlation_id else ""

        # Build log message
        msg = f"{timestamp} {color}{record.levelname:<8}{reset}{corr_str} [{record.name}] {record.getMessage()}"

        # Add exception if present
        if record.exc_info:
            msg += "\n" + self.formatException(record.exc_info)

        return msg


class LoggingConfig:
    """Centralized logging configuration"""

    def __init__(
        self,
        log_dir: str = 'logs',
        log_level: str = 'INFO',
        max_bytes: int = 10 * 1024 * 1024,  # 10 MB
        backup_count: int = 30,  # 30 days
        enable_console: bool = True,
        enable_file: bool = True,
        enable_json: bool = True,
        enable_syslog: bool = False,
        syslog_address: Optional[tuple] = None
    ):
        """
        Initialize logging configuration.

        Args:
            log_dir: Directory for log files
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            max_bytes: Max size per log file before rotation
            backup_count: Number of backup files to keep
            enable_console: Enable console logging
            enable_file: Enable file logging
            enable_json: Enable JSON-formatted file logging
            enable_syslog: Enable syslog logging (for centralization)
            syslog_address: Syslog server address (host, port)
        """
        self.log_dir = Path(log_dir)
        self.log_level = getattr(logging, log_level.upper())
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.enable_console = enable_console
        self.enable_file = enable_file
        self.enable_json = enable_json
        self.enable_syslog = enable_syslog
        self.syslog_address = syslog_address or ('localhost', 514)

        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Configure root logger
        self._configure_root_logger()

    def _configure_root_logger(self):
        """Configure the root logger"""
        root_logger = logging.getLogger()
        root_logger.setLevel(self.log_level)

        # Remove existing handlers
        root_logger.handlers.clear()

        # Add console handler
        if self.enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.log_level)
            console_handler.setFormatter(HumanReadableFormatter())
            root_logger.addHandler(console_handler)

        # Add rotating file handler (human-readable)
        if self.enable_file:
            file_handler = logging.handlers.RotatingFileHandler(
                self.log_dir / 'trading_system.log',
                maxBytes=self.max_bytes,
                backupCount=self.backup_count
            )
            file_handler.setLevel(self.log_level)
            file_handler.setFormatter(HumanReadableFormatter())
            root_logger.addHandler(file_handler)

        # Add JSON file handler (for log aggregation)
        if self.enable_json:
            json_handler = logging.handlers.RotatingFileHandler(
                self.log_dir / 'trading_system.json.log',
                maxBytes=self.max_bytes,
                backupCount=self.backup_count
            )
            json_handler.setLevel(self.log_level)
            json_handler.setFormatter(StructuredFormatter())
            root_logger.addHandler(json_handler)

        # Add syslog handler (for centralized logging)
        if self.enable_syslog:
            try:
                syslog_handler = logging.handlers.SysLogHandler(
                    address=self.syslog_address
                )
                syslog_handler.setLevel(self.log_level)
                syslog_handler.setFormatter(StructuredFormatter())
                root_logger.addHandler(syslog_handler)
            except Exception as e:
                root_logger.warning(f"Failed to configure syslog handler: {e}")

    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger instance"""
        return logging.getLogger(name)


class PerformanceLogger:
    """Context manager for logging performance metrics"""

    def __init__(self, operation: str, logger: logging.Logger, level: int = logging.INFO, **kwargs):
        """
        Initialize performance logger.

        Args:
            operation: Name of the operation being timed
            logger: Logger instance to use
            level: Log level for the performance message
            **kwargs: Additional fields to include in log
        """
        self.operation = operation
        self.logger = logger
        self.level = level
        self.extra_fields = kwargs
        self.start_time = None

    def __enter__(self):
        """Start timing"""
        self.start_time = datetime.now()
        self.logger.log(
            self.level,
            f"Starting: {self.operation}",
            extra={'extra_fields': self.extra_fields}
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End timing and log"""
        elapsed = (datetime.now() - self.start_time).total_seconds()

        log_fields = {
            **self.extra_fields,
            'operation': self.operation,
            'elapsed_seconds': elapsed,
            'success': exc_type is None
        }

        if exc_type:
            log_fields['error_type'] = exc_type.__name__
            log_fields['error_message'] = str(exc_val)

        msg = f"Completed: {self.operation} (took {elapsed:.2f}s)"
        if exc_type:
            msg += f" - FAILED: {exc_type.__name__}"

        self.logger.log(
            logging.ERROR if exc_type else self.level,
            msg,
            extra={'extra_fields': log_fields}
        )


def set_correlation_id(correlation_id: Optional[str] = None):
    """
    Set correlation ID for current thread.

    Args:
        correlation_id: Correlation ID to set (generates UUID if None)
    """
    if correlation_id is None:
        correlation_id = str(uuid.uuid4())
    _thread_local.correlation_id = correlation_id
    return correlation_id


def get_correlation_id() -> Optional[str]:
    """Get correlation ID for current thread"""
    return getattr(_thread_local, 'correlation_id', None)


def clear_correlation_id():
    """Clear correlation ID for current thread"""
    if hasattr(_thread_local, 'correlation_id'):
        delattr(_thread_local, 'correlation_id')


# Global logging config instance
_logging_config = None


def init_logging(
    log_dir: str = 'logs',
    log_level: str = None,
    **kwargs
) -> LoggingConfig:
    """
    Initialize global logging configuration.

    Args:
        log_dir: Directory for log files
        log_level: Logging level (defaults to INFO, or DEBUG if DEBUG env var set)
        **kwargs: Additional arguments for LoggingConfig

    Returns:
        LoggingConfig instance
    """
    global _logging_config

    # Determine log level
    if log_level is None:
        if os.environ.get('DEBUG', '').lower() in ['true', '1', 'yes']:
            log_level = 'DEBUG'
        else:
            log_level = os.environ.get('LOG_LEVEL', 'INFO')

    _logging_config = LoggingConfig(
        log_dir=log_dir,
        log_level=log_level,
        **kwargs
    )

    return _logging_config


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    global _logging_config

    if _logging_config is None:
        # Auto-initialize with defaults
        init_logging()

    return logging.getLogger(name)


# Convenience functions for common log operations
def log_trade(logger: logging.Logger, trade_data: Dict[str, Any]):
    """Log a trade with structured data"""
    logger.info(
        f"Trade executed: {trade_data.get('symbol')} {trade_data.get('side')} {trade_data.get('quantity')}",
        extra={'extra_fields': {'trade': trade_data}}
    )


def log_prediction(logger: logging.Logger, prediction_data: Dict[str, Any]):
    """Log a model prediction with structured data"""
    logger.info(
        f"Prediction: {prediction_data.get('symbol_1')}/{prediction_data.get('symbol_2')} = {prediction_data.get('predicted_difference')}",
        extra={'extra_fields': {'prediction': prediction_data}}
    )


def log_risk_event(logger: logging.Logger, event_data: Dict[str, Any]):
    """Log a risk management event"""
    logger.warning(
        f"Risk event: {event_data.get('event_type')} - {event_data.get('reason')}",
        extra={'extra_fields': {'risk_event': event_data}}
    )


# Example usage
if __name__ == '__main__':
    # Initialize logging
    logging_config = init_logging(log_level='DEBUG')

    # Get logger
    logger = get_logger(__name__)

    # Set correlation ID for request tracking
    correlation_id = set_correlation_id()
    logger.info(f"Started processing request with correlation ID: {correlation_id}")

    # Log at different levels
    logger.debug("Debug message with details")
    logger.info("Information message")
    logger.warning("Warning message")
    logger.error("Error message")

    # Log with performance tracking
    with PerformanceLogger("database_query", logger, operation_type="SELECT"):
        import time
        time.sleep(0.5)  # Simulate work

    # Log structured data
    log_trade(logger, {
        'symbol': 'AAPL',
        'side': 'BUY',
        'quantity': 100,
        'price': 150.00
    })

    # Log exception
    try:
        raise ValueError("Example error")
    except Exception:
        logger.exception("An error occurred during processing")

    # Clear correlation ID
    clear_correlation_id()

    print(f"\nLog files created in: {logging_config.log_dir}/")
    print("- trading_system.log (human-readable)")
    print("- trading_system.json.log (structured)")
