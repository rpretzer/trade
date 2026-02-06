"""
Application Constants
Centralized constants to avoid magic numbers throughout the codebase
"""

# ============================================================================
# Ticker Configuration
# ============================================================================

# The two symbols used for the pairs trade.  Change these to switch pairs.
# After changing, re-run process_stock_data.py → train_model.py before
# backtesting or live trading.
TICKER_LONG  = 'AAPL'
TICKER_SHORT = 'MSFT'

# Trading signal strings — derived from tickers; do not edit directly.
SIGNAL_BUY_LONG  = f'Buy {TICKER_LONG}, Sell {TICKER_SHORT}'
SIGNAL_BUY_SHORT = f'Buy {TICKER_SHORT}, Sell {TICKER_LONG}'
SIGNAL_HOLD      = 'Hold'

# Per-symbol price-sanity ranges (min, max).  Used by trading_api as a
# guard before executing orders.  Extend when adding new ticker pairs.
PRICE_SANITY_RANGES = {
    'AAPL': (50,   500),
    'MSFT': (100, 1000),
}

# ---------------------------------------------------------------------------
# Feature-column helpers
# ---------------------------------------------------------------------------

# Suffixes appended to each ticker to form feature column names.
# Order must match the column order written by process_stock_data.
_FEATURE_SUFFIXES = [
    '',                   # base close price
    '_Volume',
    '_MA5',
    '_MA20',
    '_Volume_MA5',
    '_RSI',
    '_MACD',
    '_MACD_Signal',
    '_MACD_Histogram',
    '_Reddit_Sentiment',
    '_Options_Volume',
]

# The first N suffix groups present in the "basic" feature set (when the
# full set of indicators is unavailable).
_BASIC_SUFFIX_COUNT = 5


def feature_names(ticker_long=None, ticker_short=None, normalized=True, basic_only=False):
    """Return the ordered list of feature column names for the ticker pair.

    Args:
        ticker_long:  Long-leg ticker (default TICKER_LONG).
        ticker_short: Short-leg ticker (default TICKER_SHORT).
        normalized:   If True, append '_normalized' to every name.
        basic_only:   If True, return only the first 5 feature groups
                      (price, volume, MA5, MA20, volume MA5).

    Returns:
        List of column-name strings in the order expected by the models.
    """
    tl = ticker_long or TICKER_LONG
    ts = ticker_short or TICKER_SHORT
    sfx = '_normalized' if normalized else ''
    suffixes = _FEATURE_SUFFIXES[:_BASIC_SUFFIX_COUNT] if basic_only else _FEATURE_SUFFIXES

    cols = [f'{tl}{sfx}', f'{ts}{sfx}', f'Price_Difference{sfx}']
    for s in suffixes[1:]:          # base price already covered above
        cols.append(f'{tl}{s}{sfx}')
        cols.append(f'{ts}{s}{sfx}')
    return cols


# ============================================================================
# Machine Learning Constants
# ============================================================================

# LSTM Architecture
LSTM_UNITS_LAYER1 = 50
LSTM_UNITS_LAYER2 = 30
LSTM_DROPOUT_RATE = 0.2
LSTM_TIMESTEPS = 60  # Number of days to look back

# Training Parameters
TRAINING_EPOCHS = 100
TRAINING_BATCH_SIZE = 32
TRAINING_VALIDATION_SPLIT = 0.2
EARLY_STOPPING_PATIENCE = 10

# Model Evaluation
MIN_SAMPLES_FOR_TRAINING = 1000
MIN_SAMPLES_FOR_PREDICTION = 10


# ============================================================================
# Trading Strategy Constants
# ============================================================================

# Signal Thresholds
DEFAULT_TRADING_THRESHOLD = 0.5  # Normalized price difference threshold
MIN_CONFIDENCE_THRESHOLD = 0.3
HIGH_CONFIDENCE_THRESHOLD = 0.7

# Position Sizing
DEFAULT_POSITION_SIZE_PCT = 0.10  # 10% of capital per trade
MAX_POSITION_SIZE_PCT = 0.25  # Maximum 25% per trade
MIN_POSITION_SIZE_USD = 100  # Minimum $100 position

# Risk Management
DEFAULT_MAX_DRAWDOWN_PCT = 0.05  # 5% maximum drawdown
CRITICAL_DRAWDOWN_PCT = 0.10  # 10% critical drawdown (halt trading)
DEFAULT_STOP_LOSS_PCT = 0.02  # 2% stop-loss per trade
TRAILING_STOP_PCT = 0.03  # 3% trailing stop

# Risk Limits (from risk_management.py defaults)
MAX_SINGLE_POSITION_PCT = 0.15  # 15% max concentration per symbol
MAX_SECTOR_EXPOSURE_PCT = 0.40  # 40% max sector exposure
MAX_TOTAL_EXPOSURE = 1.0  # 100% max total exposure
DEFAULT_MAX_POSITION_SIZE = 1000  # shares
DEFAULT_MAX_POSITION_VALUE = 50000  # USD


# ============================================================================
# Transaction Costs
# ============================================================================

# Commissions
COMMISSION_PER_SHARE_USD = 0.005  # $0.005 per share
MIN_COMMISSION_USD = 0.0  # Many brokers have $0 minimums now
MAX_COMMISSION_USD = 6.95  # Cap for large orders

# Fees
SEC_FEE_RATE = 0.0000278  # SEC fee (2024 rate)
TAF_FEE_RATE = 0.000166  # Trading Activity Fee
FINRA_FEE_PER_SHARE = 0.000166  # FINRA Trading Activity Fee

# Slippage
DEFAULT_SLIPPAGE_BPS = 1.0  # 1 basis point (0.01%)
HIGH_LIQUIDITY_SLIPPAGE_BPS = 0.5  # For liquid stocks
LOW_LIQUIDITY_SLIPPAGE_BPS = 5.0  # For illiquid stocks

# Borrow Costs (annual rates)
EASY_BORROW_RATE_ANNUAL = 0.01  # 1% per year
MODERATE_BORROW_RATE_ANNUAL = 0.05  # 5% per year
HARD_BORROW_RATE_ANNUAL = 0.15  # 15% per year

# Margin
MARGIN_INTEREST_RATE_ANNUAL = 0.08  # 8% annual
MAINTENANCE_MARGIN_PCT = 0.25  # 25% maintenance margin
INITIAL_MARGIN_PCT = 0.50  # 50% initial margin


# ============================================================================
# Market Hours (Eastern Time)
# ============================================================================

MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 30
MARKET_CLOSE_HOUR = 16
MARKET_CLOSE_MINUTE = 0

PRE_MARKET_OPEN_HOUR = 4
PRE_MARKET_OPEN_MINUTE = 0
AFTER_HOURS_CLOSE_HOUR = 20
AFTER_HOURS_CLOSE_MINUTE = 0


# ============================================================================
# Data Collection
# ============================================================================

# yfinance
DEFAULT_DATA_INTERVAL = "1d"  # Daily data
MAX_RETRIES_DATA_FETCH = 3
RETRY_DELAY_SECONDS = 5

# Moving Averages
MA_SHORT_WINDOW = 5  # 5-day MA
MA_LONG_WINDOW = 20  # 20-day MA
MA_VOLUME_WINDOW = 5  # 5-day volume MA

# Technical Indicators
RSI_PERIOD = 14  # Standard RSI period
MACD_FAST_PERIOD = 12
MACD_SLOW_PERIOD = 26
MACD_SIGNAL_PERIOD = 9

# Data Validation
MAX_ALLOWED_GAP_DAYS = 5  # Maximum gap in daily data
MIN_PRICE_USD = 1.0  # Penny stocks below this are excluded
MAX_PRICE_CHANGE_PCT = 0.50  # 50% - flag as potential data error


# ============================================================================
# API & Security
# ============================================================================

# Rate Limiting
API_RATE_LIMIT_CALLS = 2.0  # Calls per second
API_RATE_LIMIT_CAPACITY = 10  # Burst capacity

# Request Signing
API_SIGNATURE_MAX_AGE_SECONDS = 300  # 5 minutes
API_NONCE_CACHE_SIZE = 10000

# Circuit Breaker
CIRCUIT_BREAKER_FAILURE_THRESHOLD = 5  # Failures before opening
CIRCUIT_BREAKER_RECOVERY_TIMEOUT = 60.0  # Seconds

# Retry Logic
DEFAULT_RETRY_MAX_ATTEMPTS = 3
DEFAULT_RETRY_INITIAL_DELAY = 1.0  # seconds
DEFAULT_RETRY_MAX_DELAY = 60.0  # seconds
DEFAULT_RETRY_EXPONENTIAL_BASE = 2.0


# ============================================================================
# Model Drift Detection
# ============================================================================

# Performance Monitoring
DRIFT_DETECTION_WINDOW_SIZE = 100  # Recent predictions to track
DRIFT_ALERT_THRESHOLD_PCT = 0.15  # 15% degradation triggers alert
MIN_PREDICTIONS_FOR_DRIFT_CHECK = 30

# Feature Drift
DRIFT_SIGNIFICANCE_LEVEL = 0.01  # p-value threshold

# Thresholds for severity
DRIFT_LOW_THRESHOLD = 0.15  # 15%
DRIFT_MEDIUM_THRESHOLD = 0.25  # 25%
DRIFT_HIGH_THRESHOLD = 0.40  # 40%


# ============================================================================
# File Paths
# ============================================================================

# Data Files
PROCESSED_DATA_FILE = "processed_stock_data.csv"
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"

# Model Files
LSTM_MODEL_FILE = "lstm_price_difference_model.h5"
XGBOOST_MODEL_FILE = "xgb_price_difference_model.json"
SCALER_FILE = "scaler.pkl"
MODEL_REGISTRY_FILE = "model_registry.json"

# Configuration Files
STOCKS_CONFIG_FILE = "selected_stocks.txt"
DATE_RANGE_CONFIG_FILE = "date_range_config.txt"
MARKET_CONFIG_FILE = "market_config.txt"

# Logs
LOG_DIR = "logs"
AUDIT_LOG_DIR = "logs/audit"
TRADING_LOG_DIR = "logs/trading"


# ============================================================================
# Display & Formatting
# ============================================================================

CURRENCY_FORMAT = "${:,.2f}"
PERCENTAGE_FORMAT = "{:.2f}%"
FLOAT_PRECISION = 4
DATE_FORMAT = "%Y-%m-%d"
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
