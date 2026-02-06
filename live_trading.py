"""
Live Trading Module
Gets latest trading signals and executes trades
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import sys
import json
import time as time_module
import signal as signal_module
import logging
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from exceptions import (
    ModelLoadError, DataNotFoundError, InsufficientDataError,
    ModelPredictionError, RiskException
)
from error_handling import retry, RetryConfig
from trading_api import TradingAPI
from risk_management import RiskManager, RiskLimits
from audit_logging import AuditLogger, AuditEventType
from constants import (
    feature_names, TICKER_LONG, TICKER_SHORT,
    SIGNAL_BUY_LONG, SIGNAL_BUY_SHORT, SIGNAL_HOLD,
)

logger = logging.getLogger(__name__)

# Default ensemble weights — (LSTM, XGBoost).  Tune via backtest before
# changing.  Enabled at runtime by setting ENSEMBLE_MODE=1 in the environment.
ENSEMBLE_WEIGHTS = (0.6, 0.4)

def create_sequences(data, timesteps):
    """Create sequences for LSTM input."""
    X, y = [], []
    for i in range(timesteps, len(data)):
        X.append(data[i-timesteps:i])
        y.append(data[i, -1])
    return np.array(X), np.array(y)

def ensemble_predict(lstm_pred, xgb_pred, weights=ENSEMBLE_WEIGHTS):
    """Weighted average of two model predictions.

    Args:
        lstm_pred: Scalar prediction from the LSTM model.
        xgb_pred: Scalar prediction from the XGBoost model.
        weights: Tuple (w_lstm, w_xgb).  Normalised internally so they
                 don't have to sum to 1.

    Returns:
        Blended prediction (float).
    """
    w_lstm, w_xgb = weights
    return (w_lstm * lstm_pred + w_xgb * xgb_pred) / (w_lstm + w_xgb)


def _xgb_predict(data, feature_columns):
    """Load the XGBoost model and return its prediction for the latest row.

    Resolves the model path from the registry first; falls back to the
    default file name when the registry has no usable XGBoost entry.

    Args:
        data: 2-D numpy array of features (rows × columns), already scaled.
        feature_columns: List of column names (used only for length).

    Returns:
        XGBoost prediction (float).

    Raises:
        ModelLoadError: If xgboost is not installed.
        Any exception from model loading / prediction is allowed to propagate
        so the caller can fall back to LSTM-only.
    """
    try:
        from xgboost import XGBRegressor
    except ImportError:
        raise ModelLoadError(
            "XGBoost not available for ensemble — install with: pip install xgboost"
        )

    # Resolve path: registry → hard-coded fallback
    xgb_path = 'xgb_price_difference_model.json'
    try:
        from model_management import ModelRegistry
        _reg = ModelRegistry()
        candidates = [
            m for m in _reg.list_models()
            if m.model_type == 'XGBoost'
            and m.status.value not in ('ARCHIVED', 'DEPRECATED')
        ]
        if candidates:
            candidates.sort(key=lambda m: m.created_at, reverse=True)
            xgb_path = candidates[0].model_path
            logger.info("XGBoost resolved from registry: %s", candidates[0].version)
    except Exception:
        logger.debug("XGBoost registry lookup failed — using fallback path")

    xgb_model = XGBRegressor()
    xgb_model.load_model(xgb_path)

    # XGBoost uses a single row (the most recent data point)
    X_xgb = data[-1:].reshape(1, len(feature_columns))
    return float(xgb_model.predict(X_xgb)[0])


@retry(
    exceptions=(ModelLoadError, DataNotFoundError),
    config=RetryConfig(max_attempts=3, initial_delay=1.0)
)
def get_latest_signal(model_path='lstm_price_difference_model.h5',
                     data_file='processed_stock_data.csv',
                     timesteps=60,
                     threshold=0.5):
    """
    Get the latest trading signal from the trained model.

    Args:
        model_path: Path to trained LSTM model
        data_file: Path to processed data CSV
        timesteps: Number of timesteps for LSTM
        threshold: Threshold for signal generation

    Returns:
        Trading signal string

    Raises:
        ModelLoadError: If model cannot be loaded
        DataNotFoundError: If data file is not found
        InsufficientDataError: If not enough data for prediction
        ModelPredictionError: If prediction fails
    """
    # Lazy tensorflow import — allows the rest of this module (trading loop,
    # helpers) to be imported without tensorflow installed.
    try:
        from tensorflow import keras
    except ImportError:
        raise ModelLoadError(
            "TensorFlow not available. Install with: pip install -r requirements-lstm.txt"
        )

    # Resolve model path through A/B router when registry is available
    try:
        from model_management import ModelRegistry, ABTestController
        _registry = ModelRegistry()
        _selected = ABTestController(_registry).select_model()
        model_path = _selected.model_path
        logger.info("Model selected via registry: %s [%s]", _selected.version, _selected.status.value)
    except Exception:
        logger.debug("Registry unavailable — using explicit model_path=%s", model_path)

    # Load model
    try:
        model = keras.models.load_model(model_path)
    except (FileNotFoundError, OSError) as e:
        raise ModelLoadError(f"Failed to load model from {model_path}: {e}")

    # Load data
    try:
        df = pd.read_csv(data_file, index_col=0, parse_dates=True)
    except FileNotFoundError as e:
        raise DataNotFoundError(f"Data file not found: {data_file}")
    except Exception as e:
        raise DataNotFoundError(f"Failed to load data from {data_file}: {e}")

    feature_columns_normalized = feature_names(normalized=True)
    feature_columns_original   = feature_names(normalized=False)

    available_normalized = [col for col in feature_columns_normalized if col in df.columns]
    available_original = [col for col in feature_columns_original if col in df.columns]

    # Use normalized features if available, otherwise use original with scaling
    if len(available_normalized) > len(available_original):
        feature_columns = available_normalized
        data = df[feature_columns].values
    else:
        feature_columns = available_original
        # Scale original features
        logger.info("Using original features with StandardScaler normalization")
        scaler = StandardScaler()
        try:
            data = scaler.fit_transform(df[feature_columns])
        except Exception as e:
            raise ModelPredictionError(f"Failed to scale features: {e}")

    # Check if we have enough data
    if len(data) < timesteps:
        raise InsufficientDataError(
            f"Not enough data for prediction. Need {timesteps} timesteps, "
            f"have {len(data)}"
        )

    # Get the last sequence
    X_latest = data[-timesteps:].reshape(1, timesteps, len(feature_columns))

    # Predict (LSTM)
    try:
        prediction = model.predict(X_latest, verbose=0)[0][0]
    except Exception as e:
        raise ModelPredictionError(f"Model prediction failed: {e}")

    # --- optional ensemble blend ----------------------------------------
    if os.environ.get('ENSEMBLE_MODE'):
        try:
            xgb_pred = _xgb_predict(data, feature_columns)
            prediction = ensemble_predict(prediction, xgb_pred)
            logger.info("Ensemble prediction (LSTM + XGBoost): %.4f", prediction)
        except Exception as e:
            logger.warning("Ensemble XGBoost failed — using LSTM only: %s", e)

    # Generate signal
    if prediction > threshold:
        logger.info(f"Signal: {SIGNAL_BUY_LONG} (prediction: {prediction:.4f})")
        return SIGNAL_BUY_LONG
    elif prediction < -threshold:
        logger.info(f"Signal: {SIGNAL_BUY_SHORT} (prediction: {prediction:.4f})")
        return SIGNAL_BUY_SHORT
    else:
        logger.info(f"Signal: {SIGNAL_HOLD} (prediction: {prediction:.4f})")
        return SIGNAL_HOLD


# ---------------------------------------------------------------------------
# Market-hours helpers
# ---------------------------------------------------------------------------

POSITIONS_FILE = 'positions.json'
DEFAULT_INTERVAL_SECONDS = 300  # 5 minutes
MAX_CONSECUTIVE_ERRORS = 5


def market_is_open():
    """
    Check whether NYSE is currently open.

    Uses pandas_market_calendars when the advanced profile is installed;
    falls back to a hard-coded Mon-Fri 9:30-16:00 ET window otherwise.
    """
    eastern = ZoneInfo('America/New_York')
    now = datetime.now(eastern)

    try:
        import pandas_market_calendars as mcal
        nyse = mcal.get_calendar('NYSE')
        schedule = nyse.schedule(
            start_date=now.strftime('%Y-%m-%d'),
            end_date=now.strftime('%Y-%m-%d'),
        )
        if schedule.empty:
            return False
        market_open = schedule.iloc[0]['open'].to_pydatetime()
        market_close = schedule.iloc[0]['close'].to_pydatetime()
        # Ensure timezone-aware comparison
        if market_open.tzinfo is None:
            market_open = market_open.replace(tzinfo=eastern)
        if market_close.tzinfo is None:
            market_close = market_close.replace(tzinfo=eastern)
        return market_open <= now <= market_close
    except ImportError:
        # Fallback: hard-coded NYSE hours
        if now.weekday() >= 5:  # Saturday / Sunday
            return False
        open_time = now.replace(hour=9, minute=30, second=0, microsecond=0)
        close_time = now.replace(hour=16, minute=0, second=0, microsecond=0)
        return open_time <= now <= close_time


# ---------------------------------------------------------------------------
# Position persistence
# ---------------------------------------------------------------------------


def load_positions(filepath=POSITIONS_FILE):
    """Load persisted position state from disk.  Returns {} if missing/corrupt."""
    path = Path(filepath)
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Could not load positions from {filepath}: {e}")
    return {}


def save_positions(positions, filepath=POSITIONS_FILE):
    """Persist position state to disk as JSON."""
    try:
        with open(filepath, 'w') as f:
            json.dump(positions, f, indent=2, default=str)
    except OSError as e:
        logger.error(f"Failed to save positions to {filepath}: {e}")


# ---------------------------------------------------------------------------
# Main trading loop
# ---------------------------------------------------------------------------


def run_trading_loop(
    paper_trading=True,
    interval_seconds=DEFAULT_INTERVAL_SECONDS,
    model_path='lstm_price_difference_model.h5',
    data_file='processed_stock_data.csv',
    threshold=0.5,
    initial_capital=10000.0,
):
    """
    Continuous trading loop: predict → risk-check → execute → log → sleep.

    Handles SIGINT / SIGTERM for graceful shutdown.  Persists position state
    to disk so the loop can resume after a restart without losing track of
    open positions.

    Args:
        paper_trading: If True all orders are simulated (default).
        interval_seconds: Sleep between cycles (default 300 s / 5 min).
        model_path: Path to the trained LSTM .h5 file.
        data_file: Path to the processed CSV data file.
        threshold: Signal-generation threshold.
        initial_capital: Starting capital used to size positions.
    """
    logger.info("Trading loop starting (paper_trading=%s)", paper_trading)

    # --- one-time setup ------------------------------------------------
    api = TradingAPI(paper_trading=paper_trading)
    api.connect()

    risk_manager = RiskManager(
        limits=RiskLimits(
            max_position_size=1000,
            max_position_value=initial_capital * 0.5,
            max_total_exposure=initial_capital * 2.0,
            max_drawdown_pct=0.05,
            critical_drawdown_pct=0.10,
        ),
        initial_capital=initial_capital,
    )

    audit_logger = AuditLogger()
    audit_logger.log(
        event_type=AuditEventType.SYSTEM_START,
        user='trading_loop',
        action='loop_started',
        resource='trading_loop',
        details={'paper_trading': paper_trading, 'interval_seconds': interval_seconds},
    )

    # Restore positions persisted from a previous run
    persisted = load_positions()
    for sym, pos in persisted.items():
        risk_manager.add_position(sym, pos['quantity'], pos['avg_price'])
    if persisted:
        logger.info("Restored %d persisted position(s)", len(persisted))

    # --- graceful shutdown on SIGINT / SIGTERM --------------------------
    shutdown_requested = False

    def _handle_shutdown(signum, _frame):
        nonlocal shutdown_requested
        logger.info("Received signal %d — finishing current cycle then stopping", signum)
        shutdown_requested = True

    signal_module.signal(signal_module.SIGINT, _handle_shutdown)
    signal_module.signal(signal_module.SIGTERM, _handle_shutdown)

    # --- main loop ------------------------------------------------------
    consecutive_errors = 0

    while not shutdown_requested:
        try:
            # 1. Market-hours gate
            if not market_is_open():
                logger.info("Market closed — sleeping 60 s")
                time_module.sleep(60)
                continue

            # 2. Get prediction signal
            signal = get_latest_signal(
                model_path=model_path,
                data_file=data_file,
                threshold=threshold,
            )
            logger.info("Signal: %s", signal)

            audit_logger.log(
                event_type=AuditEventType.PREDICTION_MADE,
                user='trading_loop',
                action='signal_generated',
                resource='model',
                details={'signal': signal},
            )

            if signal == SIGNAL_HOLD:
                logger.info("Hold — no trade this cycle")
                time_module.sleep(interval_seconds)
                consecutive_errors = 0
                continue

            # 3. Fetch live prices
            long_price  = api.get_current_price(TICKER_LONG)
            short_price = api.get_current_price(TICKER_SHORT)
            current_prices = {TICKER_LONG: long_price, TICKER_SHORT: short_price}

            # 4. Determine buy / sell legs
            if signal == SIGNAL_BUY_LONG:
                buy_sym, sell_sym = TICKER_LONG, TICKER_SHORT
                buy_price, sell_price = long_price, short_price
            else:  # SIGNAL_BUY_SHORT
                buy_sym, sell_sym = TICKER_SHORT, TICKER_LONG
                buy_price, sell_price = short_price, long_price

            position_value = initial_capital * 0.10  # 10 % of capital per trade
            buy_qty = max(1, int(position_value / buy_price))
            sell_qty = max(1, int(position_value / sell_price))

            # 5. Pre-trade risk checks (raises RiskException on violation)
            risk_manager.pre_trade_check(buy_sym, buy_qty, buy_price, prices=current_prices)
            risk_manager.pre_trade_check(sell_sym, -sell_qty, sell_price, prices=current_prices)

            # 6. Execute
            buy_order = api.place_order(buy_sym, buy_qty, 'BUY', 'MARKET')
            sell_order = api.place_order(sell_sym, sell_qty, 'SELL', 'MARKET')

            if buy_order and sell_order:
                # 7. Update risk manager & persist
                risk_manager.add_position(buy_sym, buy_qty, buy_price)
                risk_manager.add_position(sell_sym, -sell_qty, sell_price)
                save_positions({
                    sym: {'quantity': pos['quantity'], 'avg_price': pos['avg_price']}
                    for sym, pos in risk_manager.positions.items()
                })

                # 8. Audit trail
                audit_logger.log(
                    event_type=AuditEventType.ORDER_PLACED,
                    user='trading_loop',
                    action='pair_trade_executed',
                    resource=f'{buy_sym}/{sell_sym}',
                    details={
                        'buy': {'symbol': buy_sym, 'qty': buy_qty, 'price': buy_price},
                        'sell': {'symbol': sell_sym, 'qty': sell_qty, 'price': sell_price},
                        'paper_trading': paper_trading,
                    },
                )
                logger.info(
                    "Executed: buy %d %s @ %.2f, sell %d %s @ %.2f",
                    buy_qty, buy_sym, buy_price, sell_qty, sell_sym, sell_price,
                )
            else:
                logger.warning("One or both orders failed — no position update")

            consecutive_errors = 0  # reset on success

        except RiskException as e:
            # Risk block is an expected control-flow event, not an error
            logger.warning("Trade blocked by risk check: %s", e)
            audit_logger.log(
                event_type=AuditEventType.RISK_LIMIT_BREACH,
                user='trading_loop',
                action='risk_check_blocked',
                resource='risk_manager',
                status='WARNING',
                details={'reason': str(e)},
            )
            consecutive_errors = 0

        except Exception as e:
            consecutive_errors += 1
            logger.error(
                "Trading loop error (%d/%d): %s",
                consecutive_errors, MAX_CONSECUTIVE_ERRORS, e,
            )
            audit_logger.log(
                event_type=AuditEventType.ERROR_OCCURRED,
                user='trading_loop',
                action='loop_error',
                resource='trading_loop',
                status='FAILURE',
                details={'error': str(e), 'consecutive_errors': consecutive_errors},
            )

            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                logger.critical(
                    "Max consecutive errors (%d) reached — halting trading loop",
                    MAX_CONSECUTIVE_ERRORS,
                )
                audit_logger.log(
                    event_type=AuditEventType.CIRCUIT_BREAKER_TRIGGERED,
                    user='trading_loop',
                    action='emergency_halt',
                    resource='trading_loop',
                    status='FAILURE',
                    details={'reason': 'max_consecutive_errors'},
                )
                # Dispatch an alert for the emergency halt
                from alert_dispatch import dispatch_alert
                dispatch_alert({
                    'severity': 'CRITICAL',
                    'message': f'Trading loop halted: {MAX_CONSECUTIVE_ERRORS} consecutive errors',
                    'last_error': str(e),
                })
                break

        time_module.sleep(interval_seconds)

    # --- cleanup --------------------------------------------------------
    audit_logger.log(
        event_type=AuditEventType.SYSTEM_STOP,
        user='trading_loop',
        action='loop_stopped',
        resource='trading_loop',
        details={'shutdown_requested': shutdown_requested},
    )
    logger.info("Trading loop stopped")
