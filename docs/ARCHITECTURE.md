# Stock Arbitrage Model - Architecture Documentation

## System Overview

The Stock Arbitrage Model is a production-grade algorithmic trading system that uses machine learning (LSTM neural networks) to predict price differences between correlated stocks and execute pairs trading strategies.

**Version**: 2.2
**Status**: Staging-validated.  Paper-trade before committing real capital.
**Last Updated**: 2026-02-06

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     DATA COLLECTION                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ yfinance │  │  Reddit  │  │ Options  │  │  Schwab  │   │
│  │   API    │  │   API    │  │   Data   │  │   API    │   │
│  └─────┬────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘   │
└────────┼────────────┼─────────────┼─────────────┼──────────┘
         │            │             │             │
         ▼            ▼             ▼             ▼
┌─────────────────────────────────────────────────────────────┐
│                  DATA PROCESSING                             │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ process_stock_data.py                                   │ │
│  │ • Download price data                                   │ │
│  │ • Calculate technical indicators (MA, RSI, MACD)        │ │
│  │ • Normalize features                                    │ │
│  │ • Merge sentiment & options data                        │ │
│  └───────────────────────┬────────────────────────────────┘ │
└────────────────────────────┼───────────────────────────────┘
                             ▼
                    processed_stock_data.csv
                             │
         ┌───────────────────┼───────────────────┐
         ▼                   ▼                   ▼
┌──────────────────┐  ┌──────────────┐  ┌──────────────────┐
│  MODEL TRAINING  │  │  BACKTESTING │  │  LIVE TRADING    │
│                  │  │              │  │                  │
│ train_model.py   │  │ backtest_    │  │ live_trading.py  │
│ • LSTM           │  │ strategy.py  │  │ • Real-time      │
│ • Walk-forward   │  │ • Risk mgmt  │  │   predictions    │
│ • Validation     │  │ • Realistic  │  │ • Order exec     │
│ • A/B testing    │  │   execution  │  │ • Risk checks    │
│                  │  │              │  │ • Auto-trading   │
└────────┬─────────┘  └──────┬───────┘  └────────┬─────────┘
         │                   │                   │
         ▼                   ▼                   ▼
    model files         backtest results    trade execution
         │                                       │
         └───────────────────┬───────────────────┘
                             ▼
                    ┌─────────────────┐
                    │  MONITORING &   │
                    │    AUDITING     │
                    │                 │
                    │ • Audit logs    │
                    │ • Drift detect  │
                    │ • Risk metrics  │
                    │ • Performance   │
                    └─────────────────┘
```

---

## Core Components

### 1. Data Collection Layer

**Purpose**: Gather market data, sentiment, and options information

**Components**:
- `yfinance`: Historical price and volume data
- `praw` (Reddit API): Sentiment analysis from r/wallstreetbets
- `options_data.py`: Options volume and implied volatility
- `trading_api.py`: Schwab API integration for live trading

**Data Flow**:
1. Daily cron job runs `run_daily_data_update.sh`
2. Downloads latest data from APIs
3. Stores raw data temporarily
4. Processes and merges into `processed_stock_data.csv`

### 2. Data Processing Pipeline

**Component**: `process_stock_data.py`

**Functions**:
- Download price data for selected stocks
- Calculate technical indicators:
  - Moving averages (MA5, MA20)
  - RSI (14-period)
  - MACD (12, 26, 9)
  - Volume indicators
- Normalize features (0-1 scaling)
- Calculate price differences (target variable)
- Merge sentiment scores
- Merge options data
- Data validation and cleaning

**Output**: `processed_stock_data.csv` with ~20-25 features per stock

### 3. Machine Learning Layer

#### Model Training (`train_model.py`)

**Primary Model**: LSTM Neural Network
- Layer 1: LSTM (50 units, dropout 0.2)
- Layer 2: LSTM (30 units, dropout 0.2)
- Layer 3: Dense (1 unit, linear activation)

**Secondary Model**: XGBoost (gradient boosting)
- Used in ensemble mode (LSTM 60% + XGBoost 40% by default)
- Gated by `ENSEMBLE_MODE` environment variable

**Training Process**:
1. Load processed data
2. Create sequences (60 timesteps)
3. Split: 80% train, 20% test (temporal order preserved)
4. Train with early stopping
5. Evaluate on test set
6. Save model to `lstm_price_difference_model.h5`

#### Model Management (`model_management.py`)

**Features**:
- **Versioning**: Track model versions with metadata
- **A/B Testing**: Compare model performance
- **Canary Deployments**: Gradual rollout (10% traffic)
- **Walk-Forward Validation**: Prevent data leakage
- **Integrity Checks**: SHA-256 checksums

#### Drift Detection (`model_drift_detection.py`)

**Monitoring**:
- **Performance**: Track MSE, MAE, directional accuracy
- **Features**: Statistical tests for distribution shifts
- **Predictions**: Detect bias in outputs
- **Alerts**: Severity-based warnings (LOW → CRITICAL)

### 4. Backtesting Engine

**Component**: `backtest_strategy.py`

**Features**:
- Realistic market simulation:
  - Market hours enforcement
  - Partial fills
  - Order rejections
  - Market impact (√volume fraction)
  - Transaction costs (commissions, SEC fees, slippage)
  - Short borrow costs
  - Margin interest
- Risk management:
  - 5% maximum drawdown (auto-liquidate at 10%)
  - 2% stop-loss per trade
  - Position size limits
- Performance metrics:
  - Total return, Sharpe ratio, max drawdown
  - Win rate, average profit per trade
  - Transaction cost analysis

### 5. Live Trading System

**Component**: `live_trading.py`

**Workflow**:
1. Fetch latest data
2. Prepare features (last 60 days)
3. Load production model
4. Generate prediction
5. Check trading signal (threshold 0.5)
6. Risk pre-trade checks
7. Execute trades via `trading_api.py`
8. Log to audit trail

**Safety Features**:
- Circuit breakers (5 failures → halt)
- Rate limiting (2 req/sec, burst 10)
- Dead man's switch
- Forced buy-in handling
- Drift detection

### 6. Risk Management System

**Component**: `risk_management.py`

**Risk Limits**:
- Max position size: 1000 shares
- Max position value: $50,000
- Max single position: 15% of portfolio
- Max sector exposure: 40%
- Max drawdown: 5% (critical: 10%)

**Risk Metrics**:
- Value at Risk (VaR) - 95% confidence
- Conditional VaR (CVaR)
- Sharpe ratio
- Sortino ratio
- Maximum drawdown

**Auto-Liquidation**:
- Triggered at 10% drawdown
- Closes all positions immediately
- Prevents catastrophic losses

### 7. Security Infrastructure

**Components**:
- `credential_manager.py`: Encrypted credential storage
- `audit_logging.py`: Tamper-proof audit trail
- `api_security.py`: Request signing and verification
- `error_handling.py`: Circuit breakers and retry logic

**Security Features**:
- AES-256 encryption for credentials
- Random salt per user
- Windows ACL / Unix chmod 600 permissions
- HMAC-SHA256 request signing
- Hash-chained audit logs (blockchain-like)
- Replay attack prevention (nonce + timestamp)

---

## Data Flow

### Training Flow

```
1. Run: python process_stock_data.py
   → Downloads data from yfinance
   → Calculates indicators
   → Outputs: processed_stock_data.csv

2. Run: python train_model.py
   → Loads processed_stock_data.csv
   → Creates sequences
   → Trains LSTM model
   → Outputs: lstm_price_difference_model.h5

3. Run: python backtest_strategy.py
   → Loads model + data
   → Simulates trading
   → Outputs: backtest_results.csv
```

### Live Trading Flow

```
1. Cron: Daily at market open
   → run_daily_data_update.sh
   → Updates processed_stock_data.csv

2. Cron: Every 5 minutes during market hours
   → live_trading.py
   → Fetches latest data
   → Generates prediction
   → Checks signal threshold
   → Risk checks
   → Executes trade (if signal)
   → Logs to audit trail
```

---

## Technology Stack

### Core
- **Python**: 3.12 (TensorFlow compatibility)
- **TensorFlow/Keras**: 2.15 (LSTM models)
- **Pandas**: 2.3.3 (data manipulation)
- **NumPy**: 2.4.0 (numerical computing)

### Machine Learning
- **scikit-learn**: 1.8.0 (preprocessing, metrics)
- **XGBoost**: 3.1.3 (alternative model)
- **SciPy**: 1.17.0 (statistical tests)

### APIs
- **yfinance**: 1.0 (market data)
- **praw**: 7.8.1 (Reddit sentiment)
- **schwab-py**: (live trading)
- **requests**: 2.32.5 (HTTP)

### Security
- **cryptography**: 41.0.7 (encryption)

### Development
- **pytest**: 9.0.2 (testing - 287 tests)
- **black**: 25.1.0 (formatting)
- **pylint**: 3.3.4 (linting)

---

## File Structure

```
stock_arbitrage_model/
├── Core Trading
│   ├── live_trading.py          # Live trading execution
│   ├── backtest_strategy.py     # Strategy backtesting
│   ├── train_model.py            # Model training
│   └── process_stock_data.py     # Data pipeline
│
├── Infrastructure
│   ├── exceptions.py             # Custom exceptions
│   ├── error_handling.py         # Retry, circuit breakers
│   ├── risk_management.py        # Risk limits & metrics
│   ├── trading_execution.py      # Realistic order execution
│   ├── transaction_costs.py      # Cost modeling
│   ├── audit_logging.py          # Tamper-proof logging
│   ├── model_management.py       # Model versioning & A/B
│   ├── model_drift_detection.py  # Drift monitoring
│   ├── api_security.py           # Request signing
│   └── credential_manager.py     # Encrypted credentials
│
├── Utilities
│   ├── ml_utils.py               # ML helper functions
│   ├── constants.py              # Tickers, signals, feature names, config
│   ├── sentiment_analysis.py     # Reddit sentiment
│   ├── options_data.py           # Options data fetching
│   ├── alert_dispatch.py         # Webhook alert routing
│   ├── storage_config.py         # Storage configuration
│   └── emergency_shutdown.py     # Safety shutdown procedures
│
├── Configuration
│   ├── requirements.txt          # Core dependencies
│   ├── requirements-lstm.txt     # + TensorFlow (LSTM training)
│   ├── requirements-live.txt     # + schwab-py (live trading)
│   ├── requirements-advanced.txt # + statsmodels, market calendars
│   ├── requirements-full.txt     # All optional profiles
│   ├── requirements-dev.txt      # Dev / test tools
│   ├── install.sh                # One-command bootstrap
│   ├── pytest.ini                # Test configuration
│   ├── .pylintrc                 # Linting rules
│   ├── .flake8                   # Code style
│   └── pyproject.toml            # Tool configuration
│
├── Testing
│   └── tests/                    # 287 unit tests
│       ├── conftest.py                  # shared mocks (TensorFlow)
│       ├── test_alert_dispatch.py
│       ├── test_api_security.py
│       ├── test_backtest_integration.py
│       ├── test_backtest_strategy.py
│       ├── test_data_validation.py
│       ├── test_ensemble.py
│       ├── test_error_handling.py
│       ├── test_forced_buyin.py
│       ├── test_model_drift.py
│       ├── test_model_management.py
│       ├── test_performance_profiling.py
│       ├── test_regime_scenarios.py
│       ├── test_risk_management.py
│       ├── test_security.py
│       └── test_trading_execution.py
│
├── Deployment
│   ├── run_daily_data_update.sh  # Cron job for data
│   └── run_backtest.sh           # Run backtest script
│
└── Data (generated)
    ├── processed_stock_data.csv  # Processed features
    ├── lstm_price_difference_model.h5  # Trained model
    ├── backtest_results.csv      # Backtest output
    └── logs/                     # Audit & trading logs
```

---

## Performance Characteristics

### Throughput
- **Data Processing**: ~1000 rows/second
- **Model Inference**: ~100 predictions/second
- **Backtesting**: ~10,000 samples in 30 seconds

### Latency
- **Prediction**: <100ms (LSTM forward pass)
- **Trade Execution**: 500-2000ms (API round-trip)
- **Risk Checks**: <10ms

### Scalability
- **Current**: Single-threaded, handles 10-20 stocks
- **Bottleneck**: CSV I/O, sequential processing
- **Max Scale**: ~100 stocks with current architecture
- **Future**: Database + async processing for 1000s of stocks

---

## Security Model

### Authentication
- API keys encrypted with AES-256
- Random salt per installation
- File permissions: 600 (owner read/write only)

### Authorization
- No multi-user support (single user system)
- Future: Role-based access control (RBAC)

### Audit Trail
- All trades logged with hash chaining
- Tamper detection via SHA-256
- Immutable append-only logs

### Network Security
- HMAC-SHA256 request signing
- Replay attack prevention (5-minute window)
- TLS for all API communications

---

## Monitoring & Observability

### Current State
- ✅ Audit logging (all trades, predictions)
- ✅ Model drift detection with alert routing to `logs/alerts.log`
- ✅ Webhook alert dispatch (`alert_dispatch.py`) for HIGH/CRITICAL events
- ✅ Risk metrics tracking
- ✅ Structured JSON logging with rotation (`logging_config.py`)
- ✅ Circuit breaker (5 consecutive errors → halt + alert)
- ❌ No metrics collection (Prometheus)
- ❌ No distributed tracing

### Future Enhancements
- Prometheus metrics export
- Grafana dashboards
- Application Performance Monitoring (APM)

---

## Known Limitations

1. **CSV Storage**: No ACID guarantees, concurrent access unsafe
2. **Single-Threaded**: Cannot process multiple stocks in parallel
3. **No Caching**: Re-downloads data frequently
4. **Manual Deployment**: No automated deployment pipeline
5. **Limited Scalability**: ~100 stocks maximum

See `PRODUCTION_ROADMAP.md` for improvement plans.

---

## Dependencies

See `requirements.txt` for exact versions.

**Critical Dependencies**:
- TensorFlow requires Python 3.11 or 3.12 (not 3.14)
- schwab-py requires Python 3.10+
- pandas-market-calendars for market hours

---

## Version History

### v2.2 (2026-02-06) - Full Pipeline Wiring
- Multi-model ensemble (LSTM + XGBoost)
- Model registry with A/B testing wired into training and inference
- Continuous live-trading loop with graceful shutdown
- Webhook alert dispatch for HIGH/CRITICAL events
- Ticker parameterisation (single source of truth)
- Regime-scenario backtests (COVID crash, bull, bear)
- 287 comprehensive tests across 15 suites

### v2.0 (2026-01-31) - Production Hardening
- 43/47 P0 issues fixed (91.5%)
- 198 comprehensive tests
- Production-grade security, risk management, error handling
- CI/CD pipeline with GitHub Actions
- Model drift detection
- API request signing

### v1.0 (2024-01-15) - Initial Implementation
- Basic LSTM model
- Simple backtesting
- No tests, minimal error handling

---

## Related Documentation

- [Deployment Guide](DEPLOYMENT.md)
- [Incident Runbook](RUNBOOK.md)
- [Performance Profiling](PERFORMANCE_PROFILING.md)
- [Disclaimer](../DISCLAIMER.md)
