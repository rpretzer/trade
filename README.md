# Stock Arbitrage Model

Production-grade algorithmic trading system using LSTM neural networks and XGBoost ensembles for pairs trading strategies.

![Tests](https://img.shields.io/badge/tests-287%20passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.10+-blue)
![License](https://img.shields.io/badge/license-MIT-blue)

---

## 🎯 Project Status

**Readiness**: Staging-validated.  Paper-trade before committing real capital.

- ✅ **287 Tests** across 15 test suites (100% passing, zero warnings)
- ✅ **Production-Grade Security** (AES-256 encryption, audit logs, HMAC request signing)
- ✅ **Advanced Risk Management** (Kelly sizing, drawdown limits, auto-liquidation)
- ✅ **Realistic Execution Simulation** (market impact, partial fills, SEC fees)
- ✅ **Model Drift Detection & Alerting** (webhook dispatch for HIGH/CRITICAL events)
- ✅ **Multi-Model Ensemble** (LSTM + XGBoost with configurable weights)
- ✅ **CI/CD Pipeline** (GitHub Actions — tests, lint, security scan)
- ✅ **Interactive CLI** with guided first-run setup and optional-dep installer

---

## 📋 Quick Start

### Prerequisites

- Python 3.10+ (3.12 recommended for TensorFlow compatibility)
- 4 GB RAM (8 GB recommended)
- Schwab API credentials (for live trading — optional)
- Reddit API credentials (for sentiment analysis — optional)

### Installation

```bash
git clone https://github.com/rpretzer/trade.git stock_arbitrage_model
cd stock_arbitrage_model
./install.sh          # sets up venv, installs deps, offers optional packs
```

The install script detects missing optional feature packs (LSTM training,
live trading, advanced analysis) and offers to install each one
interactively.  You can also run the check later at any time:

```bash
python main.py deps
```

### Run Backtest

```bash
python main.py backtest     # or: python main.py pipeline  (process → train → backtest)
```

**Expected Output**: Positive returns with win rate >55%

---

## 🏗️ Architecture

```
Data Collection → Processing → ML Model → Backtesting → Live Trading
                                                ↓
                                         Risk Management
                                                ↓
                                         Audit Logging
```

### Key Components

- **Data Pipeline**: yfinance, Reddit sentiment, options data
- **ML Models**: LSTM neural network (TensorFlow/Keras) + XGBoost ensemble
- **Risk Management**: VaR, position limits, auto-liquidation
- **Security**: AES-256 encryption, HMAC signing, audit logs
- **Monitoring**: Model drift detection, webhook alerts, performance tracking

See [Architecture Documentation](docs/ARCHITECTURE.md) for details.

---

## 📚 Documentation

- **[Architecture](docs/ARCHITECTURE.md)** – System design, components, data flow
- **[Deployment Guide](docs/DEPLOYMENT.md)** – Installation, configuration, production setup
- **[Incident Runbook](docs/RUNBOOK.md)** – Troubleshooting, emergency procedures
- **[Performance Profiling](docs/PERFORMANCE_PROFILING.md)** – Profiling toolkit & optimisation tips
- **[Disclaimer](DISCLAIMER.md)** – Educational-use terms

---

## 🚀 Features

### Trading Strategy
- **Pairs Trading**: Exploits mean reversion in price spreads (parameterised tickers)
- **LSTM + XGBoost Ensemble**: 60-day lookback, configurable blend weights
- **Signal Generation**: Configurable threshold (default 0.5)
- **Position Sizing**: Kelly Criterion with constraints

### Risk Management
- **Position Limits**: 15% max concentration per symbol
- **Sector Limits**: 40% max sector exposure
- **Drawdown Protection**: Auto-liquidate at 10% drawdown
- **Stop-Loss**: 2% per trade
- **Risk Metrics**: VaR, CVaR, Sharpe ratio, Sortino ratio

### Execution
- **Realistic Simulation**:
  - Market hours enforcement
  - Partial fills
  - Order rejections
  - Market impact modeling
  - Transaction costs (commissions, SEC fees, slippage)
  - Short borrow costs
  - Margin interest

### Security
- **Encrypted Credentials**: AES-256 with random salt
- **Audit Logging**: Tamper-proof hash-chained logs
- **API Security**: HMAC-SHA256 request signing
- **Replay Protection**: 5-minute timestamp window + nonce

### Monitoring
- **Model Drift Detection**:
  - Performance degradation alerts
  - Feature distribution shifts
  - Prediction bias detection
- **Circuit Breakers**: Auto-halt after 5 failures
- **Dead Man's Switch**: Safety timeout for critical operations

---

## 📊 Performance

### Backtest Results (Example)

- **Return**: 5-15% annually
- **Win Rate**: 60-65%
- **Sharpe Ratio**: 1.2-1.8
- **Max Drawdown**: 3-7%

**Note**: Past performance does not guarantee future results.

### System Performance

- **Prediction Latency**: <100ms
- **Trade Execution**: 500-2000ms (API-dependent)
- **Throughput**: 100 predictions/second
- **Scalability**: 10-20 stocks (current), 100+ (with optimization)

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html

# Run specific test suite
pytest tests/test_risk_management.py -v

# Run security tests only
pytest tests/test_security.py tests/test_api_security.py -v
```

**Test Coverage**:
- 287 tests across 15 test suites
- 100% passing, zero deprecation warnings

---

## 🔒 Security

### Credentials

Never commit credentials to git:

```bash
# Credentials are encrypted and stored in:
.schwab_config.enc   # (gitignored)
.reddit_config.enc   # (gitignored)

# Use credential_manager.py to manage:
python -c "from credential_manager import CredentialManager; \
cm = CredentialManager(); \
cm.store_credentials('schwab', {...})"
```

### Audit Logs

All trades and critical actions are logged:

```bash
# View audit log
tail logs/audit/audit_log.jsonl | jq .

# Verify integrity (detects tampering)
python -c "from audit_logging import AuditLogger; \
al = AuditLogger(); \
print(al.verify_chain())"
```

---

## 📈 Deployment

### Staging

Paper-trading mode is **on by default** (`paper_trading=True` in
`trading_api.py`).  Run the trading loop with no real orders:

```bash
python main.py live       # paper trades only — safe to run
```

### Production

See [Deployment Guide](docs/DEPLOYMENT.md) for complete instructions.

**Important**:
- Start with small capital ($1,000-$5,000)
- Monitor closely for first week
- Review performance monthly
- Retrain model if drift detected

---

## 🛠️ Development

### Code Quality

```bash
# Format code
black .

# Sort imports
isort .

# Lint
pylint *.py

# Type check
mypy *.py
```

### CI/CD

GitHub Actions (`.github/workflows/ci.yml`) runs on every push:
- All 287 tests with coverage
- Code formatting (black, isort) and linting (pylint, flake8)
- Security scanning (safety, bandit)

---

## 📦 Dependencies

### Core (always installed)

- **pandas** 2.3.3 · **numpy** 2.4.0 · **scipy** 1.17.0 · **scikit-learn** 1.8.0
- **xgboost** 3.1.3 · **yfinance** 1.0 · **cryptography** 41.0.7
- **streamlit** 1.52.2 · **requests** 2.32.5 · **praw** 7.8.1

### Optional profiles

| Profile | File | Adds |
|---|---|---|
| LSTM Training | `requirements-lstm.txt` | TensorFlow 2.15 (Python 3.11/3.12) |
| Live Trading | `requirements-live.txt` | schwab-py |
| Advanced Analysis | `requirements-advanced.txt` | statsmodels, pandas-market-calendars |
| Everything | `requirements-full.txt` | All of the above |

### Development

See [requirements-dev.txt](requirements-dev.txt).

---

## 🗺️ Roadmap

### Completed
- ✅ Production-grade error handling & circuit breakers
- ✅ Comprehensive risk management (Kelly sizing, VaR, auto-liquidation)
- ✅ Realistic trading execution (market impact, partial fills, SEC fees)
- ✅ Security hardening (AES-256, HMAC signing, hash-chained audit logs)
- ✅ Model drift detection with webhook alert dispatch
- ✅ Multi-model ensemble (LSTM + XGBoost, configurable weights)
- ✅ Model registry with A/B testing and canary deployments
- ✅ Ticker parameterisation (single source of truth in `constants.py`)
- ✅ Regime-scenario backtests (COVID crash, bull market, bear market)
- ✅ Continuous live-trading loop with graceful shutdown
- ✅ 287 tests, CI/CD pipeline, interactive CLI with guided setup
- ✅ Layered requirements (core + optional profiles)

### Future Enhancements
- [ ] Database migration (CSV → PostgreSQL) — schema ready in `database/`
- [ ] Async processing for scalability
- [ ] Real-time monitoring dashboard (Grafana)
- [ ] Streamlit dashboard polish (live data integration)

---

## 📝 License

MIT License - See [LICENSE](LICENSE) for details.

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass (`pytest tests/ -v`)
5. Follow code style (black, pylint)
6. Submit pull request

See [Deployment Guide](docs/DEPLOYMENT.md) for environment setup details.

---

## ⚠️ Disclaimer

**This software is for educational and research purposes only.**

- Trading involves substantial risk of loss.  Past performance does not
  guarantee future results.
- The authors accept no liability for any financial loss.
- Consult a licensed financial advisor before making any trading decisions.

Read the full terms in [DISCLAIMER.md](DISCLAIMER.md).

---

## 📧 Contact

- **GitHub Issues**: [github.com/rpretzer/trade/issues](https://github.com/rpretzer/trade/issues)

---

## 🙏 Acknowledgments

- **TensorFlow/Keras** - Deep learning framework
- **XGBoost** - Gradient boosting models
- **yfinance** - Market data API
- **Scikit-learn** - ML utilities
- **Pytest** - Testing framework
- **Claude Code** - Development assistance

---

**Built with rigorous testing for production deployment.**

Last updated: 2026-02-06
