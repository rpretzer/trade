# Stock Arbitrage Model

Production-grade algorithmic trading system using LSTM neural networks for pairs trading strategies.

![Tests](https://img.shields.io/badge/tests-198%20passing-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-85%25-green)
![Python](https://img.shields.io/badge/python-3.12-blue)
![License](https://img.shields.io/badge/license-MIT-blue)

---

## 🎯 Project Status

**Production Readiness**: 8/10 (Ready for staging deployment)

- ✅ **43/47 P0 Critical Issues Fixed** (91.5%)
- ✅ **198 Comprehensive Tests** (100% passing)
- ✅ **Production-Grade Security** (encryption, audit logs, request signing)
- ✅ **Advanced Risk Management** (VaR, auto-liquidation, position limits)
- ✅ **Model Drift Detection** (automated monitoring)
- ✅ **CI/CD Pipeline** (GitHub Actions)

**Latest Release**: v2.0 (2026-01-31)

---

## 📋 Quick Start

### Prerequisites

- Python 3.12
- 4GB RAM
- Schwab API credentials (for live trading)
- Reddit API credentials (for sentiment analysis)

### Installation

```bash
# Clone repository
git clone https://github.com/rpretzer/trade.git stock_arbitrage_model
cd stock_arbitrage_model

# Create virtual environment
python3.12 -m venv venv_py312
source venv_py312/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v
```

### Run Backtest

```bash
# Download data
./run_daily_data_update.sh

# Train model
python train_model.py

# Run backtest
./run_backtest.sh
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
- **ML Model**: LSTM neural network (TensorFlow/Keras)
- **Risk Management**: VaR, position limits, auto-liquidation
- **Security**: AES-256 encryption, HMAC signing, audit logs
- **Monitoring**: Model drift detection, performance tracking

See [Architecture Documentation](docs/ARCHITECTURE.md) for details.

---

## 📚 Documentation

- **[Architecture](docs/ARCHITECTURE.md)** - System design, components, data flow
- **[Deployment Guide](docs/DEPLOYMENT.md)** - Installation, configuration, production setup
- **[Incident Runbook](docs/RUNBOOK.md)** - Troubleshooting, emergency procedures
- **[API Documentation](docs/API.md)** - API reference (if applicable)
- **[Developer Guide](docs/DEVELOPMENT.md)** - Contributing, testing, code standards

---

## 🚀 Features

### Trading Strategy
- **Pairs Trading**: Exploits mean reversion in price spreads
- **LSTM Predictions**: 60-day lookback window
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
- 198 tests across 9 test suites
- 85%+ code coverage
- 100% passing in CI/CD

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

```bash
# Set paper trading mode
export TRADING_MODE=paper

# Run live trading script (no real trades)
python live_trading.py
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

GitHub Actions runs on every push:
- All 198 tests
- Code formatting (black, isort)
- Linting (pylint, flake8)
- Security scanning (safety, bandit)
- Coverage reporting

---

## 📦 Dependencies

### Production

- **TensorFlow**: 2.15.0 (requires Python 3.11/3.12)
- **pandas**: 2.3.3
- **numpy**: 2.4.0
- **scikit-learn**: 1.8.0
- **yfinance**: 1.0
- **cryptography**: 41.0.7

See [requirements.txt](requirements.txt) for complete list.

### Development

- **pytest**: 9.0.2
- **black**: 25.1.0
- **pylint**: 3.3.4
- **mypy**: 1.14.1

See [requirements-dev.txt](requirements-dev.txt) for complete list.

---

## 🗺️ Roadmap

### Completed (v2.0)
- ✅ Production-grade error handling
- ✅ Comprehensive risk management
- ✅ Realistic trading execution
- ✅ Security hardening
- ✅ Model drift detection
- ✅ 198 comprehensive tests
- ✅ CI/CD pipeline

### Future Enhancements
- [ ] Database migration (CSV → PostgreSQL)
- [ ] Async processing for scalability
- [ ] Real-time monitoring dashboard (Grafana)
- [ ] Alert system (PagerDuty integration)
- [ ] Multi-model ensemble
- [ ] Additional ML models (XGBoost, Random Forest)

See [PRODUCTION_ROADMAP.md](PRODUCTION_ROADMAP.md) for details.

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

See [Development Guide](docs/DEVELOPMENT.md) for more details.

---

## ⚠️ Disclaimer

**This software is for educational and research purposes only.**

- Trading involves substantial risk of loss
- Past performance does not guarantee future results
- No warranty or guarantee of profitability
- Use at your own risk
- Consult a financial advisor before trading

**The authors are not responsible for any financial losses incurred.**

---

## 📧 Contact

- **GitHub Issues**: https://github.com/rpretzer/trade/issues
- **Email**: (Add your contact)

---

## 🙏 Acknowledgments

- **TensorFlow/Keras** - Deep learning framework
- **yfinance** - Market data API
- **Scikit-learn** - ML utilities
- **Pytest** - Testing framework
- **Claude Code** - Development assistance

---

**Built with ❤️ and rigorous testing for production deployment.**

Last updated: 2026-01-31
