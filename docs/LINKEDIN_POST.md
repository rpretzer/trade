# LinkedIn Post

## Short version (~300 words)

I built a production-grade algorithmic trading system from scratch -- and I'm open-sourcing it.

The Stock Arbitrage Model uses LSTM neural networks and XGBoost ensembles to execute pairs trading strategies. It started as a basic ML experiment and evolved into a fully engineered system with 287 tests, CI/CD, and real risk management.

Here's what's under the hood:

**ML Pipeline**
- LSTM + XGBoost ensemble with configurable blend weights
- Model registry with A/B testing and canary deployments
- Walk-forward validation and model drift detection

**Risk Management**
- Kelly Criterion position sizing with VaR and CVaR metrics
- Auto-liquidation at configurable drawdown thresholds
- Pre-trade risk gates on every order path

**Execution Simulation**
- Market impact modeling, partial fills, and order rejections
- Full transaction cost model (commissions, SEC fees, slippage, short borrow costs)
- Regime-scenario backtests against COVID crash, bull, and bear markets

**Security & Operations**
- AES-256 encrypted credentials, HMAC-SHA256 request signing
- Hash-chained tamper-proof audit logs
- Webhook alert dispatch for circuit breaker events
- Continuous trading loop with graceful shutdown

**Engineering**
- 287 tests across 15 suites, zero warnings
- GitHub Actions CI/CD (test, lint, security scan)
- Layered dependency profiles (core, LSTM, live trading, advanced analysis)
- Interactive CLI with guided first-run setup

Paper trading is on by default. The system is designed so you can study it, backtest it, and learn from the architecture without risking capital.

This project taught me as much about software engineering as it did about quantitative finance: how to wire risk checks into every trade path, how to make models auditable, and how to build systems that fail safely.

Code: https://github.com/rpretzer/trade

#AlgorithmicTrading #MachineLearning #Python #QuantFinance #OpenSource #SoftwareEngineering

---

## Condensed version (~150 words)

I open-sourced a production-grade algorithmic trading system built with LSTM neural networks and XGBoost ensembles.

What started as an ML experiment became a fully engineered platform: 287 tests, AES-256 encrypted credentials, hash-chained audit logs, Kelly Criterion sizing, auto-liquidation, and a complete transaction cost model with market impact simulation.

The system runs pairs trading strategies with a model registry, A/B testing, drift detection, and webhook alerting. Every trade path goes through pre-trade risk checks. Backtests run against regime scenarios including the COVID crash.

Paper trading is on by default -- study the architecture, run backtests, learn how production trading systems are built.

This project taught me as much about building safe, auditable systems as it did about quantitative finance.

Code: https://github.com/rpretzer/trade

#AlgorithmicTrading #MachineLearning #Python #QuantFinance #OpenSource
