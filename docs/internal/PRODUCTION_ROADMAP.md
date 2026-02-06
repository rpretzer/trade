# Production Roadmap — Updated 2026-02-06

Solo-developer scope.  Organised by priority tier, not calendar weeks.
Each tier can only start once the tier above it is fully green.

---

## What's Already Done (no action needed)

These items from the original roadmap are implemented and verified:

| Area | Status | Notes |
|---|---|---|
| Credential encryption (AES-256, random salt) | Done | `credential_manager.py` |
| Plaintext credential fallback removed | Done | Hard-fails on missing `cryptography` |
| API request signing (HMAC-SHA256) | Done | `api_security.py` |
| Replay-attack prevention (nonce + timestamp) | Done | 5-min window |
| Audit logging (hash-chained) | Done | `audit_logging.py` |
| Drift detection + alert routing | Done | Alerts → `logs/alerts.log` via `logging_config.log_drift_alert` |
| Structured JSON logging with rotation | Done | `logging_config.py` |
| Circuit breakers + retry logic | Done | `error_handling.py` |
| Dead man's switch | Done | `live_trading.py` (design-level) |
| Realistic execution simulation | Done | Market impact, partial fills, SEC fees, borrow costs — `trading_execution.py` |
| Transaction cost model | Done | `transaction_costs.py` |
| Kelly position sizing | Done | `risk_management.py` |
| Risk metrics (VaR, CVaR, Sharpe, Sortino, max drawdown) | Done | `risk_management.py` |
| Auto-liquidation at 10 % drawdown | Done | `risk_management.py` |
| Model versioning + integrity checks (SHA-256) | Done | `model_management.py` |
| A/B testing + canary deployment logic | Done | `model_management.py` (see Tier 2) |
| Walk-forward validation | Done | `model_management.py` |
| Model drift detection (performance, feature, prediction) | Done | `model_drift_detection.py` |
| Forced buy-in handling | Done | `trading_execution.py` |
| OrderExecutor call-site bug fixed | Done | `backtest_strategy.py` — enum + kwarg corrections |
| TensorFlow mock in conftest | Done | `tests/conftest.py` |
| 256 unit tests, CI/CD (GitHub Actions) | Done | 3 jobs: test, lint, security |
| Layered requirements + install script | Done | `requirements-{lstm,live,advanced,full}.txt`, `install.sh` |
| Interactive dependency installer | Done | `main.py check_dependencies()` |
| All user-facing docs audited and corrected | Done | README, ARCHITECTURE, DEPLOYMENT |

---

## Tier 1 — Safety-Critical (must finish before any live capital)

These three items are the largest gaps between "code exists" and "code runs in
the actual trading path."  None of them require new algorithms — they wire
existing, tested code into the execution flow.

### 1a. Wire `RiskManager.pre_trade_check()` into every trade path

**What's broken**: `RiskManager.pre_trade_check()` is fully implemented and
tested in `risk_management.py`, but nothing calls it.  Both `backtest_strategy.py`
and the live trading path execute orders without a risk gate.

**Work**:
1. In `backtest_strategy.py`, instantiate a `RiskManager` at the top of the
   backtest loop (or reuse one across iterations).  Call `pre_trade_check()`
   before each `execute_order`.  If it raises, skip the trade and log the
   rejection.
2. The live trading path (Tier 1b below) must do the same.
3. Add two tests: one that confirms a trade is blocked when position limits
   are exceeded, one that confirms the trade goes through when limits are
   respected.

**Acceptance**: `pytest` passes, and a backtest run on 6+ months of data shows
risk-violation rejections appearing in the log.

---

### 1b. Build the continuous live-trading loop

**What's broken**: `live_trading.py` is a signal-only stub (139 lines).
It loads a model, runs inference, and returns a string.  There is no loop, no
order execution, no broker connection, no position tracking across iterations.
`live_trade()` in `main.py` does credential checks but does not start a loop.

**Work** (in `live_trading.py`):
1. Add a `run_trading_loop(paper_trading=True)` function with this structure:
   ```
   while True:
       if not market_is_open():  →  sleep until open, continue
       data   = fetch_and_process_latest_data()
       signal = get_latest_signal(data)           # already exists
       if signal is None:  →  sleep(interval), continue
       risk_manager.pre_trade_check(signal)       # Tier 1a
       result = trading_api.execute_signal(signal)
       audit_logger.log(result)
       sleep(interval)                            # default 300 s
   ```
2. Wire `main.py live` to call `run_trading_loop(paper_trading=True)` (or
   `False` after credential setup).
3. Add market-hours awareness (use `pandas_market_calendars` if the advanced
   profile is installed; otherwise hard-code NYSE 9:30–16:00 ET Mon–Fri).
4. Persist position state to a local JSON file so the loop survives restarts
   without losing track of open positions.

**Acceptance**: `python main.py live` starts, logs predictions every 5 min
during simulated market hours, exits cleanly on Ctrl-C, and persists positions.

---

### 1c. Wire external alert delivery for HIGH / CRITICAL events

**What's broken**: Drift alerts are written to `logs/alerts.log` (done).
`_send_emergency_alerts()` in the live path logs "not yet implemented".
Nothing pages a human when the system halts or a circuit breaker fires.

**Work**:
1. Add a lightweight alert-dispatch module (`alert_dispatch.py`) that accepts
   an alert dict and routes it to one or more sinks.  Start with two sinks:
   - **Webhook** (POST JSON to a caller-supplied URL — works with Slack,
     Discord, PagerDuty, or any HTTP receiver).
   - **Fallback file** (always write to `logs/alerts.log` — already happens).
2. Make the webhook URL configurable via an environment variable
   (`ALERT_WEBHOOK_URL`).  If unset, HIGH/CRITICAL alerts print a loud
   warning to stderr in addition to the file write; the system keeps running.
3. Replace the "not yet implemented" stub in the live path with a call to
   the new dispatcher.
4. Route circuit-breaker events and drift alerts through the same dispatcher.

**Acceptance**: Setting `ALERT_WEBHOOK_URL` and triggering a test alert
results in an HTTP POST to the configured endpoint.  Unsetting the variable
produces a visible stderr warning but does not crash.

---

## Tier 2 — Connect existing scaffolding to the pipeline

Code already written, never imported by anything that runs end-to-end.

### 2a. Trigger model_management A/B routing from the training flow

**What's broken**: `model_management.py` has full A/B and canary logic.
Nothing in `train_model.py` or the live path registers a new model version or
starts an A/B comparison.

**Work**:
1. At the end of `train_model.py`, after the model is saved, call
   `ModelManager.register_model(path, metadata)`.
2. In the live trading loop (Tier 1b), load the model through
   `ModelManager.get_active_model()` instead of a hard-coded path.  This
   makes A/B and canary routing automatic once a second model is registered.
3. Add a CLI command (`python main.py models`) that lists registered versions
   and their status.

### 2b. Wire the database layer into the data pipeline (optional, later)

`database/` contains a complete ORM (SQLAlchemy models, migrations, queries)
but is never imported.  This is a larger undertaking and is deprioritised
because the CSV path works correctly for single-user, single-machine use.
Revisit if:
- Concurrent access becomes a problem, or
- Data volume exceeds ~1 M rows.

Action for now: none.  Leave the scaffolding in place.

---

## Tier 3 — Enhancements (after Tier 1 and 2 are green)

### 3a. Multi-model ensemble

LSTM and XGBoost are trained and stored independently.  Neither is combined
into an ensemble prediction.

**Work**:
1. Add an `ensemble_predict(lstm_pred, xgb_pred, weights)` helper.
2. Default weights: 0.6 LSTM / 0.4 XGBoost (tune via backtest).
3. Wire into the live loop's prediction step.
4. Gate behind a feature flag so it can be disabled instantly.

### 3b. Backtest on regime scenarios

The original roadmap called for testing against COVID crash, 2021 bull,
2022 bear.  The execution engine and transaction costs are in place; this is
purely a test-data + assertion exercise.

**Work**:
1. Add a `tests/test_regime_scenarios.py` with three parameterised tests.
2. Each loads a small CSV slice of the relevant period, runs the backtest,
   and asserts expected behaviour (e.g., circuit breaker fires during the
   crash slice; positive return during the bull slice).

### 3c. Streamlit dashboard polish — DONE

Ticker columns derived from `constants.py` (no more hardcoded `aapl_price`/
`msft_price`).  Three new panels added: Alert Log (reads `logs/alerts.log`
JSONL), Model Registry (imports `ModelRegistry`), Live Trading Status (reads
`positions.json`).  All degrade gracefully when data is missing.

### 3d. Ticker parameterisation

**What's broken**: AAPL and MSFT are hard-coded in at least seven layers of
the stack.  Swapping to a different pair (GOOG/AMZN, etc.) requires editing
literal strings in half a dozen files *and* retraining the model.

**Affected files / layers**:
- `constants.py` — no ticker constants exist yet
- `process_stock_data.py` — downloads and engineers features for exactly two symbols
- Feature column names — 23 names like `AAPL_normalized`, `MSFT_RSI` are
  listed literally in `live_trading.py`, `backtest_strategy.py`, and
  `train_model.py`
- Signal strings — `"Buy AAPL, Sell MSFT"` parsed as literal text in the
  backtest loop, the live loop, and signal generation
- `backtest_strategy.py` — `price_data['AAPL']`, position tracking
- `live_trading.py` — `get_current_price('AAPL')`, leg determination
- `trading_api.py` — price sanity-check ranges hard-coded per symbol

**Work** (code only — no algorithm changes):
1. Add `TICKER_LONG` and `TICKER_SHORT` to `constants.py`.
2. In `process_stock_data.py`, read tickers from constants instead of
   hard-coding download calls.
3. Write a `feature_names(ticker_long, ticker_short)` helper that
   generates the 23 column names programmatically.  Replace every literal
   list of feature columns with a call to it.
4. Replace literal signal strings with an `f"Buy {TICKER_LONG}, Sell
   {TICKER_SHORT}"` pattern everywhere they appear.
5. Replace `price_data['AAPL']` / `price_data['MSFT']` with the constants.
6. Remove the per-symbol price sanity ranges in `trading_api.py` (or make
   them a dict keyed by symbol in constants).

**Operator step** (after the code change):
Re-run `process_stock_data.py` → `train_model.py` for any new pair before
the backtest or live loop will produce meaningful signals.  The code change
alone does not break anything for the current AAPL/MSFT pair.

---

## What the original roadmap called for that we are deliberately skipping

| Item | Why skip |
|---|---|
| AWS / cloud infrastructure ($1.5 K/mo) | Single-user, cron-based deployment works. Revisit at > 100 stocks or multi-user. |
| Terraform / Docker / Kubernetes | Same reason. |
| ELK / Loki centralised logging | Structured JSON logs + rotation already in place. File-based is sufficient at this scale. |
| PagerDuty / Opsgenie on-call rotation | Solo operator. Webhook alert (Tier 1c) covers this. |
| MLflow model registry | `model_management.py` is the registry. Revisit if ML ops complexity grows. |
| Dynaconf / Hydra config | Constants + env vars are sufficient. |
| Security penetration test ($5 K) | Not warranted at current scale / capital. |
| Multi-engineer hiring ($860 K/yr) | Solo scope. |

---

## Summary — current status and what's next

**Tier 1 — DONE.**  `pre_trade_check` wired into backtest + live paths,
continuous trading loop built, webhook alert dispatch live.

**Tier 2 — DONE.**  `train_model.py` registers both LSTM and XGBoost in
`ModelRegistry` on every training run.  `live_trading.py` resolves the active
model through `ABTestController.select_model()` (falls back to the explicit
path when the registry is empty).  `python main.py models` lists all
registered versions with colour-coded status.

**Tier 3 — DONE.**
- 3a: Ensemble prediction (LSTM + XGBoost, gated by `ENSEMBLE_MODE` env var).
  13 tests.
- 3b: Regime-scenario backtests — COVID crash, 2021 bull, 2022 bear.  10 tests
  exercising circuit breakers, positive/negative returns, and drawdown bounds.
- 3d: Full ticker parameterisation.  `constants.py` is now single source of truth
  for tickers, signal strings, feature-column names, and price sanity ranges.
  `trading_api.execute_arbitrage_trade` collapsed from two duplicated branches
  into one parameterised block.
- 3c: Dashboard polish — ticker columns derived from constants, three new panels
  (Alert Log, Model Registry, Live Trading Status) with graceful degradation.

**Additional polish (2026-02-06):**
- print() → logging migration in 5 production-path files (options_data,
  trading_api, data_validation, sentiment_analysis, credential_manager).
- `.env.example` added for distributability.

**287 tests, all green.**

**Before flipping `paper_trading=False`:** run a paper-trading marathon
(1+ week) to validate end-to-end behaviour with real market data.
