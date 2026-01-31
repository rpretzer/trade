# Incident Runbook

Quick reference guide for diagnosing and resolving common issues in production.

**Last Updated**: 2026-01-31

---

## 🚨 Emergency Procedures

### CRITICAL: System Losing Money

**Symptoms**: Large drawdown, multiple losing trades, unexpected behavior

**Immediate Actions**:

1. **STOP ALL TRADING** (60 seconds):
   ```bash
   # Stop cron jobs
   crontab -r

   # Or kill running processes
   pkill -f live_trading.py
   ```

2. **Close All Open Positions** (via Schwab web interface):
   - Log into Schwab account
   - Navigate to Positions
   - Manually close all positions at market price

3. **Document the Incident**:
   ```bash
   # Capture current state
   date >> incident_$(date +%Y%m%d_%H%M%S).log
   tail -100 logs/trading_*.log >> incident_$(date +%Y%m%d_%H%M%S).log
   tail -100 logs/audit/audit_log.jsonl >> incident_$(date +%Y%m%d_%H%M%S).log
   ```

4. **Investigate**:
   - Check audit logs for unauthorized trades
   - Review recent predictions
   - Check for model drift alerts
   - Verify data quality

5. **Root Cause Analysis**:
   - Was drawdown limit hit? (Check logs for "MAX DRAWDOWN REACHED")
   - Are predictions unrealistic? (Model drift?)
   - Bad data? (Check processed_stock_data.csv)
   - API errors? (Check for error patterns)

**Do NOT Resume Trading Until**:
- Root cause identified and fixed
- Backtest shows positive results
- Paper trading validated for 1+ week

---

## Common Issues

### Issue: No Trades Being Executed

**Symptoms**: System runs but no trades appear in logs

**Diagnosis**:

```bash
# Check if trading mode is paper
echo $TRADING_MODE

# Check recent predictions
tail -50 logs/trading_$(date +%Y%m%d).log | grep "prediction"

# Check signal threshold
grep "signal" logs/trading_$(date +%Y%m%d).log
```

**Common Causes**:

1. **Paper trading mode enabled**:
   ```bash
   # Solution
   unset TRADING_MODE
   # Or
   export TRADING_MODE=live
   ```

2. **Trading threshold too high** (e.g., 0.9):
   - Predictions rarely exceed threshold
   - Solution: Lower threshold in constants.py to 0.5-0.7

3. **Drawdown limit reached**:
   ```bash
   # Check logs
   grep "MAX DRAWDOWN" logs/*.log

   # Solution: Reset after reviewing performance
   # Or adjust risk limits if appropriate
   ```

4. **Market closed**:
   ```bash
   # Check market hours
   python -c "from trading_execution import OrderExecutor; \
   from datetime import datetime; \
   oe = OrderExecutor(); \
   print(f'Market open: {oe.is_market_open(datetime.now())}')"

   # Outputs: Market open: True/False
   ```

5. **Circuit breaker open**:
   - After 5 consecutive failures, trading halts
   - Check logs for "Circuit breaker" messages
   - Solution: Wait for recovery timeout (60s) or fix underlying issue

---

### Issue: Model Predictions Are Nonsensical

**Symptoms**: Predictions are extreme values (>10), all same value, or NaN

**Diagnosis**:

```bash
# Check recent predictions
tail -20 logs/trading_*.log | grep "Prediction:"

# Check for NaN in data
python -c "import pandas as pd; \
df = pd.read_csv('processed_stock_data.csv'); \
print(f'NaN count: {df.isna().sum().sum()}')"

# Check model file exists
ls -lh lstm_price_difference_model.h5
```

**Common Causes**:

1. **Data contains NaN values**:
   ```bash
   # Solution: Regenerate data
   ./run_daily_data_update.sh
   ```

2. **Model file corrupted**:
   ```bash
   # Verify checksum
   python -c "from model_management import calculate_model_checksum; \
   print(calculate_model_checksum('lstm_price_difference_model.h5'))"

   # Solution: Restore from backup or retrain
   ```

3. **Feature scaling mismatch**:
   - Training used normalization, inference using raw values
   - Solution: Ensure same preprocessing in train_model.py and live_trading.py

4. **Model drift**:
   ```bash
   # Check drift alerts
   grep "DRIFT ALERT" logs/*.log

   # Solution: Retrain model
   python train_model.py
   ```

---

### Issue: API Authentication Failures

**Symptoms**: Errors like "401 Unauthorized", "Invalid API key"

**Diagnosis**:

```bash
# Check credential files exist
ls -la *.enc

# Verify permissions
ls -l .schwab_config.enc  # Should be -rw------- (600)

# Test credential loading
python -c "from credential_manager import CredentialManager; \
cm = CredentialManager(); \
creds = cm.load_credentials('schwab'); \
print('Loaded:', bool(creds))"
```

**Common Causes**:

1. **Credentials not configured**:
   ```bash
   # Solution: Configure credentials
   python -c "from credential_manager import CredentialManager; \
   cm = CredentialManager(); \
   cm.store_credentials('schwab', {'app_key': 'YOUR_KEY', 'app_secret': 'YOUR_SECRET'})"
   ```

2. **Credentials expired**:
   - Schwab API keys need renewal
   - Solution: Generate new keys, update credentials

3. **File permissions wrong**:
   ```bash
   # Fix permissions
   chmod 600 .schwab_config.enc .reddit_config.enc
   ```

4. **Encryption key changed**:
   - If you reinstalled, encryption key may differ
   - Solution: Re-configure credentials

---

### Issue: Data Collection Failing

**Symptoms**: `processed_stock_data.csv` not updated, download errors

**Diagnosis**:

```bash
# Check data file timestamp
ls -lh processed_stock_data.csv

# Run data collection manually
./run_daily_data_update.sh

# Check for errors
tail -50 logs/data_update_$(date +%Y%m%d).log
```

**Common Causes**:

1. **Network connection issues**:
   ```bash
   # Test connectivity
   curl -I https://query1.finance.yahoo.com
   ```

2. **yfinance rate limiting**:
   - Yahoo Finance blocks excessive requests
   - Solution: Add delays between requests, use different IP

3. **Stock symbol delisted**:
   - Stock no longer trades
   - Solution: Remove from selected_stocks.txt

4. **API changes**:
   - yfinance API changed
   - Solution: Update yfinance: `pip install --upgrade yfinance`

---

### Issue: High CPU/Memory Usage

**Symptoms**: System slow, process killed (OOM), high load

**Diagnosis**:

```bash
# Check memory usage
ps aux | grep python | grep -v grep

# Check data file size
ls -lh processed_stock_data.csv

# Profile performance
python -m cProfile -s cumtime live_trading.py > profile.txt
head -30 profile.txt
```

**Common Causes**:

1. **Large dataset**:
   - CSV file too big for memory
   - Solution: Reduce date range or switch to database

2. **Memory leak**:
   - TensorFlow/Keras not releasing memory
   - Solution: Restart trading process periodically

3. **Too many features**:
   - 50+ features cause slow processing
   - Solution: Feature selection, remove unnecessary columns

---

### Issue: Tests Failing

**Symptoms**: `pytest` failures, CI/CD failing

**Diagnosis**:

```bash
# Run tests with verbose output
pytest tests/ -v --tb=short

# Run specific failing test
pytest tests/test_risk_management.py::TestRiskManager::test_pre_trade_check -v

# Check Python version
python --version  # Should be 3.12.x
```

**Common Causes**:

1. **Wrong Python version**:
   ```bash
   # Solution: Use Python 3.12
   python3.12 -m pytest tests/ -v
   ```

2. **Dependencies outdated**:
   ```bash
   # Solution: Reinstall dependencies
   pip install -r requirements.txt --force-reinstall
   ```

3. **Test data missing**:
   - Some tests expect specific data
   - Solution: Check test fixtures, regenerate data

---

### Issue: Audit Log Tampering Detected

**Symptoms**: "Hash chain broken", audit verification fails

**Diagnosis**:

```bash
# Verify audit log integrity
python -c "from audit_logging import AuditLogger; \
al = AuditLogger(); \
valid, error = al.verify_chain(); \
print(f'Valid: {valid}'); print(f'Error: {error}')"
```

**If Tampered**:

1. **CRITICAL SECURITY INCIDENT** - Possible breach
2. **Stop all trading immediately**
3. **Preserve evidence**:
   ```bash
   cp logs/audit/audit_log.jsonl audit_log_INCIDENT_$(date +%Y%m%d).jsonl
   ```
4. **Investigate**:
   - Who has system access?
   - When did tampering occur?
   - What was modified?
5. **Restore from backup** if tampering confirmed
6. **Change all credentials**
7. **Review security practices**

**If False Positive**:
- Log rotation may have truncated file mid-entry
- Solution: Rotate logs at midnight, not during trading

---

### Issue: Forced Buy-In Triggered

**Symptoms**: "Forced buy-in triggered", short position closed unexpectedly

**Diagnosis**:

```bash
# Check forced buy-in events
grep "forced buy-in\|BROKER RECALL\|HARD TO BORROW" logs/*.log -i

# Check borrow rate
grep "borrow rate" logs/*.log -i
```

**Reasons**:

1. **Broker recalled shares** (normal, handle gracefully)
2. **Stock became hard-to-borrow** (borrow rate >20%)
3. **Dividend approaching** (ex-dividend date within 3 days)
4. **Corporate action** (merger, bankruptcy)

**Actions**:
- System automatically closes position
- Log event in audit trail
- If frequent, avoid shorting that stock

---

### Issue: Model Drift Alert

**Symptoms**: "DRIFT ALERT" in logs, model performance degrading

**Diagnosis**:

```bash
# Check drift severity
grep "DRIFT ALERT" logs/*.log | tail -10

# Check recent performance
grep "directional_accuracy\|MSE" logs/*.log | tail -20
```

**Severity Levels**:

- **LOW**: Monitor closely (15-25% degradation)
- **MEDIUM**: Schedule retraining within 1 week
- **HIGH**: Retrain within 24 hours
- **CRITICAL**: Stop trading, retrain immediately

**Actions**:

1. **For LOW/MEDIUM**:
   ```bash
   # Retrain when convenient
   python train_model.py
   ```

2. **For HIGH/CRITICAL**:
   ```bash
   # Stop trading
   crontab -r

   # Retrain immediately
   python train_model.py

   # Backtest new model
   ./run_backtest.sh

   # Resume trading if backtest passes
   crontab -e  # Re-enable
   ```

---

## Performance Degradation

### Slow Predictions (>5 seconds)

**Diagnosis**:

```bash
# Time prediction
time python -c "from live_trading import get_latest_signal; get_latest_signal()"
```

**Solutions**:

1. **Pre-load model**:
   - Load model once at startup, reuse
   - Don't reload for every prediction

2. **Reduce data**:
   - Use only last 90 days instead of full history
   - Filter unnecessary columns

3. **Use GPU** (if available):
   ```bash
   # Check TensorFlow GPU support
   python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
   ```

---

## Monitoring & Alerts

### Set Up Email Alerts

```bash
# Install sendmail
sudo apt install sendmail

# Create alert script
cat << 'EOF' > send_alert.sh
#!/bin/bash
MESSAGE=$1
echo "$MESSAGE" | mail -s "Trading Alert" your-email@example.com
EOF

chmod +x send_alert.sh

# Use in cron or scripts
if grep -q "CRITICAL" logs/*.log; then
  ./send_alert.sh "CRITICAL: Check trading system immediately"
fi
```

### Set Up Monitoring Dashboard

```bash
# Simple monitoring with watch
watch -n 60 '
echo "=== System Status ==="
tail -5 logs/trading_*.log
echo ""
echo "=== Last 3 Trades ==="
tail -3 logs/audit/audit_log.jsonl | jq .
'
```

---

## Escalation

### When to Escalate

- **Immediate**: Money loss >5%, security breach, system compromise
- **Same Day**: Critical drift, repeated failures, data corruption
- **Next Day**: Medium drift, performance issues, minor bugs

### Escalation Contacts

1. **System Owner**: (Primary contact)
2. **DevOps Team**: (Infrastructure issues)
3. **Security Team**: (Breach, tampering)

---

## Useful Commands

```bash
# View recent trades
tail -20 logs/audit/audit_log.jsonl | jq '.action, .status, .details'

# Check system health
./run_health_check.sh  # Create this script with key checks

# Restart trading (after fixing issue)
crontab -e  # Re-enable cron jobs

# Force model reload
rm -rf __pycache__/
rm lstm_price_difference_model.h5  # Then restore from backup

# Emergency data regeneration
./run_daily_data_update.sh --force

# Test API connectivity
python -c "from trading_api import SchwabAPI; api = SchwabAPI(); print(api.get_account_balance())"
```

---

## Log Locations

- **Trading**: `logs/trading_YYYYMMDD.log`
- **Audit**: `logs/audit/audit_log.jsonl`
- **Data Updates**: `logs/data_update_YYYYMMDD.log`
- **Cron**: `logs/cron_trading.log`
- **Errors**: `logs/errors.log`

---

## Related Documentation

- [Architecture](ARCHITECTURE.md)
- [Deployment Guide](DEPLOYMENT.md)
- [Development Guide](DEVELOPMENT.md)
