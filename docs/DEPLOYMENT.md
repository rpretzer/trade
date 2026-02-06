# Deployment Guide

Complete guide to deploying the Stock Arbitrage Model to staging and production environments.

---

## Prerequisites

### System Requirements

- **OS**: Linux (Ubuntu 20.04+) or macOS
- **Python**: 3.10+ (3.12 recommended for TensorFlow 2.15)
- **Memory**: 4 GB RAM minimum, 8 GB recommended
- **Disk**: 10GB free space
- **Network**: Stable internet connection for API access

### Required Accounts

1. **GitHub**: Source code access
2. **Schwab**: Trading API credentials (for live trading)
3. **Reddit**: API credentials (for sentiment analysis)

---

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/rpretzer/trade.git stock_arbitrage_model
cd stock_arbitrage_model
```

### 2. Create Python Virtual Environment

**Using Python 3.12** (required for TensorFlow):

```bash
# Install Python 3.12 if not available
# On Ubuntu:
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.12 python3.12-venv

# Create virtual environment
python3.12 -m venv venv_py312
source venv_py312/bin/activate

# Verify Python version
python --version  # Should show Python 3.12.x
```

### 3. Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install core dependencies
pip install -r requirements.txt

# Install development dependencies (optional)
pip install -r requirements-dev.txt
```

Optional feature packs live in separate files.  The CLI can install them
for you interactively (`python main.py deps`), or install manually:

```bash
pip install -r requirements-lstm.txt       # LSTM training (needs Python 3.11/3.12)
pip install -r requirements-live.txt       # live Schwab trading
pip install -r requirements-advanced.txt   # cointegration, market calendars
pip install -r requirements-full.txt       # everything above
```

### 4. Configure Credentials

#### Schwab API (Trading)

```bash
# Create encrypted credentials
python -c "from credential_manager import CredentialManager; \
cm = CredentialManager(); \
cm.store_credentials('schwab', {'app_key': 'YOUR_APP_KEY', 'app_secret': 'YOUR_APP_SECRET', 'redirect_uri': 'https://127.0.0.1', 'account_hash': 'YOUR_ACCOUNT_HASH'})"
```

#### Reddit API (Sentiment)

```bash
# Store Reddit credentials
python -c "from credential_manager import CredentialManager; \
cm = CredentialManager(); \
cm.store_credentials('reddit', {'client_id': 'YOUR_CLIENT_ID', 'client_secret': 'YOUR_CLIENT_SECRET', 'user_agent': 'StockArbitrage/1.0'})"
```

**Security Note**: Credentials are encrypted with AES-256 and stored in `.schwab_config.enc` and `.reddit_config.enc`. Never commit these files to git.

### 5. Configure Stock Selection

Edit `selected_stocks.txt`:

```txt
AAPL
MSFT
```

Start with 2-3 highly correlated stocks.

### 6. Set Up Data Collection

```bash
# Make scripts executable
chmod +x run_daily_data_update.sh
chmod +x run_backtest.sh

# Run initial data collection
./run_daily_data_update.sh
```

This downloads historical data and creates `processed_stock_data.csv`.

---

## Training the Model

### Initial Model Training

```bash
python train_model.py
```

**Expected Output**:
```
Loading data...
Creating sequences...
Training LSTM model...
Epoch 1/100
...
Model saved to: lstm_price_difference_model.h5
```

**Training Time**: 10-30 minutes depending on hardware

**Output Files**:
- `lstm_price_difference_model.h5` - Trained model
- Training metrics logged to console

### Validate Model Performance

```bash
./run_backtest.sh
```

**Expected Output**:
```
Backtesting strategy with threshold=0.5...
Initial capital: $10,000.00
...
BACKTEST RESULTS
=====================================
Initial Capital:      $10,000.00
Final Capital:        $10,500.00
Total Return:         5.00%
Win Rate:             65.00%
```

**Important**: Backtest results should show:
- Win rate > 55%
- Positive total return
- Max drawdown < 10%

If results are poor, retrain with more data or adjust hyperparameters.

---

## Staging Deployment

### 1. Set Up Cron Jobs (Data Update)

```bash
crontab -e
```

Add:

```cron
# Update stock data daily at 4 PM Eastern (after market close)
0 16 * * 1-5 cd /path/to/stock_arbitrage_model && ./run_daily_data_update.sh

# Note: Adjust timezone as needed
```

### 2. Run in Paper Trading Mode

Paper-trading is **on by default** (`paper_trading=True` in
`trading_api.py`).  No environment variable is needed — just run:

```bash
python main.py live       # simulates trades; no real orders placed
```

**Monitor for**:
- Successful data fetching
- Model predictions generated
- Signals triggered correctly
- No crashes or errors

### 3. Monitor Logs

```bash
# Watch live trading logs
tail -f logs/trading_$(date +%Y%m%d).log

# Watch audit logs
tail -f logs/audit/audit_log.jsonl
```

### 4. Validate Model Drift Detection

```bash
# After collecting 100+ predictions, check drift
python -c "from model_drift_detection import ModelDriftMonitor; \
# (Add validation code here based on collected data)"
```

---

## Production Deployment

### Prerequisites Checklist

Before deploying to production:

- [ ] Backtest shows positive returns over 6+ months
- [ ] Paper trading ran successfully for 1+ week
- [ ] All 287 tests passing (`pytest tests/ -v`)
- [ ] Audit logs verified (no tampering detected)
- [ ] Risk limits configured appropriately
- [ ] Drift alerts reviewed in `logs/alerts.log`
- [ ] Backup plan in place

### 1. Set Capital Limits

Edit `risk_management.py` or create config:

```python
RiskLimits(
    max_position_size=100,        # Start small!
    max_position_value=5000,      # $5,000 max per position
    max_single_position_pct=0.10, # 10% max concentration
    max_drawdown_pct=0.03,        # 3% max drawdown
    critical_drawdown_pct=0.05    # 5% emergency halt
)
```

**Start with small amounts**: $1,000-$5,000 total capital initially.

### 2. Enable Live Trading

Live trading is enabled by passing `paper_trading=False` when
instantiating the trading client.  This is done through menu option 5
(`python main.py live`) after Schwab credentials have been configured
via option 6 (`python main.py setup`).  The system will refuse to place
real orders if `schwab-py` is not installed or credentials are missing.

### 3. Start the Live Trading Loop

The system runs a continuous trading loop (not cron-based).  It sleeps
automatically outside market hours and checks for signals every 5 minutes:

```bash
python main.py live       # starts continuous loop; Ctrl-C for graceful shutdown
```

The loop handles market-hours detection, position persistence (JSON file),
and a circuit breaker that halts after 5 consecutive errors.

### 4. Monitor Closely

**First Week**: Check logs hourly
**First Month**: Check logs daily
**Ongoing**: Check logs weekly + review performance monthly

```bash
# Daily monitoring script
cat << 'EOF' > monitor_daily.sh
#!/bin/bash
echo "=== Trading Performance ==="
tail -20 logs/trading_$(date +%Y%m%d).log

echo "\n=== Recent Trades ==="
tail -10 logs/audit/audit_log.jsonl | jq .

echo "\n=== System Health ==="
# Check for errors
grep -i error logs/*.log | tail -10

# Check drift alerts (structured JSON)
tail -5 logs/alerts.log 2>/dev/null || echo "No alerts yet"
EOF

chmod +x monitor_daily.sh
./monitor_daily.sh
```

### 5. Emergency Shutdown

If things go wrong:

```bash
# Stop all trading
crontab -r  # Remove all cron jobs

# Or selectively comment out trading cron job
crontab -e
# Add # before live_trading.py line

# Manually close all positions via Schwab web interface
```

---

## Configuration

### Environment Variables

```bash
# Log level (DEBUG, INFO, WARNING, ERROR)
export LOG_LEVEL=INFO

# Disable ANSI colour in CLI output
export NO_COLOR=1

# Webhook URL for HIGH/CRITICAL alerts (Slack, Discord, PagerDuty, etc.)
export ALERT_WEBHOOK_URL=https://hooks.slack.com/services/...

# Enable LSTM + XGBoost ensemble predictions
export ENSEMBLE_MODE=1
```

Paper vs. live trading is controlled in code (`paper_trading` flag),
not via an environment variable.  See option 6 in the CLI for
credential setup.

### Risk Parameters

Edit `constants.py`:

```python
# Staging values (conservative)
DEFAULT_POSITION_SIZE_PCT = 0.05  # 5% per trade
DEFAULT_MAX_DRAWDOWN_PCT = 0.03   # 3% max drawdown
DEFAULT_STOP_LOSS_PCT = 0.02      # 2% stop loss

# Production values (less conservative after validation)
DEFAULT_POSITION_SIZE_PCT = 0.10  # 10% per trade
DEFAULT_MAX_DRAWDOWN_PCT = 0.05   # 5% max drawdown
DEFAULT_STOP_LOSS_PCT = 0.02      # 2% stop loss
```

### Trading Thresholds

Edit `constants.py`:

```python
# Signal threshold for trading
DEFAULT_TRADING_THRESHOLD = 0.5

# Increase for higher confidence (fewer trades)
DEFAULT_TRADING_THRESHOLD = 0.7

# Decrease for more trades (higher risk)
DEFAULT_TRADING_THRESHOLD = 0.3
```

---

## Monitoring & Maintenance

### Daily Checks

1. **Verify data updated**:
   ```bash
   ls -lh processed_stock_data.csv
   # Should show today's date
   ```

2. **Check for errors**:
   ```bash
   grep -i error logs/*.log
   ```

3. **Review trades**:
   ```bash
   tail -20 logs/audit/audit_log.jsonl | jq .
   ```

### Weekly Checks

1. **Review performance**:
   ```bash
   python -c "from risk_management import RiskManager; \
   rm = RiskManager(); \
   metrics = rm.calculate_current_metrics([...]); \
   print(metrics)"
   ```

2. **Check model drift**:
   ```bash
   cat logs/alerts.log          # structured JSON; HIGH/CRITICAL entries are urgent
   ```

3. **Verify backups** (if configured)

### Monthly Checks

1. **Retrain model** if drift detected
2. **Review risk limits** based on performance
3. **Update dependencies**: `pip list --outdated`
4. **Security audit**: `safety check`

---

## Backup & Recovery

### Backup Critical Files

```bash
# Create backup script
cat << 'EOF' > backup.sh
#!/bin/bash
BACKUP_DIR=/path/to/backups/$(date +%Y%m%d)
mkdir -p $BACKUP_DIR

# Backup model
cp lstm_price_difference_model.h5 $BACKUP_DIR/

# Backup data
cp processed_stock_data.csv $BACKUP_DIR/

# Backup config
cp selected_stocks.txt $BACKUP_DIR/
cp *.enc $BACKUP_DIR/

# Backup logs
cp -r logs/ $BACKUP_DIR/

# Create archive
tar -czf $BACKUP_DIR.tar.gz $BACKUP_DIR
rm -rf $BACKUP_DIR

echo "Backup created: $BACKUP_DIR.tar.gz"
EOF

chmod +x backup.sh
```

Run backup weekly:

```cron
0 0 * * 0 cd /path/to/stock_arbitrage_model && ./backup.sh
```

### Recovery

```bash
# Restore from backup
tar -xzf /path/to/backups/20260131.tar.gz
cp 20260131/* ./
```

---

## Troubleshooting

See [RUNBOOK.md](RUNBOOK.md) for detailed troubleshooting.

**Common Issues**:

1. **TensorFlow import error**:
   - Ensure Python 3.12 (not 3.14)
   - `pip install tensorflow==2.15.0`

2. **Credential errors**:
   - Verify `.schwab_config.enc` exists
   - Re-run credential setup

3. **No trades executing**:
   - Verify Schwab API credentials are configured (`python main.py setup`)
   - Check trading threshold (may be too high)
   - Ensure market is open (loop sleeps outside NYSE hours)

4. **High losses**:
   - Emergency shutdown (see above)
   - Review recent trades in audit log
   - Check if drawdown limit was hit

---

## Performance Optimization

### For Higher Throughput

1. **Enable multiprocessing** (future enhancement)
2. **Use faster data storage** (PostgreSQL instead of CSV)
3. **Cache API responses** (Redis)

### For Lower Latency

1. **Pre-load model** at startup
2. **Use batch predictions**
3. **Optimize pandas operations**

See [Performance Profiling](PERFORMANCE_PROFILING.md) for details.

---

## Security Hardening

### Production Security Checklist

- [ ] Credentials encrypted (never plaintext)
- [ ] File permissions set correctly (600 for .enc files)
- [ ] API keys rotated regularly
- [ ] Audit logs enabled
- [ ] Request signing enabled
- [ ] Firewall configured
- [ ] SSH key-based auth only
- [ ] Regular security updates (`apt update && apt upgrade`)

### Audit Log Verification

```bash
# Verify audit log integrity
python -c "from audit_logging import AuditLogger; \
al = AuditLogger(); \
valid, error = al.verify_chain(); \
print(f'Valid: {valid}, Error: {error}')"
```

Should output: `Valid: True, Error: None`

---

## Rollback Procedure

If deployment fails:

1. **Stop all cron jobs**: `crontab -r`
2. **Restore from backup**: See Backup & Recovery above
3. **Verify tests pass**: `pytest tests/ -v`
4. **Review logs**: Check what went wrong
5. **Fix issue**: Make changes
6. **Test in staging**: Before re-deploying to production

---

## Next Steps

After successful deployment:

1. **Monitor for 1 week** in production with small capital
2. **Review performance** and adjust parameters
3. **Gradually increase** capital allocation
4. **Set up alerts** (email/SMS) for critical events
5. **Plan infrastructure** upgrades based on real usage

See the "Future Enhancements" section of [README](../README.md) for long-term plans.

---

## Support

For issues:
1. Check [RUNBOOK.md](RUNBOOK.md)
2. Review logs in `logs/`
3. Run tests: `pytest tests/ -v`
4. Check GitHub issues: https://github.com/rpretzer/trade/issues
