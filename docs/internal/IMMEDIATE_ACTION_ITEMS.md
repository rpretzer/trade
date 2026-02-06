# Immediate Action Items - Critical Fixes

> **STATUS: ALL RESOLVED (2026-01-29)**
> Every item below was fixed in the Phase 2 sprint.  See `FIXES_COMPLETED.md`
> for implementation details.  This file is retained as a historical record.

~~**Priority**: STOP ALL LIVE TRADING UNTIL THESE ARE FIXED~~

## ~~CRITICAL BUGS THAT WILL LOSE MONEY~~ (all fixed)

### 1. Stop-Loss Logic is Inverted (CRITICAL)
**File**: `backtest_strategy.py:320`

**Current Code (WRONG)**:
```python
if apply_stop_loss(current_price, entry_price, stop_loss_pct, is_long):
    loss = (entry_price - current_price) * pos['quantity']
    capital += loss  # ❌ ADDS LOSS TO CAPITAL (should subtract)
```

**What happens**: When stop-loss triggers, you GAIN money instead of losing money. This means:
- Backtests show false profits
- Risk management doesn't work
- Will fail catastrophically in live trading

**Fix**:
```python
if apply_stop_loss(current_price, entry_price, stop_loss_pct, is_long):
    if is_long:
        loss = (entry_price - current_price) * pos['quantity']
    else:
        loss = (current_price - entry_price) * pos['quantity']
    capital -= abs(loss)  # ✓ SUBTRACT LOSS FROM CAPITAL
```

**Action**: Fix immediately before any backtests

---

### 2. Position Sizing Uses Hardcoded Prices (CRITICAL)
**File**: `trading_api.py:375-401`

**Current Code (WRONG)**:
```python
# Place buy order for AAPL
order1 = self.place_order('AAPL', int(position_value / 150), 'BUY', 'MARKET')
                                                      # ^^^ HARDCODED $150

# Place sell order for MSFT
order2 = self.place_order('MSFT', int(position_value / 300), 'SELL', 'MARKET')
                                                      # ^^^ HARDCODED $300
```

**What happens**:
- If AAPL is actually $180, you'll try to buy too few shares
- If AAPL is actually $120, you'll try to buy too many shares
- Orders will be sized incorrectly
- Can exceed position limits or fail to fill

**Fix**:
```python
# Get current prices
aapl_price = self.get_current_price('AAPL')
msft_price = self.get_current_price('MSFT')

# Calculate quantities based on ACTUAL prices
aapl_qty = int(position_value / aapl_price)
msft_qty = int(position_value / msft_price)

# Place orders
order1 = self.place_order('AAPL', aapl_qty, 'BUY', 'MARKET')
order2 = self.place_order('MSFT', msft_qty, 'SELL', 'MARKET')
```

**Also need to implement**:
```python
def get_current_price(self, symbol):
    """Fetch current market price for symbol."""
    # Use Schwab API or market data feed
    # Add staleness check (reject if > 1 second old)
    # Add sanity check (reject if price changed > 10% from last)
    pass
```

**Action**: Implement `get_current_price()` and fix all order sizing

---

### 3. Hardcoded Salt Breaks Encryption Security (CRITICAL)
**File**: `credential_manager.py:59`

**Current Code (WRONG)**:
```python
kdf = PBKDF2HMAC(
    algorithm=hashes.SHA256(),
    length=32,
    salt=b'schwab_arbitrage_salt',  # ❌ SAME FOR ALL USERS
    iterations=100000,
)
```

**What happens**:
- All users share same salt
- If one user's password is cracked, others are vulnerable
- Rainbow table attacks are possible
- Does NOT meet security standards

**Fix**:
```python
import os

# Generate random salt per user
salt = os.urandom(32)

kdf = PBKDF2HMAC(
    algorithm=hashes.SHA256(),
    length=32,
    salt=salt,  # ✓ UNIQUE PER USER
    iterations=100000,
)

# Store salt alongside encrypted data
encrypted_data = {
    'salt': base64.b64encode(salt).decode(),
    'ciphertext': base64.b64encode(ciphertext).decode()
}
```

**Action**: Regenerate all stored credentials with unique salts

---

## 🔴 DO NOT DEPLOY TO PRODUCTION UNTIL FIXED

### 4. No Transaction Costs in Backtest
**Impact**: Backtest returns will be 20-50% higher than reality

Real costs you're missing:
- Commission: ~$0.005/share
- SEC fees: $0.00002/dollar (sells only)
- Short borrow costs: 1-10% annually
- Slippage: 0.01-0.05% per trade
- Spread: bid-ask spread cost

**Example**:
- Trade 1000 shares 10 times/day
- Cost: ~$50-100/day = $12K-25K/year
- On $100K capital, that's 12-25% of returns

**Fix**: Add transaction cost model to `backtest_strategy.py`

---

### 5. No Slippage Modeling
**Impact**: Assume perfect fills at exact prices (never happens in reality)

Reality:
- Market orders slip 0.01-0.10% depending on size
- Large orders move the market
- Volatile periods have more slippage

**Fix**: Add slippage model based on order size and volatility

---

### 6. Model Has No Validation Set
**File**: `train_model.py:172-174`

**Current Code (WRONG)**:
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, shuffle=False
)
# ❌ Only train/test split, no validation set
# ❌ Will overfit on test set during development
```

**What happens**:
- You tune model looking at test set
- Test set becomes part of training
- Overfitting to test data
- Model fails in production

**Fix**: Implement walk-forward validation:
```python
# Split data: 60% train, 20% validation, 20% test
train_end = int(len(X) * 0.6)
val_end = int(len(X) * 0.8)

X_train = X[:train_end]
X_val = X[train_end:val_end]
X_test = X[val_end:]

# Train on train set
# Tune on validation set
# NEVER look at test set until final evaluation
```

**Action**: Retrain all models with proper train/val/test split

---

### 7. No Data Quality Checks
**Files**: `process_stock_data.py`

**Missing validations**:
- Are there gaps in dates?
- Are prices reasonable? (detect errors)
- Is volume reasonable?
- Are technical indicators in valid ranges?

**Example of what can go wrong**:
```python
# Downloaded data
Date       | AAPL  | MSFT
2026-01-01 | 150.0 | 300.0
2026-01-02 | 0.01  | 301.0  # ❌ ERROR: Price dropped 99.99%
2026-01-03 | 152.0 | 302.0  # Data resumes normally
```

Without validation:
- Model trains on bad data
- Predictions will be garbage
- Trades will lose money

**Fix**: Add validation layer:
```python
def validate_price_data(df):
    """Validate price data before use."""
    # Check for missing dates
    date_range = pd.date_range(df.index.min(), df.index.max(), freq='B')
    missing_dates = date_range.difference(df.index)
    if len(missing_dates) > 0:
        raise DataQualityError(f"Missing {len(missing_dates)} trading days")

    # Check for price anomalies (> 20% change)
    returns = df.pct_change()
    outliers = returns[abs(returns) > 0.2]
    if len(outliers) > 0:
        raise DataQualityError(f"Found {len(outliers)} extreme price moves")

    # Check for zero/negative prices
    if (df <= 0).any().any():
        raise DataQualityError("Found zero or negative prices")

    return True
```

---

### 8. Deprecated Pandas Code
**File**: `process_stock_data.py:532`

**Current Code (DEPRECATED)**:
```python
price_df[col] = price_df[col].fillna(method='ffill').fillna(0)
                                           # ^^^ DEPRECATED
```

**What happens**:
- Works in pandas 1.x and 2.x
- Will break in pandas 3.0
- Code will stop working with no warning

**Fix**:
```python
price_df[col] = price_df[col].ffill().fillna(0)
```

**Action**: Update immediately (takes 5 minutes)

---

## 📋 Critical Fixes Checklist

Before ANY live trading:

- [ ] Fix stop-loss logic (adds instead of subtracts)
- [ ] Fix position sizing (uses hardcoded prices)
- [ ] Fix credential encryption (hardcoded salt)
- [ ] Add transaction costs to backtest
- [ ] Add slippage model to backtest
- [ ] Implement proper train/val/test split
- [ ] Add data quality validation
- [ ] Fix deprecated pandas code
- [ ] Test all fixes with unit tests
- [ ] Re-run backtests with fixes
- [ ] Compare old vs new backtest results
- [ ] Document performance changes

**Expected Impact**:
- Backtest returns will DROP significantly (20-50%)
- If backtests still profitable after fixes → good signal
- If backtests unprofitable after fixes → strategy doesn't work

---

## 🚨 Emergency Contacts

If you deploy before fixing these and lose money:

1. **Immediate Stop**:
   ```bash
   # Kill all trading processes
   pkill -f "trading_api.py"
   pkill -f "live_trading.py"

   # Or use emergency stop in code
   python -c "from trading_api import TradingAPI; api = TradingAPI(); api.emergency_stop()"
   ```

2. **Liquidate Positions**:
   - Log into Schwab account manually
   - Close all positions immediately
   - Document losses for post-mortem

3. **Post-Mortem**:
   - Review what went wrong
   - Check audit logs
   - Calculate actual vs expected performance
   - Fix root cause before restarting

---

## ⏱️ Time to Fix

| Issue | Estimated Time | Priority |
|-------|----------------|----------|
| Stop-loss logic | 2 hours | P0 |
| Position sizing | 4 hours | P0 |
| Credential salt | 3 hours | P0 |
| Transaction costs | 8 hours | P0 |
| Slippage model | 6 hours | P0 |
| Train/val/test split | 4 hours | P0 |
| Data validation | 6 hours | P0 |
| Pandas deprecation | 0.5 hours | P1 |
| **TOTAL** | **33.5 hours** | **~1 week** |

**Recommendation**:
- Allocate 1 week for critical fixes
- Test thoroughly (another 1 week)
- Re-run backtests (1 week)
- **Total: 3 weeks before production**

---

## After Critical Fixes

Once these 8 critical issues are fixed:
1. Re-run all backtests
2. Compare results to original
3. If still profitable → proceed to Phase 2 of roadmap
4. If not profitable → re-evaluate strategy

Then follow `PRODUCTION_ROADMAP.md` for remaining work.

**Remember**: These are just the CRITICAL fixes. There are 81 more issues in the full audit that need addressing before true production deployment.
