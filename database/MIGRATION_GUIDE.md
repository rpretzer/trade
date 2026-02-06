## Database Migration Guide

### Status: Infrastructure Complete, Application Migration In Progress

### Completed ✅
1. **Database Schema** - Created 15 production tables (schema.sql)
2. **ORM Models** - SQLAlchemy models for all tables (models.py)
3. **Connection Manager** - Connection pooling and session management (connection.py)
4. **Data Access Layer** - High-level API to replace CSV operations (data_access.py)
5. **Migration Script** - Tool to migrate existing CSV data to database (migrate_csv_to_db.py)
6. **Storage Config** - Backend switching between CSV/Database (storage_config.py)

### How to Use

#### 1. Initialize Database
```bash
# Set database URL
export DATABASE_URL="postgresql://localhost:5432/trading_db"

# Initialize schema
cd database
python connection.py
```

#### 2. Migrate Existing CSV Data
```bash
# Run migration script
python database/migrate_csv_to_db.py
```

This will migrate:
- `processed_stock_data.csv` → prices + technical_indicators tables
- `reddit_sentiment_cache.csv` → sentiment_data table
- `options_volume_cache.csv` → options_data table

#### 3. Switch to Database Backend
```bash
# Set environment variable to use database
export USE_DATABASE=true

# Or in code:
from storage_config import use_database
use_database()
```

### Application Files to Update

The following files still need updating to use `data_access.py` instead of CSV:

#### High Priority:
1. **backtest_strategy.py**
   - Replace `load_processed_data()` with `DataAccess.load_processed_stock_data()`
   - Replace `save_backtest_results()` with database insertion

2. **process_stock_data.py**
   - Replace `final_df.to_csv()` with `DataAccess.save_processed_stock_data()`

3. **live_trading.py**
   - Replace `pd.read_csv(data_file)` with `DataAccess.load_processed_stock_data()`

4. **train_model.py**
   - Replace `pd.read_csv()` with `DataAccess.load_processed_stock_data()`

5. **sentiment_analysis.py**
   - Replace cache CSV read/write with `DataAccess.load/save_sentiment_data()`

6. **options_data.py**
   - Replace cache CSV read/write with `DataAccess.load/save_options_data()`

#### Medium Priority:
7. **dashboard.py**
   - Update data loading functions
   - Query database for backtest results

8. **main.py**
   - Update file existence checks
   - Remove CSV references

### Example Conversion

**Before (CSV):**
```python
df = pd.read_csv('processed_stock_data.csv', index_col=0, parse_dates=True)
```

**After (Database):**
```python
from database import get_data_access

da = get_data_access()
df = da.load_processed_stock_data(symbols=['AAPL', 'MSFT'])
```

### Testing

```bash
# Test database connection
python database/connection.py

# Test migration
python database/migrate_csv_to_db.py

# Run tests with database backend
export USE_DATABASE=true
pytest tests/
```

### Rollback to CSV

If issues arise, switch back to CSV:
```bash
export USE_DATABASE=false
```

### Next Steps

1. Run migration script to populate database
2. Update application files one at a time
3. Test each file after updating
4. Once all files updated, deprecate CSV backend
5. Remove CSV files (after backup)
