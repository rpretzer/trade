# Database Setup Guide

## Quick Start

### 1. Install PostgreSQL

**macOS:**
```bash
brew install postgresql@14
brew services start postgresql@14
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install postgresql postgresql-contrib
sudo systemctl start postgresql
```

**Docker:**
```bash
docker run --name trading-postgres \
  -e POSTGRES_PASSWORD=your_password \
  -e POSTGRES_DB=trading_db \
  -p 5432:5432 \
  -d postgres:14
```

### 2. Create Database

```bash
# Connect to PostgreSQL
psql -U postgres

# Create database and user
CREATE DATABASE trading_db;
CREATE USER trading_app WITH PASSWORD 'your_secure_password';
GRANT ALL PRIVILEGES ON DATABASE trading_db TO trading_app;

# Exit psql
\q
```

### 3. Set Environment Variable

```bash
export DATABASE_URL="postgresql://trading_app:your_secure_password@localhost:5432/trading_db"
```

Or create a `.env` file:
```
DATABASE_URL=postgresql://trading_app:your_secure_password@localhost:5432/trading_db
```

### 4. Initialize Database Schema

```bash
# From project root
cd database
psql -U trading_app -d trading_db -f schema.sql
```

Or use Python:
```python
from database import init_database

# Initialize with schema
db = init_database()
print("Database initialized!")
```

### 5. Verify Installation

```bash
python database/connection.py
```

Expected output:
```
Database Connection Manager
======================================================================
Database connection initialized: postgresql://trading_app:****@localhost:5432/trading_db
Creating database tables...
Database tables created successfully
Executing schema file: schema.sql
Database initialized and healthy

Health check: ✓ Healthy

Example query:
Found 7 stocks:
  - AAPL: Apple Inc. (NASDAQ)
  - MSFT: Microsoft Corporation (NASDAQ)
  - ...

Database connections closed
```

## Usage Examples

### Basic Query

```python
from database import get_db_manager
from database.models import Stock, Price

db = get_db_manager()

with db.get_session() as session:
    # Query stocks
    stocks = session.query(Stock).filter_by(exchange='NASDAQ').all()

    # Get latest price
    latest_price = session.query(Price).filter_by(symbol='AAPL').order_by(Price.date.desc()).first()

    print(f"AAPL latest close: ${latest_price.close}")
```

### Insert Data

```python
from database import get_db_manager
from database.models import Price
from datetime import date

db = get_db_manager()

with db.get_session() as session:
    # Insert new price data
    price = Price(
        symbol='AAPL',
        date=date.today(),
        open=150.00,
        high=152.00,
        low=149.00,
        close=151.50,
        volume=100000000
    )
    session.add(price)
    # Commit happens automatically when context exits
```

### Bulk Insert

```python
import pandas as pd
from database import get_db_manager
from database.models import Price

db = get_db_manager()

# Load data from CSV
df = pd.read_csv('prices.csv')

with db.get_session() as session:
    # Convert DataFrame to list of dicts
    records = df.to_dict('records')

    # Bulk insert
    session.bulk_insert_mappings(Price, records)
```

### Query with Joins

```python
from database import get_db_manager
from database.models import Stock, Trade

db = get_db_manager()

with db.get_session() as session:
    # Query trades with stock details
    trades = (
        session.query(Trade, Stock)
        .join(Stock, Trade.symbol == Stock.symbol)
        .filter(Stock.sector == 'Technology')
        .all()
    )

    for trade, stock in trades:
        print(f"{stock.name}: {trade.quantity} @ ${trade.price}")
```

### Execute Raw SQL

```python
from database import get_db_manager
from sqlalchemy import text

db = get_db_manager()

with db.get_session() as session:
    result = session.execute(text("""
        SELECT symbol, SUM(quantity * price) as total_value
        FROM trades
        WHERE execution_time >= CURRENT_DATE - INTERVAL '7 days'
        GROUP BY symbol
        ORDER BY total_value DESC
        LIMIT 10
    """))

    for row in result:
        print(f"{row.symbol}: ${row.total_value:,.2f}")
```

## Database Schema

See `schema.sql` for complete schema definition.

### Key Tables

- **stocks**: Reference data for all symbols
- **prices**: OHLCV price data (partitioned by date)
- **technical_indicators**: RSI, MACD, moving averages
- **sentiment_data**: Reddit/Twitter sentiment scores
- **options_data**: Options volume and implied volatility
- **models**: ML model registry with versioning
- **model_metrics**: Performance metrics for each model
- **predictions**: Model predictions with confidence
- **orders**: All orders (pending, filled, cancelled)
- **trades**: Executed trades with transaction costs
- **positions**: Current and historical positions
- **risk_metrics**: Daily risk snapshots
- **circuit_breaker_events**: Risk limit violations
- **audit_log**: Immutable audit trail

### Useful Views

- **v_current_positions**: Current open positions summary
- **v_daily_pnl**: Daily P&L aggregation
- **v_model_performance**: Model comparison metrics

## Migrations

For schema changes in production, use Alembic:

```bash
# Install Alembic
pip install alembic

# Initialize Alembic
alembic init migrations

# Create migration
alembic revision --autogenerate -m "Add new column"

# Apply migration
alembic upgrade head
```

## Backup & Restore

### Backup

```bash
# Backup entire database
pg_dump -U trading_app trading_db > backup_$(date +%Y%m%d).sql

# Backup specific tables
pg_dump -U trading_app -t orders -t trades trading_db > trades_backup.sql
```

### Restore

```bash
# Restore from backup
psql -U trading_app trading_db < backup_20260129.sql
```

## Performance Tips

1. **Indexes**: Schema includes indexes on commonly queried columns
2. **Partitioning**: `prices` table is partitioned by date for faster queries
3. **Connection Pooling**: Uses SQLAlchemy connection pool (10 connections)
4. **Query Optimization**: Use `EXPLAIN ANALYZE` to optimize slow queries

```sql
EXPLAIN ANALYZE
SELECT * FROM prices WHERE symbol = 'AAPL' AND date >= '2024-01-01';
```

## Troubleshooting

### Connection Refused

```bash
# Check if PostgreSQL is running
pg_isready -h localhost -p 5432

# Start PostgreSQL
sudo systemctl start postgresql  # Linux
brew services start postgresql@14  # macOS
```

### Permission Denied

```sql
-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO trading_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO trading_app;
```

### Too Many Connections

```sql
-- Check current connections
SELECT count(*) FROM pg_stat_activity;

-- Kill idle connections
SELECT pg_terminate_backend(pid)
FROM pg_stat_activity
WHERE state = 'idle' AND state_change < CURRENT_TIMESTAMP - INTERVAL '5 minutes';
```

## Security

- **Never commit credentials** to version control
- Use environment variables for DATABASE_URL
- Rotate passwords regularly
- Use SSL in production:
  ```
  DATABASE_URL=postgresql://user:pass@host:5432/db?sslmode=require
  ```
- Implement row-level security for multi-tenant deployments
