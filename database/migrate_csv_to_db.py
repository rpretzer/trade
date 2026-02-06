"""
CSV to Database Migration Script

Migrates all CSV-based data to PostgreSQL database.
Run this once to populate the database with existing data.
"""

import os
import pandas as pd
from datetime import datetime
import logging
from pathlib import Path

from database import get_db_manager
from database.models import (
    Stock, Price, TechnicalIndicator, SentimentData,
    OptionsData, Model, Prediction
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CSVMigrator:
    """Migrates CSV data to PostgreSQL database"""

    def __init__(self, database_url=None):
        """Initialize migrator with database connection"""
        self.db = get_db_manager(database_url)
        self.stats = {
            'stocks': 0,
            'prices': 0,
            'technical_indicators': 0,
            'sentiment_data': 0,
            'options_data': 0,
            'errors': []
        }

    def migrate_all(self):
        """Run all migration steps"""
        logger.info("=" * 70)
        logger.info("Starting CSV to Database Migration")
        logger.info("=" * 70)

        start_time = datetime.now()

        try:
            # Step 1: Migrate processed stock data (prices + technical indicators)
            if os.path.exists('processed_stock_data.csv'):
                logger.info("\n[1/3] Migrating processed stock data...")
                self.migrate_processed_stock_data('processed_stock_data.csv')
            else:
                logger.warning("processed_stock_data.csv not found, skipping")

            # Step 2: Migrate sentiment data cache
            if os.path.exists('reddit_sentiment_cache.csv'):
                logger.info("\n[2/3] Migrating Reddit sentiment data...")
                self.migrate_sentiment_data('reddit_sentiment_cache.csv', source='reddit')
            else:
                logger.warning("reddit_sentiment_cache.csv not found, skipping")

            # Step 3: Migrate options data cache
            if os.path.exists('options_volume_cache.csv'):
                logger.info("\n[3/3] Migrating options volume data...")
                self.migrate_options_data('options_volume_cache.csv')
            else:
                logger.warning("options_volume_cache.csv not found, skipping")

            # Print migration statistics
            elapsed = datetime.now() - start_time
            self.print_summary(elapsed)

        except Exception as e:
            logger.error(f"Migration failed: {e}")
            raise

    def migrate_processed_stock_data(self, csv_file):
        """
        Migrate processed_stock_data.csv to prices and technical_indicators tables

        CSV contains: Date, AAPL_Close, MSFT_Close, AAPL_RSI, etc.
        """
        logger.info(f"Reading {csv_file}...")
        df = pd.read_csv(csv_file, index_col=0, parse_dates=True)

        # Detect symbols from column names
        symbols = set()
        for col in df.columns:
            if '_' in col:
                symbol = col.split('_')[0]
                symbols.add(symbol)

        logger.info(f"Found {len(symbols)} symbols: {sorted(symbols)}")

        # Step 1: Create stock reference data
        logger.info("Creating stock reference data...")
        with self.db.get_session() as session:
            for symbol in sorted(symbols):
                # Check if stock already exists
                existing = session.query(Stock).filter_by(symbol=symbol).first()
                if not existing:
                    stock = Stock(
                        symbol=symbol,
                        name=self._get_stock_name(symbol),
                        exchange='NASDAQ',  # Assume NASDAQ for now
                        sector='Technology',  # Would need to fetch real data
                        industry='Software'
                    )
                    session.add(stock)
                    self.stats['stocks'] += 1

        logger.info(f"Created {self.stats['stocks']} stock records")

        # Step 2: Migrate price data
        logger.info("Migrating price data...")
        price_records = []

        for symbol in sorted(symbols):
            # Look for OHLCV columns
            open_col = f'{symbol}_Open'
            high_col = f'{symbol}_High'
            low_col = f'{symbol}_Low'
            close_col = f'{symbol}_Close'
            volume_col = f'{symbol}_Volume'

            # Check which columns exist
            has_ohlcv = all(col in df.columns for col in [close_col])

            if has_ohlcv:
                for date, row in df.iterrows():
                    if pd.notna(row.get(close_col)):
                        price_records.append({
                            'symbol': symbol,
                            'date': date.date(),
                            'open': row.get(open_col, row.get(close_col)),
                            'high': row.get(high_col, row.get(close_col)),
                            'low': row.get(low_col, row.get(close_col)),
                            'close': row.get(close_col),
                            'volume': int(row.get(volume_col, 0)) if pd.notna(row.get(volume_col)) else 0,
                            'adjusted_close': row.get(close_col)
                        })

        # Bulk insert prices
        if price_records:
            logger.info(f"Inserting {len(price_records)} price records...")
            with self.db.get_session() as session:
                session.bulk_insert_mappings(Price, price_records)
            self.stats['prices'] = len(price_records)
            logger.info(f"Inserted {self.stats['prices']} price records")

        # Step 3: Migrate technical indicators
        logger.info("Migrating technical indicators...")
        indicator_records = []

        for symbol in sorted(symbols):
            # Technical indicator columns
            rsi_col = f'{symbol}_RSI'
            macd_col = f'{symbol}_MACD'
            signal_col = f'{symbol}_Signal_Line'
            ma5_col = f'{symbol}_MA_5'
            ma20_col = f'{symbol}_MA_20'
            ma50_col = f'{symbol}_MA_50'
            ma200_col = f'{symbol}_MA_200'

            for date, row in df.iterrows():
                # Only create record if at least one indicator exists
                if any(pd.notna(row.get(col)) for col in [rsi_col, macd_col, ma20_col]):
                    indicator_records.append({
                        'symbol': symbol,
                        'date': date.date(),
                        'rsi': row.get(rsi_col) if pd.notna(row.get(rsi_col)) else None,
                        'macd': row.get(macd_col) if pd.notna(row.get(macd_col)) else None,
                        'macd_signal': row.get(signal_col) if pd.notna(row.get(signal_col)) else None,
                        'macd_histogram': None,  # Not in CSV
                        'ma_5': row.get(ma5_col) if pd.notna(row.get(ma5_col)) else None,
                        'ma_20': row.get(ma20_col) if pd.notna(row.get(ma20_col)) else None,
                        'ma_50': row.get(ma50_col) if pd.notna(row.get(ma50_col)) else None,
                        'ma_200': row.get(ma200_col) if pd.notna(row.get(ma200_col)) else None,
                        'volume_ma_5': None  # Not in CSV
                    })

        # Bulk insert technical indicators
        if indicator_records:
            logger.info(f"Inserting {len(indicator_records)} technical indicator records...")
            with self.db.get_session() as session:
                session.bulk_insert_mappings(TechnicalIndicator, indicator_records)
            self.stats['technical_indicators'] = len(indicator_records)
            logger.info(f"Inserted {self.stats['technical_indicators']} technical indicator records")

    def migrate_sentiment_data(self, csv_file, source='reddit'):
        """
        Migrate sentiment cache CSV to sentiment_data table

        CSV format: Date, Symbol, mean_sentiment, median_sentiment, std_sentiment, post_count
        """
        logger.info(f"Reading {csv_file}...")
        df = pd.read_csv(csv_file, parse_dates=['Date'])

        if df.empty:
            logger.warning("No sentiment data to migrate")
            return

        # Convert to records
        sentiment_records = []
        for _, row in df.iterrows():
            sentiment_records.append({
                'symbol': row['Symbol'],
                'date': row['Date'].date() if hasattr(row['Date'], 'date') else row['Date'],
                'source': source,
                'mean_sentiment': row.get('mean_sentiment'),
                'median_sentiment': row.get('median_sentiment'),
                'std_sentiment': row.get('std_sentiment'),
                'post_count': int(row.get('post_count', 0))
            })

        # Bulk insert
        if sentiment_records:
            logger.info(f"Inserting {len(sentiment_records)} sentiment records...")
            with self.db.get_session() as session:
                session.bulk_insert_mappings(SentimentData, sentiment_records)
            self.stats['sentiment_data'] = len(sentiment_records)
            logger.info(f"Inserted {self.stats['sentiment_data']} sentiment records")

    def migrate_options_data(self, csv_file):
        """
        Migrate options volume cache CSV to options_data table

        CSV format: Date, Symbol, total_volume, calls_volume, puts_volume, contract_count
        """
        logger.info(f"Reading {csv_file}...")
        df = pd.read_csv(csv_file, parse_dates=['Date'])

        if df.empty:
            logger.warning("No options data to migrate")
            return

        # Convert to records
        options_records = []
        for _, row in df.iterrows():
            options_records.append({
                'symbol': row['Symbol'],
                'date': row['Date'].date() if hasattr(row['Date'], 'date') else row['Date'],
                'total_volume': int(row.get('total_volume', 0)),
                'calls_volume': int(row.get('calls_volume', 0)),
                'puts_volume': int(row.get('puts_volume', 0)),
                'contract_count': int(row.get('contract_count', 0)),
                'implied_volatility': row.get('implied_volatility')
            })

        # Bulk insert
        if options_records:
            logger.info(f"Inserting {len(options_records)} options records...")
            with self.db.get_session() as session:
                session.bulk_insert_mappings(OptionsData, options_records)
            self.stats['options_data'] = len(options_records)
            logger.info(f"Inserted {self.stats['options_data']} options records")

    def _get_stock_name(self, symbol):
        """Get stock name from symbol (hardcoded for now)"""
        names = {
            'AAPL': 'Apple Inc.',
            'MSFT': 'Microsoft Corporation',
            'GOOGL': 'Alphabet Inc.',
            'AMZN': 'Amazon.com Inc.',
            'META': 'Meta Platforms Inc.',
            'TSLA': 'Tesla Inc.',
            'NVDA': 'NVIDIA Corporation'
        }
        return names.get(symbol, f'{symbol} Inc.')

    def print_summary(self, elapsed_time):
        """Print migration summary"""
        logger.info("\n" + "=" * 70)
        logger.info("MIGRATION SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Stocks created:              {self.stats['stocks']:,}")
        logger.info(f"Price records inserted:      {self.stats['prices']:,}")
        logger.info(f"Technical indicators:        {self.stats['technical_indicators']:,}")
        logger.info(f"Sentiment records:           {self.stats['sentiment_data']:,}")
        logger.info(f"Options records:             {self.stats['options_data']:,}")
        logger.info(f"Elapsed time:                {elapsed_time}")

        if self.stats['errors']:
            logger.warning(f"\nErrors encountered: {len(self.stats['errors'])}")
            for error in self.stats['errors'][:10]:  # Show first 10
                logger.warning(f"  - {error}")
        else:
            logger.info("\n✅ Migration completed successfully!")

        logger.info("=" * 70)


def main():
    """Run migration"""
    import sys

    # Get database URL from environment or use default
    database_url = os.environ.get('DATABASE_URL', 'postgresql://localhost:5432/trading_db')

    print(f"Database: {database_url}")
    print("\nThis will migrate CSV data to the database.")
    print("Existing data in the database will NOT be deleted.")
    print("Duplicate records will be skipped (based on unique constraints).\n")

    response = input("Continue? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Migration cancelled")
        sys.exit(0)

    # Run migration
    migrator = CSVMigrator(database_url)
    migrator.migrate_all()


if __name__ == '__main__':
    main()
