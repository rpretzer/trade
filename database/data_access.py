"""
Database Access Layer

Provides high-level functions to replace CSV read/write operations.
Use these functions instead of pd.read_csv() and df.to_csv().
"""

import pandas as pd
from datetime import datetime, date, timedelta
from typing import List, Optional, Dict
import logging

from database import get_db_manager
from database.models import (
    Stock, Price, TechnicalIndicator, SentimentData,
    OptionsData, Model, ModelMetric, Prediction,
    Order, Trade, Position
)
from sqlalchemy import and_, or_, func

logger = logging.getLogger(__name__)


class DataAccess:
    """High-level database access for trading system"""

    def __init__(self, database_url=None):
        """Initialize data access layer"""
        self.db = get_db_manager(database_url)

    def load_processed_stock_data(
        self,
        symbols: List[str],
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        include_technical_indicators: bool = True
    ) -> pd.DataFrame:
        """
        Load stock price data with technical indicators (replaces CSV loading)

        This replaces: pd.read_csv('processed_stock_data.csv')

        Args:
            symbols: List of stock symbols (e.g., ['AAPL', 'MSFT'])
            start_date: Start date (default: 1 year ago)
            end_date: End date (default: today)
            include_technical_indicators: Include RSI, MACD, MAs

        Returns:
            DataFrame with Date index and columns like:
            AAPL_Close, MSFT_Close, AAPL_RSI, MSFT_RSI, etc.
        """
        if start_date is None:
            start_date = date.today() - timedelta(days=365)
        if end_date is None:
            end_date = date.today()

        logger.info(f"Loading stock data for {symbols} from {start_date} to {end_date}")

        with self.db.get_session() as session:
            # Load prices
            prices = session.query(Price).filter(
                and_(
                    Price.symbol.in_(symbols),
                    Price.date >= start_date,
                    Price.date <= end_date
                )
            ).all()

            # Convert to DataFrame
            price_data = []
            for p in prices:
                price_data.append({
                    'Date': p.date,
                    'Symbol': p.symbol,
                    'Open': float(p.open),
                    'High': float(p.high),
                    'Low': float(p.low),
                    'Close': float(p.close),
                    'Volume': int(p.volume)
                })

            if not price_data:
                logger.warning(f"No price data found for {symbols}")
                return pd.DataFrame()

            df_prices = pd.DataFrame(price_data)

            # Pivot to wide format (Date as index, Symbol_Column format)
            df_wide = None
            for symbol in symbols:
                symbol_df = df_prices[df_prices['Symbol'] == symbol].set_index('Date')
                symbol_df = symbol_df.drop('Symbol', axis=1)
                symbol_df.columns = [f'{symbol}_{col}' for col in symbol_df.columns]

                if df_wide is None:
                    df_wide = symbol_df
                else:
                    df_wide = df_wide.join(symbol_df, how='outer')

            # Load technical indicators if requested
            if include_technical_indicators:
                indicators = session.query(TechnicalIndicator).filter(
                    and_(
                        TechnicalIndicator.symbol.in_(symbols),
                        TechnicalIndicator.date >= start_date,
                        TechnicalIndicator.date <= end_date
                    )
                ).all()

                indicator_data = []
                for ind in indicators:
                    indicator_data.append({
                        'Date': ind.date,
                        'Symbol': ind.symbol,
                        'RSI': float(ind.rsi) if ind.rsi else None,
                        'MACD': float(ind.macd) if ind.macd else None,
                        'Signal_Line': float(ind.macd_signal) if ind.macd_signal else None,
                        'MA_5': float(ind.ma_5) if ind.ma_5 else None,
                        'MA_20': float(ind.ma_20) if ind.ma_20 else None,
                        'MA_50': float(ind.ma_50) if ind.ma_50 else None,
                        'MA_200': float(ind.ma_200) if ind.ma_200 else None
                    })

                if indicator_data:
                    df_indicators = pd.DataFrame(indicator_data)

                    # Pivot indicators
                    for symbol in symbols:
                        symbol_ind = df_indicators[df_indicators['Symbol'] == symbol].set_index('Date')
                        symbol_ind = symbol_ind.drop('Symbol', axis=1)
                        symbol_ind.columns = [f'{symbol}_{col}' for col in symbol_ind.columns]
                        df_wide = df_wide.join(symbol_ind, how='outer')

            # Sort by date
            df_wide = df_wide.sort_index()

            logger.info(f"Loaded {len(df_wide)} rows, {len(df_wide.columns)} columns")
            return df_wide

    def save_processed_stock_data(self, df: pd.DataFrame, symbols: List[str]):
        """
        Save processed stock data to database (replaces CSV saving)

        This replaces: df.to_csv('processed_stock_data.csv')

        Args:
            df: DataFrame with Date index and Symbol_Column format
            symbols: List of symbols to save
        """
        logger.info(f"Saving stock data for {symbols}")

        price_records = []
        indicator_records = []

        for symbol in symbols:
            # Extract price columns
            close_col = f'{symbol}_Close'
            if close_col not in df.columns:
                logger.warning(f"No Close data for {symbol}, skipping")
                continue

            for date_idx, row in df.iterrows():
                if pd.notna(row.get(close_col)):
                    # Price record
                    price_records.append({
                        'symbol': symbol,
                        'date': date_idx.date() if hasattr(date_idx, 'date') else date_idx,
                        'open': row.get(f'{symbol}_Open', row.get(close_col)),
                        'high': row.get(f'{symbol}_High', row.get(close_col)),
                        'low': row.get(f'{symbol}_Low', row.get(close_col)),
                        'close': row.get(close_col),
                        'volume': int(row.get(f'{symbol}_Volume', 0)),
                        'adjusted_close': row.get(close_col)
                    })

                    # Technical indicator record
                    if any(col in df.columns for col in [f'{symbol}_RSI', f'{symbol}_MACD', f'{symbol}_MA_20']):
                        indicator_records.append({
                            'symbol': symbol,
                            'date': date_idx.date() if hasattr(date_idx, 'date') else date_idx,
                            'rsi': row.get(f'{symbol}_RSI'),
                            'macd': row.get(f'{symbol}_MACD'),
                            'macd_signal': row.get(f'{symbol}_Signal_Line'),
                            'macd_histogram': None,
                            'ma_5': row.get(f'{symbol}_MA_5'),
                            'ma_20': row.get(f'{symbol}_MA_20'),
                            'ma_50': row.get(f'{symbol}_MA_50'),
                            'ma_200': row.get(f'{symbol}_MA_200'),
                            'volume_ma_5': None
                        })

        # Bulk insert prices (update on conflict)
        if price_records:
            logger.info(f"Saving {len(price_records)} price records...")
            with self.db.get_session() as session:
                # Delete existing records for these symbols/dates to avoid duplicates
                dates = list(set(r['date'] for r in price_records))
                session.query(Price).filter(
                    and_(
                        Price.symbol.in_(symbols),
                        Price.date.in_(dates)
                    )
                ).delete(synchronize_session=False)

                # Insert new records
                session.bulk_insert_mappings(Price, price_records)

        # Bulk insert technical indicators
        if indicator_records:
            logger.info(f"Saving {len(indicator_records)} technical indicator records...")
            with self.db.get_session() as session:
                # Delete existing records
                dates = list(set(r['date'] for r in indicator_records))
                session.query(TechnicalIndicator).filter(
                    and_(
                        TechnicalIndicator.symbol.in_(symbols),
                        TechnicalIndicator.date.in_(dates)
                    )
                ).delete(synchronize_session=False)

                # Insert new records
                session.bulk_insert_mappings(TechnicalIndicator, indicator_records)

        logger.info("Stock data saved successfully")

    def load_sentiment_data(
        self,
        symbols: List[str],
        source: str = 'reddit',
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> pd.DataFrame:
        """
        Load sentiment data (replaces CSV cache loading)

        Returns:
            DataFrame with Date, Symbol, mean_sentiment, etc.
        """
        if start_date is None:
            start_date = date.today() - timedelta(days=365)
        if end_date is None:
            end_date = date.today()

        with self.db.get_session() as session:
            sentiment = session.query(SentimentData).filter(
                and_(
                    SentimentData.symbol.in_(symbols),
                    SentimentData.source == source,
                    SentimentData.date >= start_date,
                    SentimentData.date <= end_date
                )
            ).all()

            data = []
            for s in sentiment:
                data.append({
                    'Date': s.date,
                    'Symbol': s.symbol,
                    'mean_sentiment': float(s.mean_sentiment) if s.mean_sentiment else None,
                    'median_sentiment': float(s.median_sentiment) if s.median_sentiment else None,
                    'std_sentiment': float(s.std_sentiment) if s.std_sentiment else None,
                    'post_count': s.post_count
                })

            return pd.DataFrame(data)

    def save_sentiment_data(self, df: pd.DataFrame, source: str = 'reddit'):
        """
        Save sentiment data to database (replaces CSV cache)

        Args:
            df: DataFrame with Date, Symbol, mean_sentiment, etc.
            source: Data source (reddit, twitter, etc.)
        """
        records = []
        for _, row in df.iterrows():
            records.append({
                'symbol': row['Symbol'],
                'date': row['Date'].date() if hasattr(row['Date'], 'date') else row['Date'],
                'source': source,
                'mean_sentiment': row.get('mean_sentiment'),
                'median_sentiment': row.get('median_sentiment'),
                'std_sentiment': row.get('std_sentiment'),
                'post_count': int(row.get('post_count', 0))
            })

        if records:
            with self.db.get_session() as session:
                # Delete existing records
                symbols = list(set(r['symbol'] for r in records))
                dates = list(set(r['date'] for r in records))
                session.query(SentimentData).filter(
                    and_(
                        SentimentData.symbol.in_(symbols),
                        SentimentData.source == source,
                        SentimentData.date.in_(dates)
                    )
                ).delete(synchronize_session=False)

                # Insert new records
                session.bulk_insert_mappings(SentimentData, records)

    def load_options_data(
        self,
        symbols: List[str],
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> pd.DataFrame:
        """Load options volume data"""
        if start_date is None:
            start_date = date.today() - timedelta(days=365)
        if end_date is None:
            end_date = date.today()

        with self.db.get_session() as session:
            options = session.query(OptionsData).filter(
                and_(
                    OptionsData.symbol.in_(symbols),
                    OptionsData.date >= start_date,
                    OptionsData.date <= end_date
                )
            ).all()

            data = []
            for o in options:
                data.append({
                    'Date': o.date,
                    'Symbol': o.symbol,
                    'total_volume': o.total_volume,
                    'calls_volume': o.calls_volume,
                    'puts_volume': o.puts_volume,
                    'contract_count': o.contract_count,
                    'implied_volatility': float(o.implied_volatility) if o.implied_volatility else None
                })

            return pd.DataFrame(data)

    def save_options_data(self, df: pd.DataFrame):
        """Save options volume data to database"""
        records = []
        for _, row in df.iterrows():
            records.append({
                'symbol': row['Symbol'],
                'date': row['Date'].date() if hasattr(row['Date'], 'date') else row['Date'],
                'total_volume': int(row.get('total_volume', 0)),
                'calls_volume': int(row.get('calls_volume', 0)),
                'puts_volume': int(row.get('puts_volume', 0)),
                'contract_count': int(row.get('contract_count', 0)),
                'implied_volatility': row.get('implied_volatility')
            })

        if records:
            with self.db.get_session() as session:
                # Delete existing records
                symbols = list(set(r['symbol'] for r in records))
                dates = list(set(r['date'] for r in records))
                session.query(OptionsData).filter(
                    and_(
                        OptionsData.symbol.in_(symbols),
                        OptionsData.date.in_(dates)
                    )
                ).delete(synchronize_session=False)

                # Insert new records
                session.bulk_insert_mappings(OptionsData, records)


# Global data access instance
_data_access = None


def get_data_access(database_url=None) -> DataAccess:
    """Get global DataAccess instance"""
    global _data_access
    if _data_access is None:
        _data_access = DataAccess(database_url)
    return _data_access
