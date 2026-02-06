"""
Data Quality Validation Module
Validates stock data before use in models or trading
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class DataQualityError(Exception):
    """Raised when data quality validation fails."""
    pass


def validate_price_data(df, symbols, check_gaps=True, max_pct_change=0.20,
                       check_volume=True):
    """
    Validate price data for quality issues.

    Args:
        df: DataFrame with stock data (index should be dates)
        symbols: List of symbols to validate
        check_gaps: Whether to check for missing dates
        max_pct_change: Maximum allowed percentage change (default 20%)
        check_volume: Whether to validate volume data

    Returns:
        tuple: (is_valid, issues_list)

    Raises:
        DataQualityError: If critical issues found
    """
    issues = []

    # 1. Check for missing dates (gaps in trading days)
    if check_gaps and len(df) > 1:
        # Generate expected business days (Mon-Fri)
        date_range = pd.date_range(
            df.index.min(),
            df.index.max(),
            freq='B'  # Business days
        )
        missing_dates = date_range.difference(df.index)

        # Allow up to 10 missing days (holidays, data issues)
        if len(missing_dates) > 10:
            issue = f"Found {len(missing_dates)} missing trading days"
            issues.append(('WARNING', issue))
        elif len(missing_dates) > 0:
            issues.append(('INFO', f"Found {len(missing_dates)} missing days (likely holidays)"))

    # 2. Check each symbol's data
    for symbol in symbols:
        if symbol not in df.columns:
            issues.append(('ERROR', f"Symbol {symbol} not found in data"))
            continue

        # 2a. Check for NaN values
        nan_count = df[symbol].isna().sum()
        if nan_count > 0:
            pct = (nan_count / len(df)) * 100
            if pct > 10:
                issues.append(('ERROR', f"{symbol}: {nan_count} NaN values ({pct:.1f}%)"))
            else:
                issues.append(('WARNING', f"{symbol}: {nan_count} NaN values ({pct:.1f}%)"))

        # 2b. Check for zero or negative prices
        invalid_prices = (df[symbol] <= 0).sum()
        if invalid_prices > 0:
            issues.append(('ERROR', f"{symbol}: {invalid_prices} zero/negative prices"))

        # 2c. Check for extreme price movements
        returns = df[symbol].pct_change(fill_method=None)
        extreme_moves = returns[abs(returns) > max_pct_change]

        if len(extreme_moves) > 0:
            for date, change in extreme_moves.items():
                issues.append((
                    'WARNING',
                    f"{symbol}: {change*100:.1f}% change on {date.date()} "
                    f"(price: ${df[symbol][date]:.2f})"
                ))

        # 2d. Check for duplicate dates
        duplicates = df.index.duplicated().sum()
        if duplicates > 0:
            issues.append(('ERROR', f"Found {duplicates} duplicate dates"))

        # 2e. Check for suspicious constant prices (stuck quotes)
        if len(df) > 10:
            # Check for 5+ consecutive identical prices
            price_changes = df[symbol].diff().fillna(0)
            zero_changes = (price_changes == 0).astype(int)
            max_consecutive = zero_changes.groupby(
                (zero_changes != zero_changes.shift()).cumsum()
            ).sum().max()

            if max_consecutive > 5:
                issues.append((
                    'WARNING',
                    f"{symbol}: {max_consecutive} consecutive identical prices (stuck quote?)"
                ))

    # 3. Check volume data if requested
    if check_volume:
        for symbol in symbols:
            volume_col = f'{symbol}_Volume'
            if volume_col in df.columns:
                # Check for negative volume
                negative_vol = (df[volume_col] < 0).sum()
                if negative_vol > 0:
                    issues.append(('ERROR', f"{symbol}: {negative_vol} negative volume values"))

                # Check for zero volume (suspicious for major stocks)
                zero_vol = (df[volume_col] == 0).sum()
                if zero_vol > len(df) * 0.05:  # More than 5% of days
                    issues.append((
                        'WARNING',
                        f"{symbol}: {zero_vol} days with zero volume ({zero_vol/len(df)*100:.1f}%)"
                    ))

    # 4. Check for data recency
    if len(df) > 0:
        last_date = df.index.max()
        days_old = (datetime.now().date() - last_date.date()).days

        if days_old > 7:
            issues.append((
                'WARNING',
                f"Data is {days_old} days old (last date: {last_date.date()})"
            ))

    # Determine if validation passed
    error_count = sum(1 for level, _ in issues if level == 'ERROR')

    return error_count == 0, issues


def validate_technical_indicators(df, symbols):
    """
    Validate technical indicators are in reasonable ranges.

    Args:
        df: DataFrame with technical indicators
        symbols: List of symbols

    Returns:
        tuple: (is_valid, issues_list)
    """
    issues = []

    for symbol in symbols:
        # RSI should be 0-100
        rsi_col = f'{symbol}_RSI'
        if rsi_col in df.columns:
            invalid_rsi = ((df[rsi_col] < 0) | (df[rsi_col] > 100)).sum()
            if invalid_rsi > 0:
                issues.append((
                    'ERROR',
                    f"{symbol}: {invalid_rsi} RSI values outside 0-100 range"
                ))

        # Check MACD values are reasonable (not extreme)
        macd_col = f'{symbol}_MACD'
        if macd_col in df.columns:
            macd_std = df[macd_col].std()
            extreme_macd = (abs(df[macd_col]) > macd_std * 5).sum()
            if extreme_macd > len(df) * 0.01:  # More than 1% of points
                issues.append((
                    'WARNING',
                    f"{symbol}: {extreme_macd} extreme MACD values (>5 std dev)"
                ))

        # Moving averages should be positive
        for ma_period in ['MA5', 'MA20']:
            ma_col = f'{symbol}_{ma_period}'
            if ma_col in df.columns:
                negative_ma = (df[ma_col] < 0).sum()
                if negative_ma > 0:
                    issues.append((
                        'ERROR',
                        f"{symbol}: {negative_ma} negative {ma_period} values"
                    ))

    error_count = sum(1 for level, _ in issues if level == 'ERROR')
    return error_count == 0, issues


def print_validation_report(issues):
    """
    Log validation report at appropriate severity levels.

    Args:
        issues: List of (level, message) tuples
    """
    if not issues:
        logger.info("Data validation passed - no issues found")
        return

    logger.info("=" * 70)
    logger.info("DATA VALIDATION REPORT")
    logger.info("=" * 70)

    errors = [msg for level, msg in issues if level == 'ERROR']
    warnings = [msg for level, msg in issues if level == 'WARNING']
    infos = [msg for level, msg in issues if level == 'INFO']

    if errors:
        logger.error("ERRORS (%d):", len(errors))
        for i, error in enumerate(errors, 1):
            logger.error("  %d. %s", i, error)

    if warnings:
        logger.warning("WARNINGS (%d):", len(warnings))
        for i, warning in enumerate(warnings, 1):
            logger.warning("  %d. %s", i, warning)

    if infos:
        logger.info("INFO (%d):", len(infos))
        for i, info_msg in enumerate(infos, 1):
            logger.info("  %d. %s", i, info_msg)

    logger.info("=" * 70)

    if errors:
        logger.error("VALIDATION FAILED - Data quality issues must be fixed")
    elif warnings:
        logger.warning("VALIDATION PASSED WITH WARNINGS - Review issues above")
    else:
        logger.info("VALIDATION PASSED - Data quality is good")

    logger.info("=" * 70)


if __name__ == "__main__":
    # Example usage
    print("Data Validation Module")
    print("Import this module to validate your stock data")
    print("\nExample:")
    print("""
    from data_validation import validate_price_data, print_validation_report

    # Validate data
    is_valid, issues = validate_price_data(df, ['AAPL', 'MSFT'])

    # Print report
    print_validation_report(issues)

    # Raise error if critical issues found
    if not is_valid:
        raise DataQualityError("Data validation failed")
    """)
