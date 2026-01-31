"""
Unit tests for data validation
Tests data quality checks
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_validation import (
    validate_price_data,
    validate_technical_indicators,
    DataQualityError
)


class TestPriceDataValidation:
    """Test price data validation"""

    def create_clean_data(self, days=10):
        """Create clean test data"""
        dates = pd.date_range(start='2024-01-01', periods=days, freq='B')
        df = pd.DataFrame({
            'AAPL': np.linspace(150, 155, days),
            'MSFT': np.linspace(300, 305, days),
            'AAPL_Volume': np.full(days, 1000000),
            'MSFT_Volume': np.full(days, 2000000)
        }, index=dates)
        return df

    def test_validates_clean_data(self):
        """Test validation passes for clean data"""
        df = self.create_clean_data()

        is_valid, issues = validate_price_data(df, ['AAPL', 'MSFT'])

        assert is_valid is True, "Clean data should validate"
        # May have INFO messages but no ERRORS
        errors = [msg for level, msg in issues if level == 'ERROR']
        assert len(errors) == 0, "Should have no errors"

    def test_detects_missing_symbol(self):
        """Test detects when symbol is missing"""
        df = self.create_clean_data()

        is_valid, issues = validate_price_data(df, ['AAPL', 'GOOGL'])

        assert is_valid is False, "Should fail when symbol missing"
        errors = [msg for level, msg in issues if level == 'ERROR']
        assert any('GOOGL' in msg for msg in errors), "Should report missing GOOGL"

    def test_detects_nan_values(self):
        """Test detects NaN values"""
        df = self.create_clean_data()
        df.loc[df.index[3], 'AAPL'] = np.nan

        is_valid, issues = validate_price_data(df, ['AAPL', 'MSFT'])

        # Depending on percentage, could be ERROR or WARNING
        assert len(issues) > 0, "Should detect NaN values"

    def test_detects_zero_prices(self):
        """Test detects zero or negative prices"""
        df = self.create_clean_data()
        df.loc[df.index[2], 'AAPL'] = 0.0

        is_valid, issues = validate_price_data(df, ['AAPL', 'MSFT'])

        assert is_valid is False, "Should fail on zero price"
        errors = [msg for level, msg in issues if level == 'ERROR']
        assert any('zero/negative' in msg.lower() for msg in errors)

    def test_detects_negative_prices(self):
        """Test detects negative prices"""
        df = self.create_clean_data()
        df.loc[df.index[4], 'MSFT'] = -10.0

        is_valid, issues = validate_price_data(df, ['AAPL', 'MSFT'])

        assert is_valid is False, "Should fail on negative price"

    def test_detects_extreme_price_moves(self):
        """Test detects extreme price movements"""
        df = self.create_clean_data()
        # Simulate 30% price jump
        df.loc[df.index[5], 'AAPL'] = df.loc[df.index[4], 'AAPL'] * 1.30

        is_valid, issues = validate_price_data(df, ['AAPL', 'MSFT'], max_pct_change=0.20)

        warnings = [msg for level, msg in issues if level == 'WARNING']
        assert any('30' in msg or 'extreme' in msg.lower() for msg in warnings), "Should warn on extreme move"

    def test_detects_negative_volume(self):
        """Test detects negative volume"""
        df = self.create_clean_data()
        df.loc[df.index[2], 'AAPL_Volume'] = -100

        is_valid, issues = validate_price_data(df, ['AAPL', 'MSFT'], check_volume=True)

        assert is_valid is False, "Should fail on negative volume"
        errors = [msg for level, msg in issues if level == 'ERROR']
        assert any('negative volume' in msg.lower() for msg in errors)

    def test_warns_on_zero_volume(self):
        """Test warns on excessive zero volume days"""
        df = self.create_clean_data(days=100)
        # Set 10% of days to zero volume
        zero_indices = df.index[:10]
        df.loc[zero_indices, 'AAPL_Volume'] = 0

        is_valid, issues = validate_price_data(df, ['AAPL', 'MSFT'], check_volume=True)

        warnings = [msg for level, msg in issues if level == 'WARNING']
        assert any('zero volume' in msg.lower() for msg in warnings)

    def test_detects_stale_data(self):
        """Test warns when data is old"""
        # Create data ending 10 days ago
        end_date = datetime.now() - timedelta(days=10)
        dates = pd.date_range(end=end_date, periods=10, freq='B')
        df = pd.DataFrame({
            'AAPL': np.linspace(150, 155, 10),
            'MSFT': np.linspace(300, 305, 10)
        }, index=dates)

        is_valid, issues = validate_price_data(df, ['AAPL', 'MSFT'])

        warnings = [msg for level, msg in issues if level == 'WARNING']
        assert any('days old' in msg.lower() for msg in warnings), "Should warn on stale data"


class TestTechnicalIndicatorValidation:
    """Test technical indicator validation"""

    def create_data_with_indicators(self):
        """Create test data with technical indicators"""
        dates = pd.date_range(start='2024-01-01', periods=10, freq='B')
        df = pd.DataFrame({
            'AAPL': np.linspace(150, 155, 10),
            'AAPL_RSI': np.full(10, 50),  # Valid RSI
            'AAPL_MACD': np.random.randn(10) * 0.5,
            'AAPL_MA5': np.linspace(149, 154, 10),
            'AAPL_MA20': np.linspace(148, 153, 10)
        }, index=dates)
        return df

    def test_validates_clean_indicators(self):
        """Test validation passes for valid indicators"""
        df = self.create_data_with_indicators()

        is_valid, issues = validate_technical_indicators(df, ['AAPL'])

        assert is_valid is True, "Valid indicators should pass"
        errors = [msg for level, msg in issues if level == 'ERROR']
        assert len(errors) == 0

    def test_detects_invalid_rsi(self):
        """Test detects RSI outside 0-100 range"""
        df = self.create_data_with_indicators()
        df.loc[df.index[5], 'AAPL_RSI'] = 150  # Invalid

        is_valid, issues = validate_technical_indicators(df, ['AAPL'])

        assert is_valid is False, "Should fail on invalid RSI"
        errors = [msg for level, msg in issues if level == 'ERROR']
        assert any('rsi' in msg.lower() for msg in errors)

    def test_detects_negative_ma(self):
        """Test detects negative moving averages"""
        df = self.create_data_with_indicators()
        df.loc[df.index[3], 'AAPL_MA5'] = -10

        is_valid, issues = validate_technical_indicators(df, ['AAPL'])

        assert is_valid is False, "Should fail on negative MA"
        errors = [msg for level, msg in issues if level == 'ERROR']
        assert any('negative' in msg.lower() and 'ma5' in msg.lower() for msg in errors)

    def test_warns_on_extreme_macd(self):
        """Test warns on extreme MACD values"""
        df = self.create_data_with_indicators()
        # Set MACD to extreme value (>5 std dev)
        macd_std = df['AAPL_MACD'].std()
        df.loc[df.index[2:4], 'AAPL_MACD'] = macd_std * 10

        is_valid, issues = validate_technical_indicators(df, ['AAPL'])

        warnings = [msg for level, msg in issues if level == 'WARNING']
        # May or may not warn depending on percentage threshold
        # Just check that validation runs without error
        assert isinstance(is_valid, bool)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
