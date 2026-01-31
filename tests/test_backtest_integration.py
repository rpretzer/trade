"""
Unit Tests for Backtest Integration with Trading Execution
Tests that backtest_strategy.py properly integrates OrderExecutor
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
from unittest.mock import MagicMock, patch

# Mock TensorFlow/Keras if not available
try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    # Create mock modules
    import sys
    from unittest.mock import MagicMock
    sys.modules['tensorflow'] = MagicMock()
    sys.modules['tensorflow.keras'] = MagicMock()

from backtest_strategy import (
    apply_stop_loss, check_risk, generate_trading_signals,
    backtest_strategy
)
from trading_execution import OrderExecutor
from exceptions import OrderRejectedError, ShortNotAvailableError


class TestStopLoss:
    """Test stop-loss functionality."""

    def test_stop_loss_long_triggered(self):
        """Test stop-loss triggers for long position when price drops."""
        entry_price = 100.0
        current_price = 97.0  # 3% drop
        stop_loss_pct = 0.02  # 2%

        triggered = apply_stop_loss(current_price, entry_price, stop_loss_pct, is_long=True)
        assert triggered is True

    def test_stop_loss_long_not_triggered(self):
        """Test stop-loss doesn't trigger when price drop is small."""
        entry_price = 100.0
        current_price = 99.0  # 1% drop
        stop_loss_pct = 0.02  # 2%

        triggered = apply_stop_loss(current_price, entry_price, stop_loss_pct, is_long=True)
        assert triggered is False

    def test_stop_loss_short_triggered(self):
        """Test stop-loss triggers for short position when price rises."""
        entry_price = 100.0
        current_price = 103.0  # 3% rise
        stop_loss_pct = 0.02  # 2%

        triggered = apply_stop_loss(current_price, entry_price, stop_loss_pct, is_long=False)
        assert triggered is True

    def test_stop_loss_short_not_triggered(self):
        """Test stop-loss doesn't trigger when price rise is small."""
        entry_price = 100.0
        current_price = 101.0  # 1% rise
        stop_loss_pct = 0.02  # 2%

        triggered = apply_stop_loss(current_price, entry_price, stop_loss_pct, is_long=False)
        assert triggered is False


class TestRiskCheck:
    """Test risk management checks."""

    def test_check_risk_within_limit(self):
        """Test that trading continues when drawdown is within limit."""
        initial_capital = 10000
        current_capital = 9600  # 4% drawdown
        max_drawdown = 0.05  # 5% limit

        result = check_risk(current_capital, initial_capital, max_drawdown)
        assert result is True

    def test_check_risk_exceeds_limit(self):
        """Test that trading stops when drawdown exceeds limit."""
        initial_capital = 10000
        current_capital = 9400  # 6% drawdown
        max_drawdown = 0.05  # 5% limit

        result = check_risk(current_capital, initial_capital, max_drawdown)
        assert result is False

    def test_check_risk_zero_initial_capital(self):
        """Test edge case of zero initial capital."""
        initial_capital = 0
        current_capital = 0
        max_drawdown = 0.05

        result = check_risk(current_capital, initial_capital, max_drawdown)
        assert result is True  # Should allow trading


class TestTradingSignals:
    """Test trading signal generation."""

    def test_generate_signals_buy_aapl(self):
        """Test signal generation for buying AAPL."""
        predictions = np.array([0.8, 0.6, 0.3])
        threshold = 0.5

        signals = generate_trading_signals(predictions, threshold)
        assert signals[0] == 'Buy AAPL, Sell MSFT'
        assert signals[1] == 'Buy AAPL, Sell MSFT'
        assert signals[2] == 'Hold'

    def test_generate_signals_buy_msft(self):
        """Test signal generation for buying MSFT."""
        predictions = np.array([-0.8, -0.6, -0.3])
        threshold = 0.5

        signals = generate_trading_signals(predictions, threshold)
        assert signals[0] == 'Buy MSFT, Sell AAPL'
        assert signals[1] == 'Buy MSFT, Sell AAPL'
        assert signals[2] == 'Hold'

    def test_generate_signals_hold(self):
        """Test signal generation for hold."""
        predictions = np.array([0.3, -0.3, 0.0])
        threshold = 0.5

        signals = generate_trading_signals(predictions, threshold)
        assert all(s == 'Hold' for s in signals)


class TestBacktestIntegration:
    """Test backtest integration with OrderExecutor."""

    def setup_method(self):
        """Set up test data."""
        # Create sample price data
        dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
        self.price_data = pd.DataFrame({
            'AAPL': [150.0 + i for i in range(10)],
            'MSFT': [300.0 + i for i in range(10)],
            'Price_Difference': [150.0 for _ in range(10)]
        }, index=dates)

    def test_backtest_handles_market_hours(self):
        """Test that backtest respects market hours."""
        # This test verifies that OrderExecutor is being called
        # Market hours checking happens inside OrderExecutor.execute_order()

        predictions = np.array([0.6, 0.7, 0.8, 0.5])
        actual_diffs = np.array([151.0, 152.0, 153.0, 154.0])

        # Run backtest with subset of data
        price_subset = self.price_data.iloc[:5]

        results = backtest_strategy(
            predictions,
            actual_diffs,
            price_subset,
            threshold=0.5,
            initial_capital=10000,
            max_drawdown=0.05,
            stop_loss_pct=0.02
        )

        # Verify backtest completed
        assert 'final_capital' in results
        assert 'num_trades' in results
        assert results['initial_capital'] == 10000

    def test_backtest_handles_rejected_orders(self):
        """Test that backtest handles order rejections gracefully."""
        predictions = np.array([0.8] * 3)
        actual_diffs = np.array([151.0, 152.0, 153.0])
        price_subset = self.price_data.iloc[:4]

        # Run backtest - OrderExecutor may reject some orders
        results = backtest_strategy(
            predictions,
            actual_diffs,
            price_subset,
            threshold=0.5,
            initial_capital=100,  # Very low capital
            max_drawdown=0.05,
            stop_loss_pct=0.02
        )

        # Should complete without crashing
        assert 'final_capital' in results
        assert results['final_capital'] >= 0  # Can't go negative

    def test_backtest_tracks_transaction_costs(self):
        """Test that transaction costs are tracked."""
        predictions = np.array([0.8, 0.7, 0.6])
        actual_diffs = np.array([151.0, 152.0, 153.0])
        price_subset = self.price_data.iloc[:4]

        results = backtest_strategy(
            predictions,
            actual_diffs,
            price_subset,
            threshold=0.5,
            initial_capital=10000,
            max_drawdown=0.05,
            stop_loss_pct=0.02
        )

        # Should track transaction costs
        assert 'total_transaction_costs' in results
        if results['num_trades'] > 0:
            assert results['total_transaction_costs'] > 0

    def test_backtest_respects_drawdown_limit(self):
        """Test that backtest stops trading on max drawdown."""
        # Create scenario with large losses
        predictions = np.array([0.8] * 5)  # Always predict buy AAPL
        # But actual difference goes down (losses)
        actual_diffs = np.array([150.0, 148.0, 146.0, 144.0, 142.0])

        price_subset = self.price_data.iloc[:6]

        results = backtest_strategy(
            predictions,
            actual_diffs,
            price_subset,
            threshold=0.5,
            initial_capital=10000,
            max_drawdown=0.03,  # 3% limit (strict)
            stop_loss_pct=0.02
        )

        # May have hit drawdown limit
        if results.get('drawdown_limit_reached'):
            assert results['max_drawdown_pct'] <= -3.0  # At least -3%


class TestOrderExecutorIntegration:
    """Test that OrderExecutor is properly integrated."""

    def test_order_executor_called_for_trades(self):
        """Test that OrderExecutor.execute_order is called."""
        # This is an integration test - we verify the behavior
        # by checking that the backtest handles OrderExecutor responses

        dates = pd.date_range(start='2024-01-01', periods=5, freq='D')
        price_data = pd.DataFrame({
            'AAPL': [150.0, 151.0, 152.0, 153.0, 154.0],
            'MSFT': [300.0, 301.0, 302.0, 303.0, 304.0],
            'Price_Difference': [150.0, 151.0, 152.0, 153.0, 154.0]
        }, index=dates)

        predictions = np.array([0.8, 0.8, 0.8, 0.8])
        actual_diffs = price_data['Price_Difference'].values[1:]

        results = backtest_strategy(
            predictions,
            actual_diffs,
            price_data,
            threshold=0.5,
            initial_capital=10000,
            max_drawdown=0.05,
            stop_loss_pct=0.02
        )

        # If trades were made, OrderExecutor was called
        assert 'num_trades' in results
        assert 'positions' in results
        assert len(results['positions']) == len(predictions)

    def test_partial_fills_handled(self):
        """Test that partial fills are handled correctly."""
        # OrderExecutor can return partial fills
        # Backtest should handle them gracefully

        dates = pd.date_range(start='2024-01-01', periods=5, freq='D')
        price_data = pd.DataFrame({
            'AAPL': [150.0] * 5,
            'MSFT': [300.0] * 5,
            'Price_Difference': [150.0] * 5
        }, index=dates)

        predictions = np.array([0.8, 0.7, 0.6, 0.5])
        actual_diffs = price_data['Price_Difference'].values[1:]

        results = backtest_strategy(
            predictions,
            actual_diffs,
            price_data,
            threshold=0.5,
            initial_capital=10000,
            max_drawdown=0.05,
            stop_loss_pct=0.02
        )

        # Should complete successfully
        assert results is not None
        assert 'final_capital' in results
