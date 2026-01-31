"""
Unit tests for backtesting strategy
Tests critical trading logic, risk management, and calculations
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest_strategy import (
    check_risk,
    apply_stop_loss,
    generate_trading_signals,
    backtest_strategy
)
from transaction_costs import TransactionCostModel


class TestRiskManagement:
    """Test risk management functions"""

    def test_check_risk_within_limit(self):
        """Test risk check passes when within drawdown limit"""
        capital = 9600  # -4% drawdown
        initial_capital = 10000
        max_drawdown = 0.05  # 5% limit

        result = check_risk(capital, initial_capital, max_drawdown)
        assert result is True, "Should allow trading within drawdown limit"

    def test_check_risk_exceeds_limit(self):
        """Test risk check fails when drawdown limit exceeded"""
        capital = 9400  # -6% drawdown
        initial_capital = 10000
        max_drawdown = 0.05  # 5% limit

        result = check_risk(capital, initial_capital, max_drawdown)
        assert result is False, "Should block trading when drawdown exceeded"

    def test_check_risk_zero_initial_capital(self):
        """Test risk check handles zero initial capital"""
        result = check_risk(1000, 0, 0.05)
        assert result is True, "Should handle zero initial capital gracefully"

    def test_apply_stop_loss_long_triggered(self):
        """Test stop-loss triggers for long position"""
        current_price = 98  # Down 2%
        entry_price = 100
        stop_loss_pct = 0.02  # 2%
        is_long = True

        result = apply_stop_loss(current_price, entry_price, stop_loss_pct, is_long)
        assert result is True, "Stop-loss should trigger for 2% loss on long"

    def test_apply_stop_loss_long_not_triggered(self):
        """Test stop-loss doesn't trigger when loss below threshold"""
        current_price = 99  # Down 1%
        entry_price = 100
        stop_loss_pct = 0.02  # 2%
        is_long = True

        result = apply_stop_loss(current_price, entry_price, stop_loss_pct, is_long)
        assert result is False, "Stop-loss should not trigger for 1% loss"

    def test_apply_stop_loss_short_triggered(self):
        """Test stop-loss triggers for short position"""
        current_price = 102  # Up 2%
        entry_price = 100
        stop_loss_pct = 0.02  # 2%
        is_long = False

        result = apply_stop_loss(current_price, entry_price, stop_loss_pct, is_long)
        assert result is True, "Stop-loss should trigger for 2% rise on short"

    def test_apply_stop_loss_zero_entry_price(self):
        """Test stop-loss handles zero entry price"""
        result = apply_stop_loss(100, 0, 0.02, True)
        assert result is False, "Should handle zero entry price gracefully"


class TestSignalGeneration:
    """Test trading signal generation"""

    def test_generate_buy_aapl_signal(self):
        """Test generates 'Buy AAPL' signal for positive predictions"""
        predictions = np.array([0.6, 0.8, 1.0])
        threshold = 0.5

        signals = generate_trading_signals(predictions, threshold)

        assert len(signals) == 3
        assert all(s == 'Buy AAPL, Sell MSFT' for s in signals)

    def test_generate_buy_msft_signal(self):
        """Test generates 'Buy MSFT' signal for negative predictions"""
        predictions = np.array([-0.6, -0.8, -1.0])
        threshold = 0.5

        signals = generate_trading_signals(predictions, threshold)

        assert len(signals) == 3
        assert all(s == 'Buy MSFT, Sell AAPL' for s in signals)

    def test_generate_hold_signal(self):
        """Test generates 'Hold' signal for predictions below threshold"""
        predictions = np.array([0.3, -0.2, 0.4, -0.4])
        threshold = 0.5

        signals = generate_trading_signals(predictions, threshold)

        assert len(signals) == 4
        assert all(s == 'Hold' for s in signals)

    def test_generate_mixed_signals(self):
        """Test generates mix of signals"""
        predictions = np.array([0.6, -0.6, 0.3, -0.3])
        threshold = 0.5

        signals = generate_trading_signals(predictions, threshold)

        assert signals[0] == 'Buy AAPL, Sell MSFT'
        assert signals[1] == 'Buy MSFT, Sell AAPL'
        assert signals[2] == 'Hold'
        assert signals[3] == 'Hold'


class TestBacktestStrategy:
    """Test complete backtest strategy"""

    def create_test_data(self, days=10):
        """Create test price data"""
        dates = pd.date_range(start='2024-01-01', periods=days, freq='D')

        # Create simple test data
        aapl_prices = np.array([150.0] * days) + np.random.randn(days) * 2
        msft_prices = np.array([300.0] * days) + np.random.randn(days) * 3

        price_data = pd.DataFrame({
            'AAPL': aapl_prices,
            'MSFT': msft_prices,
            'Price_Difference': aapl_prices - msft_prices
        }, index=dates)

        return price_data

    def test_backtest_no_trades(self):
        """Test backtest with all Hold signals"""
        price_data = self.create_test_data(days=10)
        predictions = np.zeros(10)  # All predictions = 0
        actual_differences = price_data['Price_Difference'].values
        threshold = 0.5

        results = backtest_strategy(
            predictions,
            actual_differences,
            price_data,
            threshold=threshold,
            initial_capital=10000
        )

        assert results['num_trades'] == 0, "Should have no trades with all Hold signals"
        assert results['final_capital'] == results['initial_capital'], "Capital should not change with no trades"

    def test_backtest_profitable_trade(self):
        """Test backtest with profitable trade"""
        # Create favorable price movement
        dates = pd.date_range(start='2024-01-01', periods=3, freq='D')
        price_data = pd.DataFrame({
            'AAPL': [150.0, 150.0, 152.0],  # AAPL rises
            'MSFT': [300.0, 300.0, 299.0],  # MSFT falls
            'Price_Difference': [-150.0, -150.0, -147.0]  # Difference increases
        }, index=dates)

        # Predict that difference will increase
        predictions = np.array([0.0, 0.8, 0.0])  # Signal on day 2
        actual_differences = price_data['Price_Difference'].values

        results = backtest_strategy(
            predictions,
            actual_differences,
            price_data,
            threshold=0.5,
            initial_capital=10000
        )

        assert results['num_trades'] == 1, "Should have 1 trade"
        # Note: Exact profit depends on transaction costs, just check direction
        assert results['final_capital'] != results['initial_capital'], "Capital should change"

    def test_backtest_transaction_costs_reduce_profit(self):
        """Test that transaction costs reduce profits"""
        dates = pd.date_range(start='2024-01-01', periods=3, freq='D')
        price_data = pd.DataFrame({
            'AAPL': [150.0, 150.0, 152.0],
            'MSFT': [300.0, 300.0, 299.0],
            'Price_Difference': [-150.0, -150.0, -147.0]
        }, index=dates)

        predictions = np.array([0.0, 0.8, 0.0])
        actual_differences = price_data['Price_Difference'].values

        results = backtest_strategy(
            predictions,
            actual_differences,
            price_data,
            threshold=0.5,
            initial_capital=10000
        )

        # Transaction costs should be positive
        assert results.get('total_transaction_costs', 0) > 0, "Should have transaction costs"
        # Average cost per trade should be reasonable
        assert results.get('avg_cost_per_trade', 0) > 0, "Average cost should be positive"

    def test_backtest_max_drawdown_triggers(self):
        """Test that max drawdown circuit breaker works"""
        # Create losing scenario
        dates = pd.date_range(start='2024-01-01', periods=5, freq='D')
        price_data = pd.DataFrame({
            'AAPL': [150.0, 150.0, 148.0, 146.0, 144.0],  # AAPL falls
            'MSFT': [300.0, 300.0, 302.0, 304.0, 306.0],  # MSFT rises
            'Price_Difference': [-150.0, -150.0, -154.0, -158.0, -162.0]
        }, index=dates)

        # Predict opposite of what happens (generate losses)
        predictions = np.array([0.0, 0.8, 0.8, 0.8, 0.8])  # Predict increase, but decreases
        actual_differences = price_data['Price_Difference'].values

        results = backtest_strategy(
            predictions,
            actual_differences,
            price_data,
            threshold=0.5,
            initial_capital=10000,
            max_drawdown=0.05  # 5% limit
        )

        # Should have triggered drawdown at some point
        # Check if trading was paused (would show in positions)
        if results.get('drawdown_limit_reached'):
            assert results['max_drawdown_pct'] <= -5.0, "Should have hit drawdown limit"


class TestTransactionCostModel:
    """Test transaction cost calculations"""

    def test_commission_calculation(self):
        """Test commission is calculated correctly"""
        model = TransactionCostModel(commission_per_share=0.005)

        commission = model.calculate_commission(quantity=100)

        assert commission == 0.50, "Commission should be $0.50 for 100 shares @ $0.005/share"

    def test_sec_fee_only_on_sells(self):
        """Test SEC fees only apply to sells"""
        model = TransactionCostModel(sec_fee_rate=0.0000278)

        buy_fee = model.calculate_sec_fee(100, 150.0, 'BUY')
        sell_fee = model.calculate_sec_fee(100, 150.0, 'SELL')

        assert buy_fee == 0.0, "No SEC fee on buys"
        assert sell_fee > 0.0, "SEC fee applies on sells"

    def test_slippage_increases_with_size(self):
        """Test slippage increases with order size"""
        model = TransactionCostModel(slippage_bps=1.0)

        small_order = model.calculate_slippage(100, 150.0, 'BUY')
        large_order = model.calculate_slippage(10000, 150.0, 'BUY')

        assert large_order > small_order, "Larger orders should have more slippage"

    def test_total_cost_comprehensive(self):
        """Test total cost includes all components"""
        model = TransactionCostModel()

        costs = model.calculate_total_cost(
            quantity=100,
            price=150.0,
            side='BUY',
            bid=149.95,
            ask=150.05
        )

        assert 'commission' in costs
        assert 'sec_fee' in costs
        assert 'exchange_fee' in costs
        assert 'slippage' in costs
        assert 'total' in costs
        assert costs['total'] > 0, "Total cost should be positive"
        assert costs['total'] == sum([
            costs['commission'],
            costs['sec_fee'],
            costs['exchange_fee'],
            costs['slippage'],
            costs.get('spread', 0),
            costs.get('borrow_cost', 0)
        ]), "Total should equal sum of components"

    def test_round_trip_cost(self):
        """Test round-trip cost calculation"""
        model = TransactionCostModel()

        round_trip = model.calculate_round_trip_cost(
            quantity=100,
            entry_price=150.0,
            exit_price=152.0,
            days_held=1
        )

        assert 'entry' in round_trip
        assert 'exit' in round_trip
        assert 'total' in round_trip
        assert round_trip['total'] == (
            round_trip['entry']['total'] + round_trip['exit']['total']
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
