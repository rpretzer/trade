"""
Unit Tests for Trading Execution Module
Tests Kelly Criterion, market impact, partial fills, and realistic execution
"""

import pytest
import numpy as np
from datetime import datetime, time
from trading_execution import (
    KellyCriterion, PositionSizer, MarketImpact, OrderExecutor,
    OrderSide, OrderStatus, MarketHours, BorrowCosts, MarginConfig
)


class TestKellyCriterion:
    """Test Kelly Criterion calculations."""

    def test_full_kelly_positive_edge(self):
        """Test Kelly with positive edge."""
        # Win rate 55%, avg win 2%, avg loss 1%
        kelly = KellyCriterion.calculate_full_kelly(
            win_rate=0.55,
            avg_win=0.02,
            avg_loss=0.01
        )

        # Should be positive (we have an edge)
        assert kelly > 0

        # For 55% win rate with 2:1 payoff, Kelly should be around 27.5%
        assert 0.20 < kelly < 0.35

    def test_full_kelly_no_edge(self):
        """Test Kelly with no edge (fair game)."""
        # 50% win rate, equal wins and losses
        kelly = KellyCriterion.calculate_full_kelly(
            win_rate=0.50,
            avg_win=0.01,
            avg_loss=0.01
        )

        # Should be 0 (no edge)
        assert kelly == pytest.approx(0.0, abs=0.01)

    def test_full_kelly_negative_edge(self):
        """Test Kelly with negative edge (losing game)."""
        # 40% win rate, avg loss > avg win
        kelly = KellyCriterion.calculate_full_kelly(
            win_rate=0.40,
            avg_win=0.01,
            avg_loss=0.015
        )

        # Should be 0 (don't play)
        assert kelly == 0.0

    def test_fractional_kelly(self):
        """Test fractional Kelly (quarter Kelly)."""
        full_kelly = KellyCriterion.calculate_full_kelly(
            win_rate=0.60,
            avg_win=0.02,
            avg_loss=0.01
        )

        quarter_kelly = KellyCriterion.calculate_fractional_kelly(
            win_rate=0.60,
            avg_win=0.02,
            avg_loss=0.01,
            kelly_fraction=0.25
        )

        # Quarter Kelly should be 25% of full Kelly
        assert quarter_kelly == pytest.approx(full_kelly * 0.25)

    def test_kelly_from_sharpe(self):
        """Test Kelly approximation from Sharpe ratio."""
        # Sharpe of 1.0
        kelly = KellyCriterion.calculate_from_sharpe(
            sharpe_ratio=1.0,
            kelly_fraction=0.25
        )

        # For Sharpe=1, full Kelly ≈ 1² = 1, quarter Kelly = 0.25
        assert kelly == pytest.approx(0.25, abs=0.01)

    def test_kelly_invalid_inputs(self):
        """Test Kelly with invalid inputs."""
        # Negative win rate
        kelly = KellyCriterion.calculate_full_kelly(-0.5, 0.02, 0.01)
        assert kelly == 0.0

        # Zero average win
        kelly = KellyCriterion.calculate_full_kelly(0.6, 0.0, 0.01)
        assert kelly == 0.0

        # Win rate > 1
        kelly = KellyCriterion.calculate_full_kelly(1.5, 0.02, 0.01)
        assert kelly == 0.0


class TestPositionSizer:
    """Test position sizing algorithms."""

    def test_kelly_sizing(self):
        """Test Kelly Criterion position sizing."""
        shares = PositionSizer.kelly_sizing(
            capital=100000,
            price=150.0,
            win_rate=0.55,
            avg_win=0.02,
            avg_loss=0.01,
            kelly_fraction=0.25,
            max_position_pct=0.15
        )

        # Should return some shares
        assert shares > 0

        # Should not exceed max position
        max_shares = int(100000 * 0.15 / 150)
        assert shares <= max_shares

    def test_kelly_sizing_capped(self):
        """Test that Kelly sizing is capped at max position."""
        # Very high win rate (unrealistic but for testing)
        shares = PositionSizer.kelly_sizing(
            capital=100000,
            price=100.0,
            win_rate=0.90,
            avg_win=0.05,
            avg_loss=0.01,
            kelly_fraction=1.0,  # Full Kelly
            max_position_pct=0.10  # But cap at 10%
        )

        # Should be capped at 10% of capital
        expected_max = int(100000 * 0.10 / 100)
        assert shares == expected_max

    def test_volatility_adjusted_sizing(self):
        """Test volatility-adjusted position sizing."""
        # Low volatility (half target) → should get larger position
        shares_low_vol = PositionSizer.volatility_adjusted_sizing(
            capital=100000,
            price=150.0,
            current_volatility=0.01,  # 1% daily vol
            target_volatility=0.02,
            base_position_pct=0.10
        )

        # High volatility (double target) → should get smaller position
        shares_high_vol = PositionSizer.volatility_adjusted_sizing(
            capital=100000,
            price=150.0,
            current_volatility=0.04,  # 4% daily vol
            target_volatility=0.02,
            base_position_pct=0.10
        )

        # Low vol should result in more shares
        assert shares_low_vol > shares_high_vol

    def test_volatility_sizing_bounds(self):
        """Test volatility sizing respects min/max bounds."""
        # Extremely low volatility
        shares = PositionSizer.volatility_adjusted_sizing(
            capital=100000,
            price=100.0,
            current_volatility=0.001,  # 0.1% vol (very low)
            target_volatility=0.02,
            base_position_pct=0.10
        )

        # Should be capped at 25% max
        max_shares = int(100000 * 0.25 / 100)
        assert shares <= max_shares


class TestMarketImpact:
    """Test market impact calculations."""

    def test_impact_small_order(self):
        """Test impact of small order (low market impact)."""
        impact = MarketImpact.calculate_impact(
            quantity=100,
            average_daily_volume=1000000,  # 0.01% of volume
            spread=0.01,
            price=150.0
        )

        # Small order should have minimal impact
        assert impact < 0.001  # Less than 0.1%

    def test_impact_large_order(self):
        """Test impact of large order (high market impact)."""
        impact = MarketImpact.calculate_impact(
            quantity=100000,
            average_daily_volume=1000000,  # 10% of volume
            spread=0.02,
            price=150.0
        )

        # Large order should have measurable impact
        # sqrt(0.1) * 0.1 * 0.02 ≈ 0.0006
        assert impact > 0.0003  # More than 0.03%

    def test_impact_scales_with_volume(self):
        """Test that impact scales with order size."""
        impact_small = MarketImpact.calculate_impact(
            quantity=1000,
            average_daily_volume=1000000,
            spread=0.01,
            price=150.0
        )

        impact_large = MarketImpact.calculate_impact(
            quantity=10000,
            average_daily_volume=1000000,
            spread=0.01,
            price=150.0
        )

        # Larger order should have more impact
        assert impact_large > impact_small

    def test_apply_impact_buy(self):
        """Test that buy orders increase execution price."""
        quoted_price = 150.0

        execution_price = MarketImpact.apply_impact(
            price=quoted_price,
            quantity=10000,
            side=OrderSide.BUY,
            average_daily_volume=1000000,
            spread=0.02
        )

        # Buy should push price up
        assert execution_price > quoted_price

    def test_apply_impact_sell(self):
        """Test that sell orders decrease execution price."""
        quoted_price = 150.0

        execution_price = MarketImpact.apply_impact(
            price=quoted_price,
            quantity=10000,
            side=OrderSide.SELL,
            average_daily_volume=1000000,
            spread=0.02
        )

        # Sell should push price down
        assert execution_price < quoted_price


class TestOrderExecutor:
    """Test realistic order execution."""

    def test_market_hours_regular_hours(self):
        """Test market hours detection during regular hours."""
        executor = OrderExecutor()

        # 10 AM (market open)
        timestamp = datetime(2026, 1, 30, 10, 0)
        assert executor.is_market_open(timestamp) is True

        # 3 PM (market open)
        timestamp = datetime(2026, 1, 30, 15, 0)
        assert executor.is_market_open(timestamp) is True

    def test_market_hours_closed(self):
        """Test market hours detection when closed."""
        executor = OrderExecutor(allow_after_hours=False)

        # 8 AM (before market)
        timestamp = datetime(2026, 1, 30, 8, 0)
        assert executor.is_market_open(timestamp) is False

        # 5 PM (after market)
        timestamp = datetime(2026, 1, 30, 17, 0)
        assert executor.is_market_open(timestamp) is False

    def test_market_hours_extended(self):
        """Test extended hours trading."""
        executor = OrderExecutor(allow_after_hours=True)

        # 8 AM (pre-market, extended hours allowed)
        timestamp = datetime(2026, 1, 30, 8, 0)
        assert executor.is_market_open(timestamp) is True

        # 5 PM (after-hours, extended hours allowed)
        timestamp = datetime(2026, 1, 30, 17, 0)
        assert executor.is_market_open(timestamp) is True

    def test_shortable_check(self):
        """Test checking if stock is shortable."""
        executor = OrderExecutor()

        is_shortable, borrow_rate = executor.check_shortable('AAPL')

        # Most stocks should be shortable
        assert is_shortable is True
        assert borrow_rate > 0

    def test_borrow_cost_calculation(self):
        """Test short borrow cost calculation."""
        executor = OrderExecutor()

        # 100 shares at $150, 5% borrow rate, 30 days
        cost = executor.calculate_borrow_cost(
            quantity=100,
            price=150.0,
            borrow_rate=0.05,  # 5% annual
            holding_days=30
        )

        # Position value: 100 * 150 = $15,000
        # Daily rate: 5% / 365 = 0.0137%
        # 30 days: 15000 * 0.000137 * 30 = $61.64
        expected_cost = 15000 * (0.05 / 365) * 30
        assert cost == pytest.approx(expected_cost, rel=0.01)

    def test_margin_interest_calculation(self):
        """Test margin interest calculation."""
        executor = OrderExecutor()

        # Borrowed $10,000, 8% rate, 30 days
        interest = executor.calculate_margin_interest(
            borrowed_amount=10000,
            holding_days=30
        )

        # Daily rate: 8% / 365
        # Interest: 10000 * (0.08/365) * 30 = $65.75
        expected_interest = 10000 * (0.08 / 365) * 30
        assert interest == pytest.approx(expected_interest, rel=0.01)

    def test_partial_fill_simulation(self):
        """Test partial fill simulation."""
        executor = OrderExecutor()

        # Set random seed for reproducibility
        np.random.seed(42)

        # High liquidity (should mostly fill completely)
        filled, unfilled = executor.simulate_partial_fill(
            requested_quantity=1000,
            liquidity_score=0.99
        )

        assert filled + unfilled == 1000
        assert filled > 0

    def test_order_rejection_insufficient_funds(self):
        """Test order rejection due to insufficient funds."""
        executor = OrderExecutor()

        is_rejected, reason = executor.simulate_rejection(
            quantity=1000,
            price=150.0,  # Order cost: $150,000
            available_capital=100000,  # Only have $100,000
            rejection_probability=0.0  # Disable random rejection
        )

        assert is_rejected is True
        assert "Insufficient funds" in reason

    def test_execute_order_success(self):
        """Test successful order execution."""
        executor = OrderExecutor()

        result = executor.execute_order(
            symbol='AAPL',
            quantity=100,
            side=OrderSide.BUY,
            price=150.0,
            available_capital=100000,
            timestamp=datetime(2026, 1, 30, 10, 0)  # During market hours
        )

        # Should be filled or partially filled
        assert result['status'] in [OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED]
        assert result['filled_quantity'] > 0

    def test_execute_order_market_closed(self):
        """Test order rejection when market is closed."""
        executor = OrderExecutor(allow_after_hours=False)

        result = executor.execute_order(
            symbol='AAPL',
            quantity=100,
            side=OrderSide.BUY,
            price=150.0,
            available_capital=100000,
            timestamp=datetime(2026, 1, 30, 20, 0)  # 8 PM (closed)
        )

        assert result['status'] == OrderStatus.REJECTED
        assert result['reason'] == 'Market closed'
        assert result['filled_quantity'] == 0

    def test_execute_order_not_shortable(self):
        """Test short order rejection for non-shortable stock."""
        # Create executor with custom borrow costs
        borrow_costs = BorrowCosts()
        executor = OrderExecutor(borrow_costs=borrow_costs)

        # Mock check_shortable to return False
        def mock_check(*args):
            return False, 0.0

        executor.check_shortable = mock_check

        result = executor.execute_order(
            symbol='XYZ',
            quantity=100,
            side=OrderSide.SHORT,
            price=50.0,
            available_capital=100000,
            timestamp=datetime(2026, 1, 30, 10, 0)
        )

        assert result['status'] == OrderStatus.REJECTED
        assert 'not shortable' in result['reason']

    def test_execute_order_market_impact(self):
        """Test that market impact affects execution price."""
        executor = OrderExecutor()

        result = executor.execute_order(
            symbol='AAPL',
            quantity=10000,  # Large order
            side=OrderSide.BUY,
            price=150.0,
            available_capital=2000000,
            average_daily_volume=1000000,
            spread=0.02,
            timestamp=datetime(2026, 1, 30, 10, 0)
        )

        # Execution price should be higher than quoted (buy order)
        if result['status'] != OrderStatus.REJECTED:
            assert result['execution_price'] >= result['quoted_price']
            assert result['market_impact'] >= 0
