"""
Unit Tests for Risk Management Module
Tests risk limits, metrics calculations, and pre-trade checks
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from risk_management import (
    RiskLimits, RiskMetrics, RiskManager
)
from exceptions import (
    PositionLimitExceededError, ConcentrationLimitExceededError,
    DrawdownExceededError, RiskLimitExceededError
)


class TestRiskMetrics:
    """Test risk metrics calculations."""

    def test_sharpe_ratio_calculation(self):
        """Test Sharpe ratio calculation."""
        # Create return series with known properties
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.02, 252))  # 252 trading days

        sharpe = RiskMetrics.calculate_sharpe_ratio(returns, risk_free_rate=0.02)

        # Sharpe should be reasonable (typically between -3 and 3)
        assert -3 < sharpe < 3
        assert isinstance(sharpe, float)

    def test_sortino_ratio_calculation(self):
        """Test Sortino ratio calculation."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.02, 252))

        sortino = RiskMetrics.calculate_sortino_ratio(returns, risk_free_rate=0.02)

        # Sortino should be >= Sharpe (uses only downside deviation)
        sharpe = RiskMetrics.calculate_sharpe_ratio(returns, risk_free_rate=0.02)
        assert sortino >= sharpe

    def test_max_drawdown_calculation(self):
        """Test maximum drawdown calculation."""
        # Create price series with known drawdown
        prices = pd.Series([100, 110, 105, 95, 90, 100, 110])

        max_dd, peak_date, trough_date = RiskMetrics.calculate_max_drawdown(prices)

        # Max drawdown should be from 110 to 90 = -18.18%
        assert max_dd == pytest.approx(-0.1818, abs=0.001)
        assert peak_date == 1  # Index of peak (110)
        assert trough_date == 4  # Index of trough (90)

    def test_var_calculation(self):
        """Test Value at Risk calculation."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0, 0.02, 1000))

        var_95 = RiskMetrics.calculate_var(returns, confidence=0.95)

        # VaR should be negative (represents potential loss)
        assert var_95 < 0

        # For 95% confidence, ~5% of returns should be worse than VaR
        worse_than_var = (returns <= var_95).sum()
        assert 40 < worse_than_var < 60  # Approximately 50 out of 1000

    def test_cvar_calculation(self):
        """Test Conditional VaR (Expected Shortfall)."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0, 0.02, 1000))

        var_95 = RiskMetrics.calculate_var(returns, confidence=0.95)
        cvar_95 = RiskMetrics.calculate_cvar(returns, confidence=0.95)

        # CVaR should be worse (more negative) than VaR
        assert cvar_95 < var_95

    def test_correlation_matrix(self):
        """Test correlation matrix calculation."""
        np.random.seed(42)

        # Create correlated returns
        base_returns = pd.Series(np.random.normal(0, 0.02, 100))
        returns_dict = {
            'AAPL': base_returns + np.random.normal(0, 0.01, 100),
            'MSFT': base_returns + np.random.normal(0, 0.01, 100),
            'TSLA': pd.Series(np.random.normal(0, 0.03, 100))  # Uncorrelated
        }

        corr_matrix = RiskMetrics.calculate_correlation_matrix(returns_dict)

        # AAPL and MSFT should be highly correlated
        assert corr_matrix.loc['AAPL', 'MSFT'] > 0.5

        # Diagonal should be 1.0
        assert corr_matrix.loc['AAPL', 'AAPL'] == pytest.approx(1.0)


class TestRiskManager:
    """Test RiskManager class."""

    def test_initialization(self):
        """Test risk manager initialization."""
        rm = RiskManager(initial_capital=100000)

        assert rm.initial_capital == 100000
        assert rm.current_capital == 100000
        assert len(rm.positions) == 0

    def test_add_position(self):
        """Test adding a position."""
        rm = RiskManager(initial_capital=100000)

        rm.add_position('AAPL', 100, 150.0, sector='Technology')

        assert 'AAPL' in rm.positions
        assert rm.positions['AAPL']['quantity'] == 100
        assert rm.positions['AAPL']['avg_price'] == 150.0
        assert rm.positions['AAPL']['sector'] == 'Technology'

    def test_update_position(self):
        """Test updating an existing position."""
        rm = RiskManager(initial_capital=100000)

        # Add initial position
        rm.add_position('AAPL', 100, 150.0)

        # Add more shares at different price
        rm.add_position('AAPL', 50, 160.0)

        # Average price should be (100*150 + 50*160) / 150 = 153.33
        assert rm.positions['AAPL']['quantity'] == 150
        assert rm.positions['AAPL']['avg_price'] == pytest.approx(153.33, abs=0.01)

    def test_remove_position(self):
        """Test removing a position."""
        rm = RiskManager(initial_capital=100000)

        rm.add_position('AAPL', 100, 150.0)
        assert 'AAPL' in rm.positions

        rm.remove_position('AAPL')
        assert 'AAPL' not in rm.positions

    def test_close_position_on_opposite_trade(self):
        """Test that position is removed when fully closed."""
        rm = RiskManager(initial_capital=100000)

        rm.add_position('AAPL', 100, 150.0)
        rm.add_position('AAPL', -100, 160.0)  # Close position

        assert 'AAPL' not in rm.positions

    def test_get_position_value(self):
        """Test calculating position value."""
        rm = RiskManager(initial_capital=100000)

        rm.add_position('AAPL', 100, 150.0)

        value = rm.get_position_value('AAPL', 160.0)
        assert value == 16000.0  # 100 shares * $160

    def test_get_total_exposure(self):
        """Test calculating total portfolio exposure."""
        rm = RiskManager(initial_capital=100000)

        rm.add_position('AAPL', 100, 150.0)
        rm.add_position('MSFT', 50, 300.0)

        prices = {'AAPL': 160.0, 'MSFT': 310.0}
        exposure = rm.get_total_exposure(prices)

        # 100*160 + 50*310 = 16000 + 15500 = 31500
        assert exposure == 31500.0

    def test_get_sector_exposure(self):
        """Test calculating sector exposure."""
        rm = RiskManager(initial_capital=100000)

        rm.add_position('AAPL', 100, 150.0, sector='Technology')
        rm.add_position('MSFT', 50, 300.0, sector='Technology')
        rm.add_position('JPM', 50, 150.0, sector='Finance')

        prices = {'AAPL': 160.0, 'MSFT': 310.0, 'JPM': 155.0}
        tech_exposure = rm.get_sector_exposure('Technology', prices)

        # Tech: 16000 + 15500 = 31500
        # Total: 31500 + 7750 = 39250
        # Tech %: 31500 / 39250 = 0.8025
        assert tech_exposure == pytest.approx(0.8025, abs=0.001)

    def test_position_size_limit_exceeded(self):
        """Test that position size limit is enforced."""
        limits = RiskLimits(max_position_size=500)
        rm = RiskManager(limits=limits, initial_capital=100000)

        # Try to add 600 shares (exceeds limit of 500)
        with pytest.raises(PositionLimitExceededError):
            rm.check_position_limit('AAPL', 600, 150.0)

    def test_position_value_limit_exceeded(self):
        """Test that position value limit is enforced."""
        limits = RiskLimits(max_position_value=30000)
        rm = RiskManager(limits=limits, initial_capital=100000)

        # 250 shares * $150 = $37,500 (exceeds $30,000 limit)
        with pytest.raises(PositionLimitExceededError):
            rm.check_position_limit('AAPL', 250, 150.0)

    def test_concentration_limit_exceeded(self):
        """Test that concentration limit is enforced."""
        limits = RiskLimits(max_single_position_pct=0.20)  # Max 20%
        rm = RiskManager(limits=limits, initial_capital=100000)

        # Add initial position
        rm.add_position('MSFT', 100, 300.0)  # $30,000

        prices = {'AAPL': 150.0, 'MSFT': 300.0}

        # Try to add AAPL position that would be 30% (300*150=45000 out of 150000)
        with pytest.raises(ConcentrationLimitExceededError):
            rm.check_concentration_limit('AAPL', 300, 150.0, prices)

    def test_drawdown_check_triggers(self):
        """Test that drawdown check triggers at limits."""
        limits = RiskLimits(critical_drawdown_pct=0.10)
        rm = RiskManager(limits=limits, initial_capital=100000)

        # Simulate equity curve with 12% drawdown
        rm.update_capital(100000)
        rm.update_capital(105000)  # Peak
        rm.update_capital(92400)  # 12% below peak

        # Should raise exception for critical drawdown
        with pytest.raises(DrawdownExceededError):
            rm.check_drawdown()

    def test_drawdown_check_with_liquidation(self):
        """Test that liquidation callback is called on critical drawdown."""
        liquidated = [False]

        def liquidate_all():
            liquidated[0] = True

        limits = RiskLimits(critical_drawdown_pct=0.10)
        rm = RiskManager(limits=limits, initial_capital=100000)

        # Simulate critical drawdown
        rm.update_capital(100000)
        rm.update_capital(105000)
        rm.update_capital(92400)  # 12% drawdown

        with pytest.raises(DrawdownExceededError):
            rm.check_drawdown(liquidate_callback=liquidate_all)

        assert liquidated[0] is True

    def test_total_exposure_limit(self):
        """Test total exposure limit."""
        limits = RiskLimits(
            max_total_exposure=50000,
            max_single_position_pct=0.50  # Increase to avoid concentration limit
        )
        rm = RiskManager(limits=limits, initial_capital=100000)

        rm.add_position('AAPL', 100, 150.0)
        rm.add_position('MSFT', 100, 300.0)

        prices = {'AAPL': 150.0, 'MSFT': 300.0, 'TSLA': 200.0}

        # Current exposure: 15000 + 30000 = 45000
        # Adding 100 TSLA at $200 = 20000
        # Total would be 65000 > 50000 limit
        with pytest.raises(RiskLimitExceededError):
            rm.pre_trade_check('TSLA', 100, 200.0, prices=prices)

    def test_pre_trade_check_passes(self):
        """Test that valid trade passes pre-trade check."""
        limits = RiskLimits(max_single_position_pct=1.0)  # Allow 100% for this test
        rm = RiskManager(limits=limits, initial_capital=100000)

        prices = {'AAPL': 150.0}

        # Should not raise any exception
        rm.pre_trade_check('AAPL', 100, 150.0, sector='Technology', prices=prices)

    def test_calculate_current_metrics(self):
        """Test calculating current risk metrics."""
        rm = RiskManager(initial_capital=100000)

        # Add some returns
        rm.update_capital(102000)  # +2%
        rm.update_capital(101000)  # -0.98%
        rm.update_capital(103000)  # +1.98%

        metrics = rm.calculate_current_metrics()

        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics
        assert 'current_capital' in metrics
        assert 'total_return' in metrics

        assert metrics['current_capital'] == 103000
        assert metrics['total_return'] == pytest.approx(0.03)  # 3% return

    def test_short_position_handling(self):
        """Test that short positions are handled correctly."""
        rm = RiskManager(initial_capital=100000)

        # Add short position (negative quantity)
        rm.add_position('AAPL', -100, 150.0)

        assert rm.positions['AAPL']['quantity'] == -100

        # Position value should be negative
        value = rm.get_position_value('AAPL', 140.0)
        assert value == -14000  # -100 * 140

        # But exposure should be absolute
        prices = {'AAPL': 140.0}
        exposure = rm.get_total_exposure(prices)
        assert exposure == 14000  # Absolute value
