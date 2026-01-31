"""
Risk Management Module
Comprehensive risk controls including position limits, VaR, drawdown management
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from exceptions import (
    RiskLimitExceededError, PositionLimitExceededError,
    DrawdownExceededError, ConcentrationLimitExceededError,
    CorrelationLimitExceededError
)

logger = logging.getLogger(__name__)


@dataclass
class RiskLimits:
    """Risk limit configuration."""

    # Position limits
    max_position_size: float = 1000  # Max shares per symbol
    max_position_value: float = 50000  # Max dollar value per position
    max_total_exposure: float = 200000  # Max total portfolio value

    # Concentration limits
    max_single_position_pct: float = 0.15  # Max 15% in one position
    max_sector_exposure_pct: float = 0.40  # Max 40% in one sector
    max_correlated_exposure_pct: float = 0.50  # Max 50% in correlated assets

    # Drawdown limits
    max_drawdown_pct: float = 0.05  # Max 5% drawdown before action
    critical_drawdown_pct: float = 0.10  # Liquidate all at 10% drawdown

    # Risk metrics
    min_sharpe_ratio: float = 0.5  # Minimum acceptable Sharpe ratio
    max_var_95: float = 0.02  # Max 2% daily VaR at 95% confidence
    max_var_99: float = 0.04  # Max 4% daily VaR at 99% confidence

    # Correlation limits
    correlation_threshold: float = 0.7  # Correlation above this is "highly correlated"


class RiskMetrics:
    """Calculate and track risk metrics."""

    @staticmethod
    def calculate_returns(prices: pd.Series) -> pd.Series:
        """
        Calculate returns from price series.

        Args:
            prices: Price series

        Returns:
            Returns series
        """
        return prices.pct_change().dropna()

    @staticmethod
    def calculate_sharpe_ratio(
        returns: pd.Series,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate Sharpe ratio.

        Args:
            returns: Return series
            risk_free_rate: Annual risk-free rate
            periods_per_year: Trading periods per year

        Returns:
            Sharpe ratio
        """
        if len(returns) == 0:
            return 0.0

        excess_returns = returns - (risk_free_rate / periods_per_year)
        if excess_returns.std() == 0:
            return 0.0

        sharpe = np.sqrt(periods_per_year) * excess_returns.mean() / excess_returns.std()
        return float(sharpe)

    @staticmethod
    def calculate_sortino_ratio(
        returns: pd.Series,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate Sortino ratio (uses downside deviation).

        Args:
            returns: Return series
            risk_free_rate: Annual risk-free rate
            periods_per_year: Trading periods per year

        Returns:
            Sortino ratio
        """
        if len(returns) == 0:
            return 0.0

        excess_returns = returns - (risk_free_rate / periods_per_year)
        downside_returns = excess_returns[excess_returns < 0]

        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0

        sortino = np.sqrt(periods_per_year) * excess_returns.mean() / downside_returns.std()
        return float(sortino)

    @staticmethod
    def calculate_max_drawdown(prices: pd.Series) -> Tuple[float, datetime, datetime]:
        """
        Calculate maximum drawdown.

        Args:
            prices: Price series

        Returns:
            Tuple of (max_drawdown, peak_date, trough_date)
        """
        if len(prices) == 0:
            return 0.0, None, None

        cumulative = (1 + prices.pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max

        max_dd = drawdown.min()
        trough_date = drawdown.idxmin()

        # Find peak before trough
        peak_date = cumulative[:trough_date].idxmax()

        return float(max_dd), peak_date, trough_date

    @staticmethod
    def calculate_var(
        returns: pd.Series,
        confidence: float = 0.95
    ) -> float:
        """
        Calculate Value at Risk (historical method).

        Args:
            returns: Return series
            confidence: Confidence level (e.g., 0.95 for 95%)

        Returns:
            VaR value (negative number representing potential loss)
        """
        if len(returns) == 0:
            return 0.0

        var = np.percentile(returns, (1 - confidence) * 100)
        return float(var)

    @staticmethod
    def calculate_cvar(
        returns: pd.Series,
        confidence: float = 0.95
    ) -> float:
        """
        Calculate Conditional Value at Risk (Expected Shortfall).

        Args:
            returns: Return series
            confidence: Confidence level

        Returns:
            CVaR value (expected loss beyond VaR)
        """
        if len(returns) == 0:
            return 0.0

        var = RiskMetrics.calculate_var(returns, confidence)
        cvar = returns[returns <= var].mean()
        return float(cvar)

    @staticmethod
    def calculate_correlation_matrix(
        returns_dict: Dict[str, pd.Series]
    ) -> pd.DataFrame:
        """
        Calculate correlation matrix for multiple assets.

        Args:
            returns_dict: Dict of {symbol: returns_series}

        Returns:
            Correlation matrix DataFrame
        """
        df = pd.DataFrame(returns_dict)
        return df.corr()


class RiskManager:
    """
    Comprehensive risk management system.

    Enforces position limits, tracks risk metrics, manages drawdowns.
    """

    def __init__(
        self,
        limits: Optional[RiskLimits] = None,
        initial_capital: float = 100000.0
    ):
        """
        Initialize risk manager.

        Args:
            limits: Risk limits configuration
            initial_capital: Starting capital
        """
        self.limits = limits or RiskLimits()
        self.initial_capital = initial_capital
        self.current_capital = initial_capital

        # Position tracking
        self.positions: Dict[str, Dict] = {}  # {symbol: {quantity, avg_price, sector}}
        self.position_history: List[Dict] = []

        # Returns tracking
        self.daily_returns: pd.Series = pd.Series(dtype=float)
        self.equity_curve: pd.Series = pd.Series(dtype=float)

        # Risk metrics cache
        self._metrics_cache = {}
        self._metrics_cache_time = None

        logger.info(f"Risk manager initialized with capital: ${initial_capital:,.2f}")

    def update_capital(self, capital: float):
        """
        Update current capital and record return.

        Args:
            capital: New capital value
        """
        if self.current_capital > 0:
            daily_return = (capital - self.current_capital) / self.current_capital
            self.daily_returns[datetime.now()] = daily_return

        self.current_capital = capital
        self.equity_curve[datetime.now()] = capital

        # Clear metrics cache
        self._metrics_cache = {}

    def add_position(
        self,
        symbol: str,
        quantity: int,
        price: float,
        sector: Optional[str] = None
    ):
        """
        Add or update position.

        Args:
            symbol: Stock symbol
            quantity: Number of shares (negative for short)
            price: Entry price
            sector: Sector classification
        """
        if symbol in self.positions:
            # Update existing position
            pos = self.positions[symbol]
            total_quantity = pos['quantity'] + quantity
            if total_quantity != 0:
                # Calculate new average price
                total_cost = (pos['quantity'] * pos['avg_price']) + (quantity * price)
                pos['avg_price'] = total_cost / total_quantity
                pos['quantity'] = total_quantity
            else:
                # Position closed
                del self.positions[symbol]
        else:
            # New position
            self.positions[symbol] = {
                'quantity': quantity,
                'avg_price': price,
                'sector': sector,
                'entry_date': datetime.now()
            }

        logger.info(
            f"Position updated: {symbol} = {quantity} shares @ ${price:.2f}"
        )

    def remove_position(self, symbol: str):
        """Remove position."""
        if symbol in self.positions:
            del self.positions[symbol]
            logger.info(f"Position removed: {symbol}")

    def get_position_value(self, symbol: str, current_price: float) -> float:
        """
        Get current value of a position.

        Args:
            symbol: Stock symbol
            current_price: Current market price

        Returns:
            Position value (positive for long, negative for short)
        """
        if symbol not in self.positions:
            return 0.0

        pos = self.positions[symbol]
        return pos['quantity'] * current_price

    def get_total_exposure(self, prices: Dict[str, float]) -> float:
        """
        Get total portfolio exposure.

        Args:
            prices: Dict of {symbol: current_price}

        Returns:
            Total absolute exposure
        """
        total = 0.0
        for symbol, pos in self.positions.items():
            if symbol in prices:
                total += abs(self.get_position_value(symbol, prices[symbol]))
        return total

    def get_sector_exposure(
        self,
        sector: str,
        prices: Dict[str, float]
    ) -> float:
        """
        Get exposure to a specific sector.

        Args:
            sector: Sector name
            prices: Dict of {symbol: current_price}

        Returns:
            Sector exposure as fraction of portfolio
        """
        sector_value = 0.0
        for symbol, pos in self.positions.items():
            if pos.get('sector') == sector and symbol in prices:
                sector_value += abs(self.get_position_value(symbol, prices[symbol]))

        total_exposure = self.get_total_exposure(prices)
        if total_exposure == 0:
            return 0.0

        return sector_value / total_exposure

    def check_position_limit(
        self,
        symbol: str,
        quantity: int,
        price: float
    ):
        """
        Check if adding position would exceed limits.

        Args:
            symbol: Stock symbol
            quantity: Shares to add
            price: Price per share

        Raises:
            PositionLimitExceededError: If limit would be exceeded
        """
        # Get current position
        current_qty = 0
        if symbol in self.positions:
            current_qty = self.positions[symbol]['quantity']

        new_qty = current_qty + quantity
        new_value = abs(new_qty * price)

        # Check quantity limit
        if abs(new_qty) > self.limits.max_position_size:
            raise PositionLimitExceededError(
                f"Position size limit exceeded for {symbol}: "
                f"{abs(new_qty)} > {self.limits.max_position_size} shares"
            )

        # Check value limit
        if new_value > self.limits.max_position_value:
            raise PositionLimitExceededError(
                f"Position value limit exceeded for {symbol}: "
                f"${new_value:,.2f} > ${self.limits.max_position_value:,.2f}"
            )

    def check_concentration_limit(
        self,
        symbol: str,
        quantity: int,
        price: float,
        prices: Dict[str, float]
    ):
        """
        Check if position would exceed concentration limits.

        Args:
            symbol: Stock symbol
            quantity: Shares to add
            price: Price per share
            prices: Dict of all current prices

        Raises:
            ConcentrationLimitExceededError: If limit would be exceeded
        """
        # Calculate new position value
        current_qty = self.positions.get(symbol, {}).get('quantity', 0)
        new_qty = current_qty + quantity
        new_position_value = abs(new_qty * price)

        # Calculate total portfolio value
        total_value = self.get_total_exposure(prices) - abs(current_qty * price) + new_position_value

        if total_value == 0:
            return

        # Check single position concentration
        concentration = new_position_value / total_value
        if concentration > self.limits.max_single_position_pct:
            raise ConcentrationLimitExceededError(
                f"Position concentration limit exceeded for {symbol}: "
                f"{concentration:.1%} > {self.limits.max_single_position_pct:.1%}"
            )

    def check_drawdown(self, liquidate_callback: Optional[callable] = None):
        """
        Check if drawdown limits are exceeded.

        Args:
            liquidate_callback: Optional function to call for liquidation

        Raises:
            DrawdownExceededError: If critical drawdown exceeded
        """
        if len(self.equity_curve) < 2:
            return

        max_dd, peak_date, trough_date = RiskMetrics.calculate_max_drawdown(
            self.equity_curve
        )

        logger.info(f"Current drawdown: {max_dd:.2%}")

        # Check critical drawdown (liquidate everything)
        if abs(max_dd) > self.limits.critical_drawdown_pct:
            logger.critical(
                f"CRITICAL DRAWDOWN EXCEEDED: {abs(max_dd):.2%} > "
                f"{self.limits.critical_drawdown_pct:.2%}"
            )

            if liquidate_callback:
                logger.critical("Liquidating all positions...")
                liquidate_callback()

            raise DrawdownExceededError(
                f"Critical drawdown exceeded: {abs(max_dd):.2%}"
            )

        # Check max drawdown (warning, reduce exposure)
        if abs(max_dd) > self.limits.max_drawdown_pct:
            logger.warning(
                f"Max drawdown exceeded: {abs(max_dd):.2%} > "
                f"{self.limits.max_drawdown_pct:.2%}"
            )
            # Don't raise exception, but log warning
            # Trading system should reduce risk

    def calculate_current_metrics(self) -> Dict:
        """
        Calculate current risk metrics.

        Returns:
            Dict of risk metrics
        """
        metrics = {}

        if len(self.daily_returns) > 0:
            metrics['sharpe_ratio'] = RiskMetrics.calculate_sharpe_ratio(self.daily_returns)
            metrics['sortino_ratio'] = RiskMetrics.calculate_sortino_ratio(self.daily_returns)
            metrics['var_95'] = RiskMetrics.calculate_var(self.daily_returns, 0.95)
            metrics['var_99'] = RiskMetrics.calculate_var(self.daily_returns, 0.99)
            metrics['cvar_95'] = RiskMetrics.calculate_cvar(self.daily_returns, 0.95)
            metrics['cvar_99'] = RiskMetrics.calculate_cvar(self.daily_returns, 0.99)

        if len(self.equity_curve) > 0:
            max_dd, peak_date, trough_date = RiskMetrics.calculate_max_drawdown(self.equity_curve)
            metrics['max_drawdown'] = max_dd
            metrics['max_drawdown_peak'] = peak_date
            metrics['max_drawdown_trough'] = trough_date

        metrics['current_capital'] = self.current_capital
        metrics['total_return'] = (self.current_capital - self.initial_capital) / self.initial_capital
        metrics['num_positions'] = len(self.positions)

        return metrics

    def pre_trade_check(
        self,
        symbol: str,
        quantity: int,
        price: float,
        sector: Optional[str] = None,
        prices: Optional[Dict[str, float]] = None
    ):
        """
        Comprehensive pre-trade risk check.

        Args:
            symbol: Stock symbol
            quantity: Shares to trade
            price: Price per share
            sector: Sector classification
            prices: Dict of current prices for all positions

        Raises:
            Various risk exceptions if limits exceeded
        """
        logger.info(f"Pre-trade risk check: {symbol} {quantity} @ ${price:.2f}")

        # Check position limits
        self.check_position_limit(symbol, quantity, price)

        # Check concentration limits
        if prices:
            self.check_concentration_limit(symbol, quantity, price, prices)

        # Check total exposure
        if prices:
            current_exposure = self.get_total_exposure(prices)
            position_value = abs(quantity * price)
            new_exposure = current_exposure + position_value

            if new_exposure > self.limits.max_total_exposure:
                raise RiskLimitExceededError(
                    f"Total exposure limit exceeded: "
                    f"${new_exposure:,.2f} > ${self.limits.max_total_exposure:,.2f}"
                )

        # Check drawdown
        self.check_drawdown()

        logger.info("Pre-trade risk check passed")
