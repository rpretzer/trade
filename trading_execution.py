"""
Trading Execution Module
Realistic order execution with slippage, partial fills, market impact, and costs
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
from datetime import datetime, time, timedelta
from dataclasses import dataclass
from enum import Enum
from exceptions import (
    OrderRejectedError, InvalidOrderError, InsufficientFundsError,
    ShortNotAvailableError
)

logger = logging.getLogger(__name__)


class OrderSide(Enum):
    """Order side enumeration."""
    BUY = "BUY"
    SELL = "SELL"
    SHORT = "SHORT"
    COVER = "COVER"


class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = "PENDING"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"


@dataclass
class MarketHours:
    """Market hours configuration."""
    market_open: time = time(9, 30)  # 9:30 AM
    market_close: time = time(16, 0)  # 4:00 PM
    pre_market_open: time = time(4, 0)  # 4:00 AM
    after_hours_close: time = time(20, 0)  # 8:00 PM


@dataclass
class BorrowCosts:
    """Short selling borrow cost configuration."""
    # Annual borrow rates by difficulty
    easy_to_borrow_rate: float = 0.01  # 1% per year
    moderate_borrow_rate: float = 0.05  # 5% per year
    hard_to_borrow_rate: float = 0.15  # 15% per year

    # Thresholds for categorization (can be customized per symbol)
    hard_to_borrow_symbols: List[str] = None

    def __post_init__(self):
        if self.hard_to_borrow_symbols is None:
            self.hard_to_borrow_symbols = []

    def get_borrow_rate(self, symbol: str) -> float:
        """Get annual borrow rate for a symbol."""
        if symbol in self.hard_to_borrow_symbols:
            return self.hard_to_borrow_rate
        # In practice, would query broker API for actual rate
        return self.easy_to_borrow_rate


@dataclass
class MarginConfig:
    """Margin trading configuration."""
    margin_interest_rate: float = 0.08  # 8% annual
    maintenance_margin: float = 0.25  # 25% maintenance margin
    initial_margin: float = 0.50  # 50% initial margin


class KellyCriterion:
    """
    Kelly Criterion position sizing calculator.

    Calculates optimal position size based on win rate and win/loss ratios.
    """

    @staticmethod
    def calculate_full_kelly(
        win_rate: float,
        avg_win: float,
        avg_loss: float
    ) -> float:
        """
        Calculate full Kelly fraction.

        Args:
            win_rate: Probability of winning (0-1)
            avg_win: Average win size (as fraction, e.g., 0.02 = 2%)
            avg_loss: Average loss size (as fraction, positive number)

        Returns:
            Kelly fraction (0-1)
        """
        if avg_win <= 0 or avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
            return 0.0

        # Kelly formula: f = (p*W - q*L) / W
        # where p = win rate, q = loss rate, W = avg win, L = avg loss
        loss_rate = 1 - win_rate
        kelly = (win_rate * avg_win - loss_rate * avg_loss) / avg_win

        return max(0.0, kelly)

    @staticmethod
    def calculate_fractional_kelly(
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        kelly_fraction: float = 0.25
    ) -> float:
        """
        Calculate fractional Kelly (safer than full Kelly).

        Args:
            win_rate: Probability of winning
            avg_win: Average win size
            avg_loss: Average loss size
            kelly_fraction: Fraction of full Kelly to use (default 0.25 = quarter Kelly)

        Returns:
            Fractional Kelly position size
        """
        full_kelly = KellyCriterion.calculate_full_kelly(win_rate, avg_win, avg_loss)
        return full_kelly * kelly_fraction

    @staticmethod
    def calculate_from_sharpe(
        sharpe_ratio: float,
        kelly_fraction: float = 0.25
    ) -> float:
        """
        Approximate Kelly from Sharpe ratio.

        Kelly ≈ Sharpe² (for normal returns)

        Args:
            sharpe_ratio: Strategy Sharpe ratio
            kelly_fraction: Fraction of full Kelly

        Returns:
            Position size
        """
        if sharpe_ratio <= 0:
            return 0.0

        full_kelly = sharpe_ratio ** 2
        return min(full_kelly * kelly_fraction, 1.0)


class PositionSizer:
    """
    Advanced position sizing with multiple methods.
    """

    @staticmethod
    def kelly_sizing(
        capital: float,
        price: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        kelly_fraction: float = 0.25,
        max_position_pct: float = 0.15
    ) -> int:
        """
        Calculate position size using Kelly Criterion.

        Args:
            capital: Available capital
            price: Stock price
            win_rate: Historical win rate
            avg_win: Average winning trade %
            avg_loss: Average losing trade %
            kelly_fraction: Fraction of Kelly to use (0.25 = quarter Kelly)
            max_position_pct: Maximum position as % of capital

        Returns:
            Number of shares to trade
        """
        kelly_pct = KellyCriterion.calculate_fractional_kelly(
            win_rate, avg_win, avg_loss, kelly_fraction
        )

        # Cap at maximum position percentage
        position_pct = min(kelly_pct, max_position_pct)

        # Calculate shares
        position_value = capital * position_pct
        shares = int(position_value / price)

        logger.info(
            f"Kelly sizing: win_rate={win_rate:.2%}, kelly={kelly_pct:.2%}, "
            f"capped={position_pct:.2%}, shares={shares}"
        )

        return shares

    @staticmethod
    def volatility_adjusted_sizing(
        capital: float,
        price: float,
        current_volatility: float,
        target_volatility: float = 0.02,
        base_position_pct: float = 0.10
    ) -> int:
        """
        Adjust position size based on volatility.

        Lower volatility → larger position
        Higher volatility → smaller position

        Args:
            capital: Available capital
            price: Stock price
            current_volatility: Current daily volatility (std dev of returns)
            target_volatility: Target portfolio volatility
            base_position_pct: Base position size

        Returns:
            Number of shares
        """
        if current_volatility <= 0:
            return 0

        # Adjust position size inversely with volatility
        volatility_multiplier = target_volatility / current_volatility
        adjusted_pct = base_position_pct * volatility_multiplier

        # Cap at reasonable limits
        adjusted_pct = min(adjusted_pct, 0.25)  # Max 25%
        adjusted_pct = max(adjusted_pct, 0.01)  # Min 1%

        position_value = capital * adjusted_pct
        shares = int(position_value / price)

        logger.info(
            f"Volatility-adjusted sizing: vol={current_volatility:.2%}, "
            f"multiplier={volatility_multiplier:.2f}, shares={shares}"
        )

        return shares


class MarketImpact:
    """
    Model market impact of orders.

    Large orders move the market against you.
    """

    @staticmethod
    def calculate_impact(
        quantity: int,
        average_daily_volume: int,
        spread: float,
        price: float
    ) -> float:
        """
        Calculate market impact cost.

        Args:
            quantity: Order size in shares
            average_daily_volume: Average daily volume
            spread: Bid-ask spread
            price: Current price

        Returns:
            Impact cost per share
        """
        if average_daily_volume <= 0:
            return 0.0

        # Order size as fraction of daily volume
        volume_fraction = abs(quantity) / average_daily_volume

        # Base impact: proportional to sqrt(volume fraction)
        # This is a simplified model; real models are more complex
        base_impact = 0.1 * np.sqrt(volume_fraction)

        # Impact scales with spread
        impact = base_impact * spread

        logger.debug(
            f"Market impact: {quantity} shares, {volume_fraction:.2%} of volume, "
            f"impact={impact:.4f}"
        )

        return impact

    @staticmethod
    def apply_impact(
        price: float,
        quantity: int,
        side: OrderSide,
        average_daily_volume: int,
        spread: float
    ) -> float:
        """
        Apply market impact to execution price.

        Args:
            price: Quoted price
            quantity: Order size
            side: Buy or sell
            average_daily_volume: Daily volume
            spread: Bid-ask spread

        Returns:
            Execution price after impact
        """
        impact = MarketImpact.calculate_impact(
            quantity, average_daily_volume, spread, price
        )

        # Buy orders push price up, sell orders push down
        if side in [OrderSide.BUY, OrderSide.COVER]:
            execution_price = price * (1 + impact)
        else:
            execution_price = price * (1 - impact)

        return execution_price


class OrderExecutor:
    """
    Realistic order execution with partial fills, rejections, and costs.
    """

    def __init__(
        self,
        market_hours: Optional[MarketHours] = None,
        borrow_costs: Optional[BorrowCosts] = None,
        margin_config: Optional[MarginConfig] = None,
        allow_after_hours: bool = False
    ):
        """
        Initialize order executor.

        Args:
            market_hours: Market hours configuration
            borrow_costs: Borrow cost configuration
            margin_config: Margin configuration
            allow_after_hours: Allow trading in pre/post market
        """
        self.market_hours = market_hours or MarketHours()
        self.borrow_costs = borrow_costs or BorrowCosts()
        self.margin_config = margin_config or MarginConfig()
        self.allow_after_hours = allow_after_hours

    def is_market_open(self, timestamp: datetime) -> bool:
        """Check if market is open at given time."""
        current_time = timestamp.time()

        # Regular hours
        if self.market_hours.market_open <= current_time <= self.market_hours.market_close:
            return True

        # Extended hours (if allowed)
        if self.allow_after_hours:
            if self.market_hours.pre_market_open <= current_time < self.market_hours.market_open:
                return True
            if self.market_hours.market_close < current_time <= self.market_hours.after_hours_close:
                return True

        return False

    def check_shortable(self, symbol: str) -> Tuple[bool, float]:
        """
        Check if stock can be shorted and get borrow rate.

        Args:
            symbol: Stock symbol

        Returns:
            Tuple of (is_shortable, borrow_rate)
        """
        # In practice, would query broker API
        # For now, assume most stocks are shortable

        # Some stocks are hard to borrow or not shortable
        not_shortable = []  # Could populate with known hard-to-borrow stocks

        if symbol in not_shortable:
            return False, 0.0

        borrow_rate = self.borrow_costs.get_borrow_rate(symbol)
        return True, borrow_rate

    def calculate_borrow_cost(
        self,
        quantity: int,
        price: float,
        borrow_rate: float,
        holding_days: int
    ) -> float:
        """
        Calculate short borrow cost.

        Args:
            quantity: Number of shares (positive)
            price: Share price
            borrow_rate: Annual borrow rate
            holding_days: Days position is held

        Returns:
            Borrow cost in dollars
        """
        position_value = quantity * price
        daily_rate = borrow_rate / 365
        cost = position_value * daily_rate * holding_days

        return cost

    def calculate_margin_interest(
        self,
        borrowed_amount: float,
        holding_days: int
    ) -> float:
        """
        Calculate margin interest cost.

        Args:
            borrowed_amount: Amount borrowed on margin
            holding_days: Days position is held

        Returns:
            Interest cost in dollars
        """
        daily_rate = self.margin_config.margin_interest_rate / 365
        interest = borrowed_amount * daily_rate * holding_days

        return interest

    def simulate_partial_fill(
        self,
        requested_quantity: int,
        liquidity_score: float = 0.95
    ) -> Tuple[int, int]:
        """
        Simulate partial fill based on liquidity.

        Args:
            requested_quantity: Requested number of shares
            liquidity_score: Probability of full fill (0-1)

        Returns:
            Tuple of (filled_quantity, unfilled_quantity)
        """
        # High liquidity stocks usually fill completely
        # Low liquidity may have partial fills

        if np.random.random() < liquidity_score:
            # Full fill
            return requested_quantity, 0

        # Partial fill (75-95% of order)
        fill_rate = np.random.uniform(0.75, 0.95)
        filled = int(requested_quantity * fill_rate)
        unfilled = requested_quantity - filled

        logger.warning(
            f"Partial fill: {filled}/{requested_quantity} shares filled "
            f"({fill_rate:.1%})"
        )

        return filled, unfilled

    def simulate_rejection(
        self,
        quantity: int,
        price: float,
        available_capital: float,
        rejection_probability: float = 0.01
    ) -> Tuple[bool, Optional[str]]:
        """
        Simulate order rejection.

        Args:
            quantity: Order quantity
            price: Order price
            available_capital: Available capital
            rejection_probability: Base rejection probability

        Returns:
            Tuple of (is_rejected, rejection_reason)
        """
        # Check insufficient funds
        order_cost = quantity * price
        if order_cost > available_capital:
            return True, "Insufficient funds"

        # Random rejection (network issues, broker systems, etc.)
        if np.random.random() < rejection_probability:
            reasons = [
                "Broker system error",
                "Network timeout",
                "Position limit exceeded at broker",
                "Stock halted"
            ]
            reason = np.random.choice(reasons)
            return True, reason

        return False, None

    def execute_order(
        self,
        symbol: str,
        quantity: int,
        side: OrderSide,
        price: float,
        available_capital: float,
        average_daily_volume: int = 1000000,
        spread: float = 0.01,
        timestamp: Optional[datetime] = None
    ) -> Dict:
        """
        Execute order with realistic simulation.

        Args:
            symbol: Stock symbol
            quantity: Number of shares
            side: Order side
            price: Current market price
            available_capital: Available capital
            average_daily_volume: Average daily volume
            spread: Bid-ask spread
            timestamp: Order timestamp

        Returns:
            Dict with execution details
        """
        if timestamp is None:
            timestamp = datetime.now()

        logger.info(
            f"Executing order: {side.value} {quantity} {symbol} @ ${price:.2f}"
        )

        # Check market hours
        if not self.is_market_open(timestamp):
            logger.error("Market is closed")
            return {
                'status': OrderStatus.REJECTED,
                'reason': 'Market closed',
                'filled_quantity': 0
            }

        # Check if shorting
        if side == OrderSide.SHORT:
            is_shortable, borrow_rate = self.check_shortable(symbol)
            if not is_shortable:
                logger.error(f"{symbol} is not shortable")
                return {
                    'status': OrderStatus.REJECTED,
                    'reason': 'Stock not shortable',
                    'filled_quantity': 0
                }

        # Check for rejection
        is_rejected, rejection_reason = self.simulate_rejection(
            quantity, price, available_capital
        )
        if is_rejected:
            logger.error(f"Order rejected: {rejection_reason}")
            return {
                'status': OrderStatus.REJECTED,
                'reason': rejection_reason,
                'filled_quantity': 0
            }

        # Apply market impact
        execution_price = MarketImpact.apply_impact(
            price, quantity, side, average_daily_volume, spread
        )

        # Simulate partial fill
        filled_qty, unfilled_qty = self.simulate_partial_fill(quantity)

        # Determine status
        if filled_qty == quantity:
            status = OrderStatus.FILLED
        elif filled_qty > 0:
            status = OrderStatus.PARTIALLY_FILLED
        else:
            status = OrderStatus.REJECTED

        result = {
            'status': status,
            'symbol': symbol,
            'side': side,
            'requested_quantity': quantity,
            'filled_quantity': filled_qty,
            'unfilled_quantity': unfilled_qty,
            'quoted_price': price,
            'execution_price': execution_price,
            'market_impact': execution_price - price,
            'timestamp': timestamp
        }

        logger.info(
            f"Order result: {status.value}, filled {filled_qty}/{quantity} @ ${execution_price:.2f}"
        )

        return result


class ForcedBuyInReason(Enum):
    """Reasons for forced buy-in of short position."""
    BROKER_RECALL = "BROKER_RECALL"  # Broker recalls borrowed shares
    HARD_TO_BORROW = "HARD_TO_BORROW"  # Stock became hard to borrow
    DIVIDEND_APPROACHING = "DIVIDEND_APPROACHING"  # Ex-dividend date approaching
    CORPORATE_ACTION = "CORPORATE_ACTION"  # Merger, bankruptcy, etc.
    POSITION_LIMIT = "POSITION_LIMIT"  # Broker position limit exceeded


@dataclass
class ForcedBuyInEvent:
    """Event data for forced buy-in."""
    symbol: str
    quantity: int  # Shares to be bought in
    reason: ForcedBuyInReason
    timestamp: datetime
    deadline: datetime  # When buy-in must be completed
    penalty_rate: float = 0.05  # Penalty if not closed by deadline (5%)

    def to_dict(self) -> Dict:
        """Convert to dictionary for logging."""
        return {
            'symbol': self.symbol,
            'quantity': self.quantity,
            'reason': self.reason.value,
            'timestamp': self.timestamp.isoformat(),
            'deadline': self.deadline.isoformat(),
            'penalty_rate': self.penalty_rate
        }


class ForcedBuyInManager:
    """
    Manages forced buy-in scenarios for short positions.

    Tracks short positions and triggers forced buy-ins when:
    - Broker recalls shares
    - Stock becomes hard to borrow
    - Dividend approaching
    - Corporate actions
    """

    def __init__(
        self,
        borrow_rate_threshold: float = 0.20,  # 20% borrow rate = forced buy-in
        dividend_lookback_days: int = 3  # Trigger 3 days before ex-dividend
    ):
        """
        Initialize forced buy-in manager.

        Args:
            borrow_rate_threshold: Borrow rate that triggers forced buy-in
            dividend_lookback_days: Days before ex-dividend to trigger buy-in
        """
        self.borrow_rate_threshold = borrow_rate_threshold
        self.dividend_lookback_days = dividend_lookback_days
        self.active_events: List[ForcedBuyInEvent] = []

    def check_borrow_rate_increase(
        self,
        symbol: str,
        old_rate: float,
        new_rate: float,
        quantity: int,
        timestamp: datetime
    ) -> Optional[ForcedBuyInEvent]:
        """
        Check if borrow rate increase triggers forced buy-in.

        Args:
            symbol: Stock symbol
            old_rate: Previous borrow rate
            new_rate: New borrow rate
            quantity: Short position size
            timestamp: Current timestamp

        Returns:
            ForcedBuyInEvent if triggered, None otherwise
        """
        if new_rate >= self.borrow_rate_threshold and new_rate > old_rate * 2:
            # Significant rate increase to hard-to-borrow territory
            deadline = timestamp + timedelta(days=1)  # 1 day to close

            event = ForcedBuyInEvent(
                symbol=symbol,
                quantity=quantity,
                reason=ForcedBuyInReason.HARD_TO_BORROW,
                timestamp=timestamp,
                deadline=deadline,
                penalty_rate=new_rate  # Use new borrow rate as penalty
            )

            self.active_events.append(event)
            logger.warning(
                f"Forced buy-in triggered for {symbol}: "
                f"Borrow rate increased from {old_rate:.2%} to {new_rate:.2%}"
            )

            return event

        return None

    def check_broker_recall(
        self,
        symbol: str,
        quantity: int,
        timestamp: datetime,
        notice_hours: int = 24
    ) -> ForcedBuyInEvent:
        """
        Simulate broker share recall (usually short notice).

        Args:
            symbol: Stock symbol
            quantity: Shares being recalled
            timestamp: Recall timestamp
            notice_hours: Hours given to close position

        Returns:
            ForcedBuyInEvent
        """
        deadline = timestamp + timedelta(hours=notice_hours)

        event = ForcedBuyInEvent(
            symbol=symbol,
            quantity=quantity,
            reason=ForcedBuyInReason.BROKER_RECALL,
            timestamp=timestamp,
            deadline=deadline,
            penalty_rate=0.10  # 10% penalty if miss deadline
        )

        self.active_events.append(event)
        logger.critical(
            f"BROKER RECALL for {symbol}: {quantity} shares must be covered by {deadline}"
        )

        return event

    def check_dividend_approaching(
        self,
        symbol: str,
        quantity: int,
        ex_dividend_date: datetime,
        current_time: datetime,
        dividend_amount: float,
        price: float
    ) -> Optional[ForcedBuyInEvent]:
        """
        Check if dividend is approaching (short must pay dividend).

        Args:
            symbol: Stock symbol
            quantity: Short position size
            ex_dividend_date: Ex-dividend date
            current_time: Current timestamp
            dividend_amount: Dividend per share
            price: Current stock price

        Returns:
            ForcedBuyInEvent if dividend is large relative to price
        """
        days_to_ex_dividend = (ex_dividend_date - current_time).days

        # If dividend is significant (> 2% of price) and approaching
        dividend_yield = dividend_amount / price

        if days_to_ex_dividend <= self.dividend_lookback_days and dividend_yield > 0.02:
            deadline = ex_dividend_date - timedelta(days=1)

            event = ForcedBuyInEvent(
                symbol=symbol,
                quantity=quantity,
                reason=ForcedBuyInReason.DIVIDEND_APPROACHING,
                timestamp=current_time,
                deadline=deadline,
                penalty_rate=dividend_yield  # Must pay dividend if not closed
            )

            self.active_events.append(event)
            logger.warning(
                f"Forced buy-in recommended for {symbol}: "
                f"{dividend_yield:.2%} dividend in {days_to_ex_dividend} days"
            )

            return event

        return None

    def check_corporate_action(
        self,
        symbol: str,
        quantity: int,
        action_type: str,
        action_date: datetime,
        current_time: datetime
    ) -> ForcedBuyInEvent:
        """
        Handle corporate action (merger, bankruptcy, delisting).

        Args:
            symbol: Stock symbol
            quantity: Short position size
            action_type: Type of corporate action
            action_date: When action occurs
            current_time: Current timestamp

        Returns:
            ForcedBuyInEvent
        """
        # Usually forced to close before action date
        deadline = action_date - timedelta(days=1)

        event = ForcedBuyInEvent(
            symbol=symbol,
            quantity=quantity,
            reason=ForcedBuyInReason.CORPORATE_ACTION,
            timestamp=current_time,
            deadline=deadline,
            penalty_rate=0.20  # High penalty for corporate actions
        )

        self.active_events.append(event)
        logger.critical(
            f"CORPORATE ACTION for {symbol} ({action_type}): "
            f"Position must be closed by {deadline}"
        )

        return event

    def get_active_events(self, symbol: Optional[str] = None) -> List[ForcedBuyInEvent]:
        """
        Get active forced buy-in events.

        Args:
            symbol: Filter by symbol (optional)

        Returns:
            List of active events
        """
        if symbol:
            return [e for e in self.active_events if e.symbol == symbol]
        return self.active_events.copy()

    def resolve_event(self, event: ForcedBuyInEvent, timestamp: datetime) -> Dict:
        """
        Mark event as resolved (position closed).

        Args:
            event: Event to resolve
            timestamp: When position was closed

        Returns:
            Resolution details
        """
        if timestamp > event.deadline:
            # Missed deadline - apply penalty
            penalty = event.penalty_rate
            logger.error(
                f"Forced buy-in deadline MISSED for {event.symbol}: "
                f"{penalty:.2%} penalty applied"
            )
        else:
            penalty = 0.0
            logger.info(f"Forced buy-in completed for {event.symbol}")

        # Remove from active events
        if event in self.active_events:
            self.active_events.remove(event)

        return {
            'symbol': event.symbol,
            'reason': event.reason.value,
            'closed_on_time': timestamp <= event.deadline,
            'penalty_rate': penalty,
            'deadline': event.deadline,
            'actual_close_time': timestamp
        }
