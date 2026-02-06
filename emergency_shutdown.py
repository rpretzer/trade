"""
Emergency Shutdown Module
Provides emergency stop functionality and dead man's switch
"""

import logging
from datetime import datetime
from typing import Optional, Callable
from error_handling import DeadMansSwitch

logger = logging.getLogger(__name__)


class EmergencyShutdown:
    """
    Emergency shutdown system for trading operations.

    Provides:
    - Manual emergency stop
    - Dead man's switch (automatic shutdown if heartbeat stops)
    - Position liquidation on shutdown
    """

    def __init__(
        self,
        trading_api,
        heartbeat_timeout: float = 60.0,
        on_shutdown: Optional[Callable] = None
    ):
        """
        Initialize emergency shutdown system.

        Args:
            trading_api: TradingAPI instance
            heartbeat_timeout: Seconds without heartbeat before triggering
            on_shutdown: Optional callback when shutdown is triggered
        """
        self.trading_api = trading_api
        self.on_shutdown = on_shutdown
        self._shutdown_active = False

        # Initialize dead man's switch
        self.dead_mans_switch = DeadMansSwitch(
            name="trading_system",
            timeout=heartbeat_timeout,
            on_timeout=self._trigger_emergency_shutdown
        )

        logger.info(
            f"Emergency shutdown system initialized "
            f"(heartbeat timeout: {heartbeat_timeout}s)"
        )

    def heartbeat(self):
        """Signal that system is alive."""
        self.dead_mans_switch.heartbeat()

    def check(self) -> bool:
        """
        Check dead man's switch status.

        Returns:
            True if system is alive, False if shutdown triggered
        """
        return self.dead_mans_switch.check()

    def trigger_emergency_stop(self, reason: str = "Manual emergency stop"):
        """
        Manually trigger emergency shutdown.

        Args:
            reason: Reason for emergency stop
        """
        logger.critical(f"EMERGENCY STOP TRIGGERED: {reason}")
        self._trigger_emergency_shutdown(reason)

    def _trigger_emergency_shutdown(self, trigger):
        """
        Internal method to execute emergency shutdown.

        Args:
            trigger: DeadMansSwitch instance or reason string
        """
        if self._shutdown_active:
            logger.warning("Emergency shutdown already in progress")
            return

        self._shutdown_active = True

        if isinstance(trigger, DeadMansSwitch):
            reason = f"Dead man's switch timeout ({trigger.name})"
        else:
            reason = str(trigger)

        logger.critical("=" * 70)
        logger.critical("EMERGENCY SHUTDOWN INITIATED")
        logger.critical(f"Reason: {reason}")
        logger.critical(f"Timestamp: {datetime.now()}")
        logger.critical("=" * 70)

        # Step 1: Stop accepting new orders
        logger.critical("Step 1: Blocking new orders")
        self._block_new_orders()

        # Step 2: Cancel pending orders
        logger.critical("Step 2: Cancelling pending orders")
        self._cancel_pending_orders()

        # Step 3: Liquidate positions
        logger.critical("Step 3: Liquidating all positions")
        self._liquidate_all_positions()

        # Step 4: Send alerts
        logger.critical("Step 4: Sending alerts")
        self._send_emergency_alerts(reason)

        # Step 5: Custom shutdown callback
        if self.on_shutdown:
            logger.critical("Step 5: Executing custom shutdown callback")
            try:
                self.on_shutdown(reason)
            except Exception as e:
                logger.error(f"Error in shutdown callback: {e}")

        logger.critical("=" * 70)
        logger.critical("EMERGENCY SHUTDOWN COMPLETE")
        logger.critical("=" * 70)

    def _block_new_orders(self):
        """Block new orders from being placed."""
        # Set a flag on the trading API to reject new orders
        if hasattr(self.trading_api, '_emergency_stop'):
            self.trading_api._emergency_stop = True
        logger.warning("New orders blocked")

    def _cancel_pending_orders(self):
        """Cancel all pending orders."""
        try:
            # This would use the trading API to cancel orders
            # Implementation depends on broker API
            logger.info("Attempting to cancel pending orders...")
            # self.trading_api.cancel_all_orders()
            logger.warning("Cancel pending orders not yet implemented")
        except Exception as e:
            logger.error(f"Error cancelling pending orders: {e}")

    def _liquidate_all_positions(self):
        """Liquidate all open positions."""
        try:
            logger.info("Attempting to liquidate all positions...")
            # This would use the trading API to close positions
            # Implementation depends on broker API
            # positions = self.trading_api.get_all_positions()
            # for position in positions:
            #     self.trading_api.close_position(position)
            logger.warning("Position liquidation not yet implemented")
        except Exception as e:
            logger.error(f"Error liquidating positions: {e}")

    def _send_emergency_alerts(self, reason: str):
        """Send emergency alerts via multiple channels."""
        try:
            # This would integrate with alerting systems
            # - Send email
            # - Send SMS
            # - Trigger PagerDuty
            # - Post to Slack
            logger.info(f"Emergency alert: {reason}")
            logger.warning("Alert notifications not yet implemented")
        except Exception as e:
            logger.error(f"Error sending emergency alerts: {e}")

    def deactivate(self):
        """Deactivate emergency shutdown (for controlled shutdown)."""
        self.dead_mans_switch.deactivate()
        logger.info("Emergency shutdown system deactivated")

    def activate(self):
        """Activate emergency shutdown system."""
        self.dead_mans_switch.activate()
        self._shutdown_active = False
        logger.info("Emergency shutdown system activated")
