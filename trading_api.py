"""
Trading API Module
Connects to Charles Schwab API for live trading
Includes paper trading mode for safe testing

Note: TD Ameritrade API was deprecated after Schwab's acquisition.
This module now uses schwab-py (https://schwab-py.readthedocs.io/)
"""

import os
import json
from datetime import datetime
import time
import asyncio
import logging
from exceptions import (
    APIConnectionError, APIAuthenticationError, APIResponseError,
    PriceNotAvailableError, StaleDataError, InvalidOrderError,
    InsufficientFundsError, OrderRejectedError
)
from constants import (
    TICKER_LONG, TICKER_SHORT,
    SIGNAL_BUY_LONG, SIGNAL_BUY_SHORT, SIGNAL_HOLD,
    PRICE_SANITY_RANGES,
)
from error_handling import CircuitBreaker, retry, RetryConfig, RateLimiter

logger = logging.getLogger(__name__)

# Try importing Schwab client
try:
    from schwab.client import AsyncClient
    from schwab.auth import client_from_token_file, client_from_manual_flow
    SCHWAB_AVAILABLE = True
except ImportError:
    SCHWAB_AVAILABLE = False
    logger.warning("Schwab client not available. Install with: pip install schwab-py")
    logger.warning("TD Ameritrade API is deprecated. Schwab API requires Python 3.10+")

class TradingAPI:
    """
    Trading API wrapper for Charles Schwab API with paper trading support.
    
    Note: TD Ameritrade API was deprecated after Schwab's acquisition (May 2024).
    This class now uses schwab-py library. Key differences:
    - Tokens expire in 7 days (not 90 days)
    - Must create new app (old TDA apps don't work)
    - Uses async/await syntax
    - Requires Python 3.10+
    - Callback URL must use 127.0.0.1 (not localhost)
    
    See: https://schwab-py.readthedocs.io/en/latest/tda-transition.html
    """
    
    def __init__(self, paper_trading=True, app_key=None, app_secret=None, 
                 redirect_uri='http://127.0.0.1', account_id=None, 
                 credentials_path='schwab_credentials.json'):
        """
        Initialize trading API.
        
        Args:
            paper_trading: If True, simulate trades without real money
            app_key: Schwab app key (from developer portal)
            app_secret: Schwab app secret (from developer portal)
            redirect_uri: Redirect URI for OAuth (must use 127.0.0.1, not localhost)
            account_id: Schwab account ID (hash value)
            credentials_path: Path to credentials file
        """
        self.paper_trading = paper_trading
        self.credentials_path = credentials_path
        self.client = None
        self.trade_history = []

        # Initialize circuit breakers
        self._api_circuit_breaker = CircuitBreaker(
            name="schwab_api",
            failure_threshold=5,
            recovery_timeout=60.0,
            on_open=self._on_circuit_breaker_open
        )

        self._price_circuit_breaker = CircuitBreaker(
            name="price_data",
            failure_threshold=3,
            recovery_timeout=30.0
        )

        # Initialize rate limiter (Schwab allows ~120 requests/minute)
        self._rate_limiter = RateLimiter(
            name="schwab_api",
            rate=2.0,  # 2 requests per second = 120/minute
            capacity=10  # Allow bursts of up to 10 requests
        )
        
        # Try to load app credentials from encrypted storage if not provided
        if not app_key or not app_secret:
            try:
                from credential_manager import CredentialManager
                cred_manager = CredentialManager()
                loaded_key, loaded_secret = cred_manager.load_credentials()
                if loaded_key and loaded_secret:
                    app_key = app_key or loaded_key
                    app_secret = app_secret or loaded_secret
            except ImportError as e:
                logger.error(
                    "CredentialManager not available. "
                    "Install cryptography: pip install cryptography"
                )
            except Exception as e:
                logger.error(f"Error loading credentials: {e}")
        
        self.app_key = app_key or os.environ.get('SCHWAB_APP_KEY')
        self.app_secret = app_secret or os.environ.get('SCHWAB_APP_SECRET')
        self.redirect_uri = redirect_uri or os.environ.get('SCHWAB_REDIRECT_URI', 'http://127.0.0.1')
        self.account_id = account_id or os.environ.get('SCHWAB_ACCOUNT_ID')
        
        if not paper_trading and not SCHWAB_AVAILABLE:
            raise ImportError("Schwab API not available. Use paper trading mode or install schwab-py (requires Python 3.10+)")

    def _on_circuit_breaker_open(self, circuit_breaker):
        """
        Callback when circuit breaker opens.

        Args:
            circuit_breaker: CircuitBreaker instance that opened
        """
        logger.critical(
            f"CIRCUIT BREAKER OPENED: {circuit_breaker.name} - "
            f"API calls blocked for {circuit_breaker.recovery_timeout}s"
        )
        # Could trigger alerts here
        # Could initiate emergency shutdown if critical
    
    def connect(self):
        """
        Connect to Schwab API (or initialize paper trading mode).
        
        Note: Schwab tokens expire in 7 days. You must regenerate them
        using the OAuth flow. See schwab-py documentation for details.
        """
        if self.paper_trading:
            logger.info("Paper Trading Mode: All trades will be simulated")
            self.client = None  # No API connection needed for paper trading
            return True
        
        if not SCHWAB_AVAILABLE:
            logger.error("Schwab API not available")
            logger.error("Install with: pip install schwab-py")
            logger.error("Requires Python 3.10+")
            return False
        
        try:
            # Try to load saved credentials
            if os.path.exists(self.credentials_path):
                try:
                    # schwab-py can load from token file
                    self.client = client_from_token_file(
                        self.credentials_path,
                        self.app_key,
                        self.app_secret
                    )
                    if self.client:
                        logger.info("Connected to Schwab API using saved credentials")
                        return True
                except Exception as e:
                    logger.warning("Could not load saved credentials: %s", e)
                    logger.warning("You may need to regenerate your token (they expire in 7 days)")
            
            # If no valid credentials, need to run OAuth flow
            if not self.client:
                logger.warning("Schwab API Authentication Required")
                logger.info("To connect to Schwab API:")
                logger.info("1. Create a new app at: https://developer.schwab.com")
                logger.info("2. Note: App approval can take multiple days")
                logger.info("3. Use 127.0.0.1 (not localhost) as callback URL")
                logger.info("4. Run the OAuth flow (requires browser interaction)")
                logger.info("5. Tokens expire in 7 days (not 90 days like TDA)")
                logger.warning("For now, switching to paper trading mode...")
                logger.info("See: https://schwab-py.readthedocs.io/en/latest/")
                self.paper_trading = True
                return True
            
            return True
            
        except Exception as e:
            logger.error("Error connecting to Schwab API: %s", e)
            logger.warning("Switching to paper trading mode...")
            self.paper_trading = True
            return True
    
    async def _get_account_balance_async(self):
        """Async helper to get account balance from Schwab API."""
        if not self.client:
            return None
        
        try:
            # Get all accounts
            accounts_response = await self.client.get_accounts()
            accounts = accounts_response.json()
            
            if accounts and len(accounts) > 0:
                # Find the account (by hash if account_id provided)
                account = accounts[0]
                if self.account_id:
                    account = next((a for a in accounts if a.get('hashValue') == self.account_id), account)
                
                if account:
                    return {
                        'available_funds': account.get('currentBalances', {}).get('cashAvailableForTrading', 0),
                        'buying_power': account.get('currentBalances', {}).get('buyingPower', 0),
                        'total_value': account.get('currentBalances', {}).get('totalValue', 0),
                        'paper_trading': False
                    }
            return None
        except Exception as e:
            logger.error("Error fetching account balance: %s", e)
            return None
    
    def get_account_balance(self):
        """Get account balance (synchronous wrapper for async method)."""
        if self.paper_trading:
            # Simulated balance for paper trading
            return {
                'available_funds': 10000.0,
                'buying_power': 20000.0,
                'total_value': 10000.0,
                'paper_trading': True
            }
        
        if not self.client:
            return None
        
        try:
            # Run async method in event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is already running, we need to use a different approach
                # For now, return None and suggest using async methods
                logger.warning("Cannot run async method in existing event loop")
                logger.warning("Consider using async methods directly or running in a new thread")
                return None
            else:
                return loop.run_until_complete(self._get_account_balance_async())
        except RuntimeError:
            # No event loop, create one
            return asyncio.run(self._get_account_balance_async())
        except Exception as e:
            logger.error("Error fetching account balance: %s", e)
            return None
    
    async def _get_current_price_async(self, symbol):
        """Async helper to get current price from Schwab API."""
        if not self.client:
            return None

        try:
            # Get quote from Schwab API
            quote_response = await self.client.get_quote(symbol)
            quote = quote_response.json() if hasattr(quote_response, 'json') else quote_response

            if quote and symbol in quote:
                quote_data = quote[symbol]
                # Get last price or mark price
                price = quote_data.get('lastPrice') or quote_data.get('mark')

                if price and price > 0:
                    return {
                        'price': price,
                        'timestamp': datetime.now(),
                        'bid': quote_data.get('bidPrice'),
                        'ask': quote_data.get('askPrice'),
                        'volume': quote_data.get('totalVolume')
                    }
            return None
        except Exception as e:
            logger.error("Error fetching price for %s: %s", symbol, e)
            return None

    def get_current_price(self, symbol, max_staleness_seconds=2):
        """
        Get current market price for a symbol.

        Args:
            symbol: Stock symbol
            max_staleness_seconds: Maximum age of price data (default 2 seconds)

        Returns:
            float: Current price

        Raises:
            PriceNotAvailableError: If price cannot be fetched
            StaleDataError: If price data is too old
            APIConnectionError: If API connection fails
        """
        # Acquire rate limit token
        if not self.paper_trading:
            if not self._rate_limiter.acquire():
                logger.warning(f"Rate limit reached for price fetch: {symbol}")
                time.sleep(0.5)  # Brief wait before proceeding
                self._rate_limiter.wait_and_acquire(timeout=5.0)

        def _fetch_price():
            if self.paper_trading:
                # For paper trading, fetch from yfinance
                try:
                    import yfinance as yf
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    # Try multiple price fields
                    price = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose')
                    if price and price > 0:
                        return price
                    raise PriceNotAvailableError(f"No valid price available for {symbol}")
                except Exception as e:
                    if isinstance(e, PriceNotAvailableError):
                        raise
                    raise PriceNotAvailableError(f"Error fetching price for {symbol}: {e}")

            if not self.client:
                raise APIConnectionError("API client not connected")

            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    raise APIConnectionError("Cannot run async method in existing event loop")
                else:
                    price_data = loop.run_until_complete(self._get_current_price_async(symbol))
            except RuntimeError:
                price_data = asyncio.run(self._get_current_price_async(symbol))
            except Exception as e:
                if isinstance(e, (APIConnectionError, PriceNotAvailableError)):
                    raise
                raise APIConnectionError(f"Error fetching price for {symbol}: {e}")

            if not price_data:
                raise PriceNotAvailableError(f"No price data returned for {symbol}")

            # Check staleness
            age = (datetime.now() - price_data['timestamp']).total_seconds()
            if age > max_staleness_seconds:
                raise StaleDataError(
                    f"Price for {symbol} is {age:.1f}s old (limit: {max_staleness_seconds}s)"
                )

            return price_data['price']

        # Use circuit breaker for price fetches
        try:
            return self._price_circuit_breaker.call(_fetch_price)
        except Exception as e:
            logger.error(f"Failed to fetch price for {symbol}: {e}")
            raise

    async def _get_position_async(self, symbol):
        """Async helper to get position from Schwab API."""
        if not self.client:
            return {'quantity': 0, 'average_price': 0}

        try:
            accounts_response = await self.client.get_accounts()
            accounts = accounts_response.json()

            if accounts:
                account = accounts[0]
                if self.account_id:
                    account = next((a for a in accounts if a.get('hashValue') == self.account_id), account)

                if account:
                    positions = account.get('positions', [])
                    for pos in positions:
                        instrument = pos.get('instrument', {})
                        if instrument.get('symbol') == symbol:
                            return {
                                'quantity': pos.get('longQuantity', 0) - pos.get('shortQuantity', 0),
                                'average_price': pos.get('averagePrice', 0)
                            }
            return {'quantity': 0, 'average_price': 0}
        except Exception as e:
            logger.error("Error fetching position for %s: %s", symbol, e)
            return {'quantity': 0, 'average_price': 0}
    
    def get_position(self, symbol):
        """Get current position for a symbol (synchronous wrapper)."""
        if self.paper_trading:
            # In paper trading, track positions in memory
            # Paper trading positions are tracked by live_trading.save_positions/load_positions.
            return {'quantity': 0, 'average_price': 0}
        
        if not self.client:
            return {'quantity': 0, 'average_price': 0}
        
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                logger.warning("Cannot run async method in existing event loop")
                return {'quantity': 0, 'average_price': 0}
            else:
                return loop.run_until_complete(self._get_position_async(symbol))
        except RuntimeError:
            return asyncio.run(self._get_position_async(symbol))
        except Exception as e:
            logger.error("Error fetching position for %s: %s", symbol, e)
            return {'quantity': 0, 'average_price': 0}
    
    async def _place_order_async(self, symbol, quantity, side, order_type='MARKET', price=None):
        """Async helper to place order via Schwab API."""
        if not self.client or not self.account_id:
            return None
        
        try:
            from schwab.orders import equity_buy_market, equity_sell_market, equity_buy_limit, equity_sell_limit
            
            # Build order based on type and side
            if order_type == 'MARKET':
                if side == 'BUY':
                    order = equity_buy_market(symbol, quantity)
                else:  # SELL
                    order = equity_sell_market(symbol, quantity)
            elif order_type == 'LIMIT' and price:
                if side == 'BUY':
                    order = equity_buy_limit(symbol, quantity, price)
                else:  # SELL
                    order = equity_sell_limit(symbol, quantity, price)
            else:
                logger.error("Unsupported order type: %s", order_type)
                return None
            
            # Place order
            response = await self.client.place_order(self.account_id, order)
            return response.json() if hasattr(response, 'json') else response
            
        except Exception as e:
            logger.error("Error placing order: %s", e)
            return None
    
    def place_order(self, symbol, quantity, side, order_type='MARKET', price=None):
        """
        Place an order.
        
        Args:
            symbol: Stock symbol
            quantity: Number of shares
            side: 'BUY' or 'SELL'
            order_type: 'MARKET' or 'LIMIT'
            price: Limit price (required for LIMIT orders)
            
        Returns:
            Order confirmation or None if failed
        """
        if self.paper_trading:
            # Simulate order execution
            order = {
                'symbol': symbol,
                'quantity': quantity,
                'side': side,
                'order_type': order_type,
                'price': price,
                'timestamp': datetime.now().isoformat(),
                'status': 'FILLED',
                'paper_trading': True
            }
            self.trade_history.append(order)
            return order
        
        if not self.client:
            logger.error("Not connected to Schwab API")
            return None
        
        try:
            if not self.account_id:
                logger.error("Account ID not set")
                return None
            
            # Run async method
            loop = asyncio.get_event_loop()
            if loop.is_running():
                logger.warning("Cannot run async method in existing event loop")
                logger.warning("Consider using async methods directly")
                return None
            else:
                return loop.run_until_complete(self._place_order_async(symbol, quantity, side, order_type, price))
        except RuntimeError:
            return asyncio.run(self._place_order_async(symbol, quantity, side, order_type, price))
        except Exception as e:
            logger.error("Error placing order: %s", e)
            return None
    
    def execute_arbitrage_trade(self, signal, account_balance, max_position_size=0.1):
        """
        Execute arbitrage trade based on signal.

        Args:
            signal: Trading signal (SIGNAL_BUY_LONG, SIGNAL_BUY_SHORT, or SIGNAL_HOLD)
            account_balance: Account balance dictionary
            max_position_size: Maximum percentage of account to use per trade (default 0.1 = 10%)

        Returns:
            List of executed orders
        """
        orders = []

        if signal == SIGNAL_HOLD:
            logger.info("No trade: Holding position")
            return orders

        if signal == SIGNAL_BUY_LONG:
            buy_sym, sell_sym = TICKER_LONG, TICKER_SHORT
        elif signal == SIGNAL_BUY_SHORT:
            buy_sym, sell_sym = TICKER_SHORT, TICKER_LONG
        else:
            logger.warning("Unrecognised signal '%s'. No trade.", signal)
            return orders

        available_funds = account_balance.get('available_funds', 0)
        position_value = available_funds * max_position_size

        try:
            logger.info("Executing: %s", signal)
            logger.info("Using $%.2f per position", position_value)

            # Fetch live prices for both legs
            buy_price  = self.get_current_price(buy_sym)
            sell_price = self.get_current_price(sell_sym)

            if not buy_price or not sell_price:
                logger.error("Could not fetch current prices. Aborting trade.")
                return orders

            # Sanity-check each price against its configured range
            for sym, price in [(buy_sym, buy_price), (sell_sym, sell_price)]:
                lo, hi = PRICE_SANITY_RANGES.get(sym, (0, float('inf')))
                if price < lo or price > hi:
                    logger.warning("%s price $%.2f seems unusual. Aborting trade.", sym, price)
                    return orders

            buy_qty  = int(position_value / buy_price)
            sell_qty = int(position_value / sell_price)

            logger.info("Current prices: %s=$%.2f, %s=$%.2f", buy_sym, buy_price, sell_sym, sell_price)
            logger.info("Order quantities: %s=%s shares, %s=%s shares", buy_sym, buy_qty, sell_sym, sell_qty)

            # Place buy order
            order1 = self.place_order(buy_sym, buy_qty, 'BUY', 'MARKET')
            if order1:
                orders.append(order1)
                logger.info("Buy order placed for %s: %s shares @ $%.2f", buy_sym, buy_qty, buy_price)

            # Place sell order
            order2 = self.place_order(sell_sym, sell_qty, 'SELL', 'MARKET')
            if order2:
                orders.append(order2)
                logger.info("Sell order placed for %s: %s shares @ $%.2f", sell_sym, sell_qty, sell_price)

        except Exception as e:
            logger.error("Error executing trade: %s", e)

        return orders
    
    async def _get_trade_history_async(self):
        """Async helper to get trade history from Schwab API."""
        if not self.client or not self.account_id:
            return []
        
        try:
            # Get orders for the account
            orders_response = await self.client.get_orders_by_path(self.account_id)
            orders = orders_response.json() if hasattr(orders_response, 'json') else orders_response
            return orders if isinstance(orders, list) else []
        except Exception as e:
            logger.error("Error fetching trade history: %s", e)
            return []
    
    def get_trade_history(self):
        """Get trade history (synchronous wrapper)."""
        if self.paper_trading:
            return self.trade_history
        else:
            if not self.client:
                return []
            
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    logger.warning("Cannot run async method in existing event loop")
                    return []
                else:
                    return loop.run_until_complete(self._get_trade_history_async())
            except RuntimeError:
                return asyncio.run(self._get_trade_history_async())
            except Exception as e:
                logger.error("Error fetching trade history: %s", e)
                return []
