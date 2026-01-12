"""
Trading API Module
Connects to TD Ameritrade API for live trading
Includes paper trading mode for safe testing
"""

import os
import json
from datetime import datetime
import time

# Try importing TD Ameritrade client
try:
    from td.client import TDClient
    TD_AVAILABLE = True
except ImportError:
    TD_AVAILABLE = False
    print("Warning: TD Ameritrade client not available. Install with: pip install td-ameritrade-python-api")

class TradingAPI:
    """Trading API wrapper for TD Ameritrade with paper trading support."""
    
    def __init__(self, paper_trading=True, client_id=None, redirect_uri='http://localhost', 
                 account_id=None, credentials_path='td_credentials.json'):
        """
        Initialize trading API.
        
        Args:
            paper_trading: If True, simulate trades without real money
            client_id: TD Ameritrade client ID
            redirect_uri: Redirect URI for OAuth
            account_id: TD Ameritrade account ID
            credentials_path: Path to credentials file
        """
        self.paper_trading = paper_trading
        self.client_id = client_id or os.environ.get('TD_CLIENT_ID')
        self.redirect_uri = redirect_uri or os.environ.get('TD_REDIRECT_URI', 'http://localhost')
        self.account_id = account_id or os.environ.get('TD_ACCOUNT_ID')
        self.credentials_path = credentials_path
        self.session = None
        self.trade_history = []
        
        if not paper_trading and not TD_AVAILABLE:
            raise ImportError("TD Ameritrade API not available. Use paper trading mode or install td-ameritrade-python-api")
    
    def connect(self):
        """Connect to TD Ameritrade API (or initialize paper trading mode)."""
        if self.paper_trading:
            print("📝 Paper Trading Mode: All trades will be simulated")
            self.session = None  # No API connection needed for paper trading
            return True
        
        if not TD_AVAILABLE:
            print("Error: TD Ameritrade API not available")
            return False
        
        try:
            # Initialize TD Client
            self.session = TDClient(
                client_id=self.client_id,
                redirect_uri=self.redirect_uri
            )
            
            # Try to load saved credentials
            if os.path.exists(self.credentials_path):
                try:
                    with open(self.credentials_path, 'r') as f:
                        credentials = json.load(f)
                        self.session.access_token = credentials.get('access_token')
                        self.session.refresh_token = credentials.get('refresh_token')
                except Exception:
                    pass
            
            # Login (this will prompt for credentials if needed)
            if not hasattr(self.session, 'access_token') or not self.session.access_token:
                print("Please authenticate with TD Ameritrade...")
                print("You may need to complete OAuth flow in your browser")
                # Note: Full OAuth flow requires user interaction
                # For now, we'll use paper trading mode
                print("Switching to paper trading mode...")
                self.paper_trading = True
                return True
            
            return True
            
        except Exception as e:
            print(f"Error connecting to TD Ameritrade: {e}")
            print("Switching to paper trading mode...")
            self.paper_trading = True
            return True
    
    def get_account_balance(self):
        """Get account balance."""
        if self.paper_trading:
            # Simulated balance for paper trading
            return {
                'available_funds': 10000.0,
                'buying_power': 20000.0,
                'total_value': 10000.0,
                'paper_trading': True
            }
        
        try:
            accounts = self.session.get_accounts(fields=['positions', 'orders'])
            if accounts and len(accounts) > 0:
                account = accounts[0] if not self.account_id else \
                          next((a for a in accounts if a['securitiesAccount']['accountId'] == self.account_id), None)
                
                if account:
                    securities = account['securitiesAccount']
                    return {
                        'available_funds': securities.get('currentBalances', {}).get('availableFunds', 0),
                        'buying_power': securities.get('currentBalances', {}).get('buyingPower', 0),
                        'total_value': securities.get('currentBalances', {}).get('totalValue', 0),
                        'paper_trading': False
                    }
            return None
        except Exception as e:
            print(f"Error fetching account balance: {e}")
            return None
    
    def get_position(self, symbol):
        """Get current position for a symbol."""
        if self.paper_trading:
            # In paper trading, track positions in memory
            # For simplicity, return no position
            return {'quantity': 0, 'average_price': 0}
        
        try:
            accounts = self.session.get_accounts(fields=['positions'])
            if accounts:
                account = accounts[0] if not self.account_id else \
                          next((a for a in accounts if a['securitiesAccount']['accountId'] == self.account_id), None)
                
                if account:
                    positions = account['securitiesAccount'].get('positions', [])
                    for pos in positions:
                        if pos['instrument']['symbol'] == symbol:
                            return {
                                'quantity': pos['longQuantity'] - pos['shortQuantity'],
                                'average_price': pos['averagePrice']
                            }
            return {'quantity': 0, 'average_price': 0}
        except Exception as e:
            print(f"Error fetching position for {symbol}: {e}")
            return {'quantity': 0, 'average_price': 0}
    
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
        
        if not self.session:
            print("Error: Not connected to TD Ameritrade API")
            return None
        
        try:
            if not self.account_id:
                print("Error: Account ID not set")
                return None
            
            # Build order payload
            order_payload = {
                "orderType": order_type,
                "session": "NORMAL",
                "duration": "DAY",
                "orderStrategyType": "SINGLE",
                "orderLegCollection": [
                    {
                        "instruction": side,
                        "quantity": quantity,
                        "instrument": {
                            "symbol": symbol,
                            "assetType": "EQUITY"
                        }
                    }
                ]
            }
            
            if order_type == 'LIMIT' and price:
                order_payload["price"] = price
            
            # Place order
            response = self.session.place_order(
                account=self.account_id,
                order=order_payload
            )
            
            return response
            
        except Exception as e:
            print(f"Error placing order: {e}")
            return None
    
    def execute_arbitrage_trade(self, signal, account_balance, max_position_size=0.1):
        """
        Execute arbitrage trade based on signal.
        
        Args:
            signal: Trading signal ('Buy AAPL, Sell MSFT', 'Buy MSFT, Sell AAPL', or 'Hold')
            account_balance: Account balance dictionary
            max_position_size: Maximum percentage of account to use per trade (default 0.1 = 10%)
            
        Returns:
            List of executed orders
        """
        orders = []
        
        if signal == 'Hold':
            print("No trade: Holding position")
            return orders
        
        available_funds = account_balance.get('available_funds', 0)
        position_value = available_funds * max_position_size
        
        try:
            if signal == 'Buy AAPL, Sell MSFT':
                # Buy AAPL, Sell MSFT
                # For simplicity, we'll use equal dollar amounts
                print(f"Executing: Buy AAPL, Sell MSFT")
                print(f"Using ${position_value:.2f} per position")
                
                # Place buy order for AAPL
                order1 = self.place_order('AAPL', int(position_value / 150), 'BUY', 'MARKET')
                if order1:
                    orders.append(order1)
                    print(f"✓ Buy order placed for AAPL")
                
                # Place sell order for MSFT
                order2 = self.place_order('MSFT', int(position_value / 300), 'SELL', 'MARKET')
                if order2:
                    orders.append(order2)
                    print(f"✓ Sell order placed for MSFT")
                    
            elif signal == 'Buy MSFT, Sell AAPL':
                # Buy MSFT, Sell AAPL
                print(f"Executing: Buy MSFT, Sell AAPL")
                print(f"Using ${position_value:.2f} per position")
                
                # Place buy order for MSFT
                order1 = self.place_order('MSFT', int(position_value / 300), 'BUY', 'MARKET')
                if order1:
                    orders.append(order1)
                    print(f"✓ Buy order placed for MSFT")
                
                # Place sell order for AAPL
                order2 = self.place_order('AAPL', int(position_value / 150), 'SELL', 'MARKET')
                if order2:
                    orders.append(order2)
                    print(f"✓ Sell order placed for AAPL")
        
        except Exception as e:
            print(f"Error executing trade: {e}")
        
        return orders
    
    def get_trade_history(self):
        """Get trade history."""
        if self.paper_trading:
            return self.trade_history
        else:
            # Fetch from TD Ameritrade API
            try:
                orders = self.session.get_orders(self.account_id)
                return orders
            except Exception as e:
                print(f"Error fetching trade history: {e}")
                return []
