"""
Transaction Cost Model
Calculates realistic trading costs for backtesting
"""


class TransactionCostModel:
    """
    Models realistic transaction costs for stock trading.

    Includes:
    - Commission fees
    - SEC fees
    - Exchange fees
    - Slippage
    - Bid-ask spread
    - Short borrow costs
    """

    def __init__(self,
                 commission_per_share=0.005,
                 sec_fee_rate=0.0000278,  # $27.80 per million
                 exchange_fee_rate=0.000013,  # ~$0.0003 per $100
                 slippage_bps=1.0,  # 1 basis point = 0.01%
                 short_borrow_rate_annual=0.01):  # 1% annual
        """
        Initialize transaction cost model.

        Args:
            commission_per_share: Commission per share (default $0.005)
            sec_fee_rate: SEC fee rate per dollar sold
            exchange_fee_rate: Exchange fee rate per dollar
            slippage_bps: Slippage in basis points (default 1bp)
            short_borrow_rate_annual: Annual short borrow rate (default 1%)
        """
        self.commission_per_share = commission_per_share
        self.sec_fee_rate = sec_fee_rate
        self.exchange_fee_rate = exchange_fee_rate
        self.slippage_bps = slippage_bps / 10000  # Convert to decimal
        self.short_borrow_rate_daily = short_borrow_rate_annual / 365

    def calculate_commission(self, quantity):
        """
        Calculate commission fees.

        Args:
            quantity: Number of shares

        Returns:
            float: Commission cost
        """
        return abs(quantity) * self.commission_per_share

    def calculate_sec_fee(self, quantity, price, side):
        """
        Calculate SEC fees (only on sells).

        Args:
            quantity: Number of shares
            price: Price per share
            side: 'BUY' or 'SELL'

        Returns:
            float: SEC fee
        """
        if side == 'SELL':
            dollar_amount = abs(quantity) * price
            return dollar_amount * self.sec_fee_rate
        return 0.0

    def calculate_exchange_fee(self, quantity, price):
        """
        Calculate exchange fees.

        Args:
            quantity: Number of shares
            price: Price per share

        Returns:
            float: Exchange fee
        """
        dollar_amount = abs(quantity) * price
        return dollar_amount * self.exchange_fee_rate

    def calculate_slippage(self, quantity, price, side, volatility_multiplier=1.0):
        """
        Calculate slippage cost.

        Slippage increases with:
        - Order size (larger orders = more slippage)
        - Volatility (more volatile = more slippage)

        Args:
            quantity: Number of shares
            price: Price per share
            side: 'BUY' or 'SELL'
            volatility_multiplier: Multiplier for volatile conditions (default 1.0)

        Returns:
            float: Slippage cost
        """
        # Base slippage
        base_slippage = price * self.slippage_bps * volatility_multiplier

        # Slippage increases with order size (simplified model)
        # For every 1000 shares, add 0.5bp more slippage
        size_multiplier = 1.0 + (abs(quantity) / 1000) * 0.005

        slippage_per_share = base_slippage * size_multiplier

        # For buys, we pay more; for sells, we receive less
        if side == 'BUY':
            total_slippage = abs(quantity) * slippage_per_share
        else:  # SELL
            total_slippage = abs(quantity) * slippage_per_share

        return total_slippage

    def calculate_spread_cost(self, quantity, bid, ask, side):
        """
        Calculate bid-ask spread cost.

        Args:
            quantity: Number of shares
            bid: Bid price
            ask: Ask price
            side: 'BUY' or 'SELL'

        Returns:
            float: Spread cost
        """
        if bid is None or ask is None or bid >= ask:
            # Estimate spread as 0.05% if not available
            mid_price = (bid + ask) / 2 if (bid and ask) else None
            if mid_price:
                spread = mid_price * 0.0005
            else:
                return 0.0
        else:
            spread = ask - bid

        # Pay half the spread
        spread_cost = abs(quantity) * (spread / 2)

        return spread_cost

    def calculate_short_borrow_cost(self, quantity, price, days_held, borrow_rate_annual=None):
        """
        Calculate cost of borrowing shares for short positions.

        Args:
            quantity: Number of shares (negative for short)
            price: Price per share
            days_held: Number of days position was held
            borrow_rate_annual: Override annual borrow rate (for hard-to-borrow stocks)

        Returns:
            float: Borrow cost
        """
        if quantity >= 0:
            # Not a short position
            return 0.0

        borrow_rate = borrow_rate_annual if borrow_rate_annual else (self.short_borrow_rate_daily * 365)
        daily_rate = borrow_rate / 365

        short_value = abs(quantity) * price
        borrow_cost = short_value * daily_rate * days_held

        return borrow_cost

    def calculate_total_cost(self, quantity, price, side, bid=None, ask=None,
                            volatility_multiplier=1.0, days_held=0,
                            borrow_rate_annual=None):
        """
        Calculate total transaction cost for a trade.

        Args:
            quantity: Number of shares
            price: Execution price
            side: 'BUY' or 'SELL'
            bid: Bid price (optional, for spread calculation)
            ask: Ask price (optional, for spread calculation)
            volatility_multiplier: Multiplier for slippage in volatile conditions
            days_held: Days position was held (for short borrow costs)
            borrow_rate_annual: Override borrow rate for hard-to-borrow stocks

        Returns:
            dict: Breakdown of costs
        """
        costs = {
            'commission': self.calculate_commission(quantity),
            'sec_fee': self.calculate_sec_fee(quantity, price, side),
            'exchange_fee': self.calculate_exchange_fee(quantity, price),
            'slippage': self.calculate_slippage(quantity, price, side, volatility_multiplier),
            'spread': self.calculate_spread_cost(quantity, bid, ask, side) if bid and ask else 0.0,
            'borrow_cost': self.calculate_short_borrow_cost(
                -abs(quantity) if side == 'SELL' else 0,
                price,
                days_held,
                borrow_rate_annual
            )
        }

        costs['total'] = sum(costs.values())

        return costs

    def calculate_round_trip_cost(self, quantity, entry_price, exit_price,
                                  days_held=1, volatility_multiplier=1.0):
        """
        Calculate cost of a complete round-trip trade (entry + exit).

        Args:
            quantity: Number of shares
            entry_price: Entry price
            exit_price: Exit price
            days_held: Days position was held
            volatility_multiplier: Slippage multiplier

        Returns:
            dict: Breakdown of round-trip costs
        """
        # Entry costs (buy)
        entry_costs = self.calculate_total_cost(
            quantity, entry_price, 'BUY',
            volatility_multiplier=volatility_multiplier
        )

        # Exit costs (sell)
        exit_costs = self.calculate_total_cost(
            quantity, exit_price, 'SELL',
            volatility_multiplier=volatility_multiplier,
            days_held=days_held
        )

        return {
            'entry': entry_costs,
            'exit': exit_costs,
            'total': entry_costs['total'] + exit_costs['total'],
            'percentage': ((entry_costs['total'] + exit_costs['total']) /
                          (quantity * entry_price)) * 100
        }


# Example usage
if __name__ == "__main__":
    model = TransactionCostModel()

    print("Transaction Cost Model Example")
    print("=" * 70)

    # Example trade: Buy 100 shares of AAPL at $150
    quantity = 100
    price = 150.0
    bid = 149.95
    ask = 150.05

    costs = model.calculate_total_cost(
        quantity=quantity,
        price=price,
        side='BUY',
        bid=bid,
        ask=ask
    )

    print(f"\nBuy {quantity} shares @ ${price:.2f}")
    print(f"  Commission:   ${costs['commission']:.2f}")
    print(f"  SEC Fee:      ${costs['sec_fee']:.2f}")
    print(f"  Exchange Fee: ${costs['exchange_fee']:.2f}")
    print(f"  Slippage:     ${costs['slippage']:.2f}")
    print(f"  Spread:       ${costs['spread']:.2f}")
    print(f"  Borrow Cost:  ${costs['borrow_cost']:.2f}")
    print(f"  TOTAL:        ${costs['total']:.2f}")

    print(f"\nAs percentage of trade: {(costs['total'] / (quantity * price)) * 100:.4f}%")

    # Round-trip example
    print("\n" + "=" * 70)
    round_trip = model.calculate_round_trip_cost(
        quantity=100,
        entry_price=150.0,
        exit_price=152.0,
        days_held=5
    )

    print(f"Round-trip: Buy 100 @ $150, Sell 100 @ $152 (5 days)")
    print(f"  Entry costs: ${round_trip['entry']['total']:.2f}")
    print(f"  Exit costs:  ${round_trip['exit']['total']:.2f}")
    print(f"  TOTAL COST:  ${round_trip['total']:.2f}")
    print(f"  As % of entry: {round_trip['percentage']:.4f}%")

    gross_profit = (152 - 150) * 100
    net_profit = gross_profit - round_trip['total']
    print(f"\n  Gross profit: ${gross_profit:.2f}")
    print(f"  Net profit:   ${net_profit:.2f}")
    print(f"  Cost impact:  {(round_trip['total'] / gross_profit) * 100:.1f}% of gross profit")
