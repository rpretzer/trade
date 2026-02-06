import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
from datetime import datetime, time
from transaction_costs import TransactionCostModel
from trading_execution import OrderExecutor, MarketImpact, BorrowCosts, MarginConfig, OrderSide, OrderStatus
from exceptions import OrderRejectedError, InsufficientFundsError, ShortNotAvailableError, RiskException
from risk_management import RiskManager, RiskLimits
from constants import (
    feature_names, TICKER_LONG, TICKER_SHORT,
    SIGNAL_BUY_LONG, SIGNAL_BUY_SHORT, SIGNAL_HOLD,
)

def load_processed_data(csv_file='processed_stock_data.csv'):
    """
    Load the processed stock data.
    
    Returns:
        df: DataFrame with processed data
        original_data: DataFrame with original price data
    """
    df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
    
    # Get original price data
    original_columns = [TICKER_LONG, TICKER_SHORT, 'Price_Difference']
    if all(col in df.columns for col in original_columns):
        original_data = df[original_columns].copy()
    else:
        original_data = None
    
    return df, original_data

def create_sequences(data, timesteps):
    """
    Create sequences for LSTM input (same as in train_model.py).
    
    Args:
        data: Array with features (shape: samples, features)
        timesteps: Number of previous timesteps to use
        
    Returns:
        X: Array of shape (samples, timesteps, features)
        y: Array of shape (samples,) - actual target values
    """
    X, y = [], []
    
    for i in range(timesteps, len(data)):
        X.append(data[i-timesteps:i])
        y.append(data[i, -1])  # Price difference is the last column
    
    return np.array(X), np.array(y)

def load_model_and_data(model_path='lstm_price_difference_model.h5', 
                        data_file='processed_stock_data.csv',
                        timesteps=60, test_size=0.2):
    """
    Load the trained model and prepare test data.
    
    Returns:
        model: Trained Keras model
        X_test: Test sequences
        y_test: Actual test targets
        test_dates: Dates corresponding to test predictions
        original_prices: Original price data for test set
    """
    print(f"Loading model from {model_path}...")
    model = keras.models.load_model(model_path)
    
    print(f"Loading data from {data_file}...")
    df, original_data = load_processed_data(data_file)
    
    feature_columns_normalized = feature_names(normalized=True)
    feature_columns_original   = feature_names(normalized=False)
    
    # Check what features are actually available
    available_normalized = [col for col in feature_columns_normalized if col in df.columns]
    available_original = [col for col in feature_columns_original if col in df.columns]
    
    # Use normalized if more are available, otherwise use original
    if len(available_normalized) > len(available_original) and len(available_normalized) > 0:
        feature_columns = available_normalized
        print(f"Using normalized features ({len(feature_columns)} features)")
        data = df[feature_columns].values
        scaler = None  # Data already normalized
    elif len(available_original) > 0:
        feature_columns = available_original
        print(f"Using original features and scaling ({len(feature_columns)} features)")
        scaler = StandardScaler()
        data = scaler.fit_transform(df[feature_columns])
    else:
        # Fallback to basic features if new ones aren't available
        print("Warning: Advanced features not found. Using basic features only.")
        basic_features = feature_names(basic_only=True)
        feature_columns = [col for col in basic_features if col in df.columns]
        if not feature_columns:
            feature_columns = feature_names(normalized=False)[:5]
            scaler = StandardScaler()
            data = scaler.fit_transform(df[feature_columns])
        else:
            data = df[feature_columns].values
            scaler = None
    
    print(f"Features included: {len(feature_columns)}")
    print(f"Feature list: {feature_columns[:10]}...")  # Show first 10 features
    
    # Create sequences
    X, y = create_sequences(data, timesteps)
    
    # Split into train/test (same as training)
    n_test = int(len(X) * test_size)
    X_test = X[-n_test:]
    y_test = y[-n_test:]
    
    # Get corresponding dates and original prices
    test_dates = df.index[timesteps:][-n_test:]
    
    if original_data is not None:
        test_price_data = original_data.iloc[timesteps:][-n_test:].copy()
    else:
        test_price_data = None
    
    print(f"Test set: {X_test.shape[0]} samples")
    return model, X_test, y_test, test_dates, test_price_data

def predict_next_day(model, X_test):
    """
    Use the trained LSTM model to predict the next day's price difference.
    
    Args:
        model: Trained Keras model
        X_test: Test sequences
        
    Returns:
        predictions: Predicted price differences
    """
    print("\nMaking predictions on test data...")
    predictions = model.predict(X_test, verbose=0)
    predictions = predictions.flatten()  # Convert to 1D array
    print(f"Generated {len(predictions)} predictions")
    return predictions

def check_risk(capital, initial_capital, max_drawdown=0.05):
    """
    Check if maximum drawdown limit has been reached.
    
    Args:
        capital: Current capital
        initial_capital: Starting capital
        max_drawdown: Maximum allowed drawdown (default 0.05 = 5%)
        
    Returns:
        True if trading should continue, False if drawdown limit reached
    """
    if initial_capital <= 0:
        return True
    
    drawdown = (initial_capital - capital) / initial_capital
    
    if drawdown > max_drawdown:
        return False
    return True

def apply_stop_loss(current_price, entry_price, stop_loss_pct=0.02, is_long=True):
    """
    Check if stop-loss should be triggered for a position.
    
    Args:
        current_price: Current price of the position
        entry_price: Entry price of the position
        stop_loss_pct: Stop-loss percentage (default 0.02 = 2%)
        is_long: True if long position, False if short position
        
    Returns:
        True if stop-loss should be triggered, False otherwise
    """
    if entry_price <= 0:
        return False
    
    if is_long:
        # For long positions: trigger if price drops by stop_loss_pct
        loss_pct = (entry_price - current_price) / entry_price
        return loss_pct >= stop_loss_pct
    else:
        # For short positions: trigger if price rises by stop_loss_pct
        loss_pct = (current_price - entry_price) / entry_price
        return loss_pct >= stop_loss_pct

def generate_trading_signals(predictions, threshold=0.5):
    """
    Generate trading signals based on predictions.
    
    Args:
        predictions: Predicted price differences (normalized)
        threshold: Threshold for generating signals (in normalized scale)
        
    Returns:
        signals: List of trading signals ('Buy AAPL', 'Buy MSFT', or 'Hold')
    """
    signals = []
    
    for pred in predictions:
        if pred > threshold:
            signals.append(SIGNAL_BUY_LONG)
        elif pred < -threshold:
            signals.append(SIGNAL_BUY_SHORT)
        else:
            signals.append(SIGNAL_HOLD)
    
    return signals

def backtest_strategy(predictions, actual_differences, price_data, threshold=0.5, 
                     initial_capital=10000, max_drawdown=0.05, stop_loss_pct=0.02):
    """
    Backtest the trading strategy on test data with risk management.
    
    The strategy:
    - If predicted_diff > threshold: Buy AAPL, Sell MSFT (betting difference increases)
    - If predicted_diff < -threshold: Buy MSFT, Sell AAPL (betting difference decreases)
    - Otherwise: Hold (no trade)
    
    Risk Management:
    - Stop-loss: 2% per trade (default, configurable)
    - Max drawdown: 5% (default, configurable) - pauses trading if exceeded
    
    Profit calculation:
    - For "Buy AAPL, Sell MSFT": profit if actual_diff increases
    - For "Buy MSFT, Sell AAPL": profit if actual_diff decreases
    
    Args:
        predictions: Predicted price differences (normalized)
        actual_differences: Actual price differences
        price_data: DataFrame with AAPL and MSFT prices
        threshold: Threshold for trading signals (normalized)
        initial_capital: Starting capital
        max_drawdown: Maximum allowed drawdown (default 0.05 = 5%)
        stop_loss_pct: Stop-loss percentage per trade (default 0.02 = 2%)
        
    Returns:
        results: Dictionary with backtest results
    """
    print(f"\nBacktesting strategy with threshold={threshold}...")
    print(f"Initial capital: ${initial_capital:,.2f}")
    print(f"Risk Management:")
    print(f"  Max Drawdown: {max_drawdown*100:.1f}%")
    print(f"  Stop-Loss per trade: {stop_loss_pct*100:.1f}%")

    # Initialize transaction cost model
    cost_model = TransactionCostModel(
        commission_per_share=0.005,
        sec_fee_rate=0.0000278,
        slippage_bps=1.0,
        short_borrow_rate_annual=0.01
    )
    print(f"  Transaction costs: Commission $0.005/share, Slippage 1bp, SEC fees")

    # Initialize order executor for realistic execution
    borrow_costs = BorrowCosts(
        easy_to_borrow_rate=0.01,
        moderate_borrow_rate=0.05,
        hard_to_borrow_rate=0.15
    )
    margin_config = MarginConfig(
        margin_interest_rate=0.08,
        maintenance_margin=0.25,
        initial_margin=0.50
    )
    order_executor = OrderExecutor(
        borrow_costs=borrow_costs,
        margin_config=margin_config,
        allow_after_hours=False  # Enforce market hours
    )
    print(f"  Order execution: Market hours, shortable checks, partial fills enabled")

    # Initialize risk manager for pre-trade checks
    risk_limits = RiskLimits(
        max_position_size=1000,
        max_position_value=initial_capital * 0.5,
        max_total_exposure=initial_capital * 2.0,
        max_single_position_pct=0.15,
        max_drawdown_pct=max_drawdown,
        critical_drawdown_pct=max_drawdown * 2,
    )
    risk_manager = RiskManager(limits=risk_limits, initial_capital=initial_capital)
    print(f"  Risk manager: max position ${risk_limits.max_position_value:,.0f}, "
          f"max drawdown {risk_limits.max_drawdown_pct*100:.1f}%")

    capital = initial_capital
    positions = []  # Track all positions
    trades = []  # Track whether trades were executed
    equity_curve = [capital]
    total_transaction_costs = 0.0  # Track cumulative costs
    
    # Track active positions for stop-loss
    active_positions = []  # List of dicts: {'symbol': 'AAPL', 'entry_price': 150, 'quantity': 10, 'is_long': True}
    
    # Risk management state
    trading_paused = False
    drawdown_reached = False
    
    # Generate signals
    signals = generate_trading_signals(predictions, threshold)
    
    # Calculate profits/losses
    prev_diff = None
    
    for i in range(len(predictions)):
        # Check drawdown limit before each trade
        if not check_risk(capital, initial_capital, max_drawdown):
            if not trading_paused:
                print(f"\n⚠️  MAX DRAWDOWN REACHED at day {i}!")
                print(f"   Current capital: ${capital:,.2f}")
                print(f"   Drawdown: {((initial_capital - capital) / initial_capital * 100):.2f}%")
                print(f"   Trading paused for risk management.")
            trading_paused = True
            drawdown_reached = True
            # Close all positions
            active_positions = []
        
        # Check stop-loss on active positions
        for pos in active_positions[:]:  # Copy list to iterate safely
            symbol = pos['symbol']
            entry_price = pos['entry_price']
            is_long = pos['is_long']
            current_price = price_data[symbol].iloc[i]
            
            if apply_stop_loss(current_price, entry_price, stop_loss_pct, is_long):
                # Stop-loss triggered - close position
                if is_long:
                    loss = (entry_price - current_price) * pos['quantity']
                else:
                    loss = (current_price - entry_price) * pos['quantity']

                capital -= abs(loss)  # FIXED: Subtract loss from capital
                active_positions.remove(pos)
                risk_manager.remove_position(symbol)

                print(f"  Stop-loss triggered for {symbol} at ${current_price:.2f} (entry: ${entry_price:.2f})")
        
        signal = signals[i]
        aapl_price = price_data[TICKER_LONG].iloc[i]
        msft_price = price_data[TICKER_SHORT].iloc[i]
        actual_diff = actual_differences[i]
        
        trade_result = {
            'date': price_data.index[i],
            'signal': signal,
            'aapl_price': aapl_price,
            'msft_price': msft_price,
            'predicted_diff': predictions[i],
            'actual_diff': actual_diff,
            'profit': 0,
            'return_pct': 0,
            'trading_paused': trading_paused,
            'stop_loss_triggered': False
        }
        
        if signal != 'Hold' and prev_diff is not None and not trading_paused:
            # Execute trade (only if not paused)
            position_size = capital * 0.1  # Use 10% of capital per trade

            if signal == SIGNAL_BUY_LONG:
                # Betting difference will increase
                # Enter: long TICKER_LONG, short TICKER_SHORT
                diff_change = actual_diff - prev_diff

                # Calculate quantity for each leg
                aapl_qty = int(position_size / aapl_price)
                msft_qty = int(position_size / msft_price)

                # Get current timestamp (use index date + 10 AM)
                trade_time = datetime.combine(price_data.index[i].date(), time(10, 0))

                trade_executed = False
                trade_costs = 0.0
                profit = 0.0

                try:
                    # Pre-trade risk checks
                    current_prices = {TICKER_LONG: aapl_price, TICKER_SHORT: msft_price}
                    risk_manager.pre_trade_check(TICKER_LONG, aapl_qty, aapl_price, prices=current_prices)
                    risk_manager.pre_trade_check(TICKER_SHORT, -msft_qty, msft_price, prices=current_prices)

                    # Execute buy TICKER_LONG order
                    aapl_result = order_executor.execute_order(
                        symbol=TICKER_LONG,
                        quantity=aapl_qty,
                        side=OrderSide.BUY,
                        price=aapl_price,
                        available_capital=capital,
                        timestamp=trade_time,
                        average_daily_volume=1000000,
                        spread=0.01,
                    )

                    # Execute short TICKER_SHORT order
                    msft_result = order_executor.execute_order(
                        symbol=TICKER_SHORT,
                        quantity=msft_qty,
                        side=OrderSide.SHORT,
                        price=msft_price,
                        available_capital=capital,
                        timestamp=trade_time,
                        average_daily_volume=1000000,
                        spread=0.01,
                    )

                    # Both orders executed successfully
                    if aapl_result['status'] == OrderStatus.FILLED and msft_result['status'] == OrderStatus.FILLED:
                        trade_executed = True

                        # Get actual filled quantities and prices
                        aapl_filled_qty = aapl_result['filled_quantity']
                        msft_filled_qty = msft_result['filled_quantity']
                        aapl_avg_price = aapl_result['average_price']
                        msft_avg_price = msft_result['average_price']

                        # Calculate costs from execution results
                        trade_costs = aapl_result['total_cost'] + msft_result['total_cost']
                        total_transaction_costs += trade_costs

                        # Calculate profit with actual filled quantities
                        gross_profit = position_size * (diff_change / (aapl_avg_price + msft_avg_price))
                        profit = gross_profit - trade_costs

                        # Track positions
                        if aapl_filled_qty > 0:
                            active_positions.append({
                                'symbol': TICKER_LONG,
                                'entry_price': aapl_avg_price,
                                'quantity': aapl_filled_qty,
                                'is_long': True
                            })
                        if msft_filled_qty > 0:
                            active_positions.append({
                                'symbol': TICKER_SHORT,
                                'entry_price': msft_avg_price,
                                'quantity': msft_filled_qty,
                                'is_long': False
                            })

                        # Sync risk manager position state
                        risk_manager.add_position(TICKER_LONG, aapl_filled_qty, aapl_avg_price)
                        risk_manager.add_position(TICKER_SHORT, -msft_filled_qty, msft_avg_price)
                    elif aapl_result['status'] == 'PARTIAL' or msft_result['status'] == 'PARTIAL':
                        # Handle partial fills
                        trade_executed = True
                        trade_costs = aapl_result.get('total_cost', 0) + msft_result.get('total_cost', 0)
                        total_transaction_costs += trade_costs

                        # Reduced profit due to partial fill
                        fill_rate = (aapl_result['filled_quantity'] + msft_result['filled_quantity']) / (aapl_qty + msft_qty)
                        gross_profit = position_size * fill_rate * (diff_change / (aapl_price + msft_price))
                        profit = gross_profit - trade_costs

                        print(f"  Partial fill: {TICKER_LONG} {aapl_result['filled_quantity']}/{aapl_qty}, "
                              f"{TICKER_SHORT} {msft_result['filled_quantity']}/{msft_qty}")

                except RiskException as e:
                    print(f"  Trade blocked by risk check: {e}")
                    trade_executed = False
                    profit = 0.0
                except (OrderRejectedError, InsufficientFundsError, ShortNotAvailableError) as e:
                    # Order rejected - no trade executed
                    print(f"  Order rejected for {signal}: {e}")
                    trade_executed = False
                    profit = 0.0
                except Exception as e:
                    # Unexpected error - log and skip trade
                    print(f"  Trade execution error for {signal}: {e}")
                    trade_executed = False
                    profit = 0.0

                if trade_executed:
                    capital += profit
                    trade_result['profit'] = profit
                    trade_result['return_pct'] = (diff_change / (aapl_price + msft_price)) * 100
                    trades.append(True)
                else:
                    trades.append(False)
                
            elif signal == SIGNAL_BUY_SHORT:
                # Betting difference will decrease
                # Enter: long TICKER_SHORT, short TICKER_LONG
                diff_change = prev_diff - actual_diff

                # Calculate quantity for each leg
                aapl_qty = int(position_size / aapl_price)
                msft_qty = int(position_size / msft_price)

                # Get current timestamp (use index date + 10 AM)
                trade_time = datetime.combine(price_data.index[i].date(), time(10, 0))

                trade_executed = False
                trade_costs = 0.0
                profit = 0.0

                try:
                    # Pre-trade risk checks
                    current_prices = {TICKER_LONG: aapl_price, TICKER_SHORT: msft_price}
                    risk_manager.pre_trade_check(TICKER_SHORT, msft_qty, msft_price, prices=current_prices)
                    risk_manager.pre_trade_check(TICKER_LONG, -aapl_qty, aapl_price, prices=current_prices)

                    # Execute buy TICKER_SHORT order
                    msft_result = order_executor.execute_order(
                        symbol=TICKER_SHORT,
                        quantity=msft_qty,
                        side=OrderSide.BUY,
                        price=msft_price,
                        available_capital=capital,
                        timestamp=trade_time,
                        average_daily_volume=1000000,
                        spread=0.01,
                    )

                    # Execute short TICKER_LONG order
                    aapl_result = order_executor.execute_order(
                        symbol=TICKER_LONG,
                        quantity=aapl_qty,
                        side=OrderSide.SHORT,
                        price=aapl_price,
                        available_capital=capital,
                        timestamp=trade_time,
                        average_daily_volume=1000000,
                        spread=0.01,
                    )

                    # Both orders executed successfully
                    if msft_result['status'] == OrderStatus.FILLED and aapl_result['status'] == OrderStatus.FILLED:
                        trade_executed = True

                        # Get actual filled quantities and prices
                        msft_filled_qty = msft_result['filled_quantity']
                        aapl_filled_qty = aapl_result['filled_quantity']
                        msft_avg_price = msft_result['average_price']
                        aapl_avg_price = aapl_result['average_price']

                        # Calculate costs from execution results
                        trade_costs = msft_result['total_cost'] + aapl_result['total_cost']
                        total_transaction_costs += trade_costs

                        # Calculate profit with actual filled quantities
                        gross_profit = position_size * (diff_change / (msft_avg_price + aapl_avg_price))
                        profit = gross_profit - trade_costs

                        # Track positions
                        if msft_filled_qty > 0:
                            active_positions.append({
                                'symbol': TICKER_SHORT,
                                'entry_price': msft_avg_price,
                                'quantity': msft_filled_qty,
                                'is_long': True
                            })
                        if aapl_filled_qty > 0:
                            active_positions.append({
                                'symbol': TICKER_LONG,
                                'entry_price': aapl_avg_price,
                                'quantity': aapl_filled_qty,
                                'is_long': False
                            })

                        # Sync risk manager position state
                        risk_manager.add_position(TICKER_SHORT, msft_filled_qty, msft_avg_price)
                        risk_manager.add_position(TICKER_LONG, -aapl_filled_qty, aapl_avg_price)
                    elif msft_result['status'] == 'PARTIAL' or aapl_result['status'] == 'PARTIAL':
                        # Handle partial fills
                        trade_executed = True
                        trade_costs = msft_result.get('total_cost', 0) + aapl_result.get('total_cost', 0)
                        total_transaction_costs += trade_costs

                        # Reduced profit due to partial fill
                        fill_rate = (msft_result['filled_quantity'] + aapl_result['filled_quantity']) / (msft_qty + aapl_qty)
                        gross_profit = position_size * fill_rate * (diff_change / (msft_price + aapl_price))
                        profit = gross_profit - trade_costs

                        print(f"  Partial fill: {TICKER_SHORT} {msft_result['filled_quantity']}/{msft_qty}, "
                              f"{TICKER_LONG} {aapl_result['filled_quantity']}/{aapl_qty}")

                except RiskException as e:
                    print(f"  Trade blocked by risk check: {e}")
                    trade_executed = False
                    profit = 0.0
                except (OrderRejectedError, InsufficientFundsError, ShortNotAvailableError) as e:
                    # Order rejected - no trade executed
                    print(f"  Order rejected for {signal}: {e}")
                    trade_executed = False
                    profit = 0.0
                except Exception as e:
                    # Unexpected error - log and skip trade
                    print(f"  Trade execution error for {signal}: {e}")
                    trade_executed = False
                    profit = 0.0

                if trade_executed:
                    capital += profit
                    trade_result['profit'] = profit
                    trade_result['return_pct'] = (diff_change / (msft_price + aapl_price)) * 100
                    trades.append(True)
                else:
                    trades.append(False)
        else:
            if trading_paused:
                trades.append(False)  # No trade due to pause
            else:
                trades.append(False)  # No trade (Hold or first day)
            trade_result['profit'] = 0
            trade_result['return_pct'] = 0
        
        positions.append(trade_result)
        equity_curve.append(capital)
        risk_manager.update_capital(capital)
        prev_diff = actual_diff  # Update for next iteration
    
    # Calculate metrics
    total_return = ((capital - initial_capital) / initial_capital) * 100
    num_trades = sum(trades)
    winning_trades = sum(1 for p in positions if p['profit'] > 0)
    losing_trades = sum(1 for p in positions if p['profit'] < 0)
    stop_loss_trades = sum(1 for p in positions if p.get('stop_loss_triggered', False))
    win_rate = (winning_trades / num_trades * 100) if num_trades > 0 else 0
    
    total_profit = sum(p['profit'] for p in positions)
    avg_profit = total_profit / num_trades if num_trades > 0 else 0
    
    # Calculate max drawdown
    equity_array = np.array(equity_curve)
    running_max = np.maximum.accumulate(equity_array)
    drawdown = (equity_array - running_max) / running_max * 100
    max_drawdown_actual = np.min(drawdown)
    
    # Calculate net profit (after all costs)
    net_profit = total_profit  # Already includes transaction costs in our calculation

    results = {
        'initial_capital': initial_capital,
        'final_capital': capital,
        'total_return_pct': total_return,
        'total_profit': total_profit,
        'total_transaction_costs': total_transaction_costs,
        'net_profit': net_profit,
        'num_trades': num_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'stop_loss_trades': stop_loss_trades,
        'win_rate': win_rate,
        'avg_profit_per_trade': avg_profit,
        'avg_cost_per_trade': total_transaction_costs / num_trades if num_trades > 0 else 0,
        'max_drawdown_pct': max_drawdown_actual,
        'drawdown_limit_reached': drawdown_reached,
        'positions': positions,
        'equity_curve': equity_curve,
        'signals': signals
    }

    return results

def print_backtest_results(results):
    """Print backtest results in a formatted way."""
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    print(f"Initial Capital:      ${results['initial_capital']:,.2f}")
    print(f"Final Capital:        ${results['final_capital']:,.2f}")
    print(f"Total Return:         {results['total_return_pct']:.2f}%")
    print(f"Total Profit/Loss:    ${results['total_profit']:,.2f}")
    print(f"\nTransaction Costs:")
    print(f"  Total Costs:        ${results.get('total_transaction_costs', 0):,.2f}")
    print(f"  Avg Cost/Trade:     ${results.get('avg_cost_per_trade', 0):,.2f}")
    if results.get('total_profit', 0) != 0:
        cost_pct = (results.get('total_transaction_costs', 0) / abs(results['total_profit'])) * 100
        print(f"  Costs as % of P&L:  {cost_pct:.2f}%")
    print(f"\nTrading Statistics:")
    print(f"  Total Trades:       {results['num_trades']}")
    print(f"  Winning Trades:     {results['winning_trades']}")
    print(f"  Losing Trades:      {results['losing_trades']}")
    if 'stop_loss_trades' in results:
        print(f"  Stop-Loss Triggered: {results['stop_loss_trades']}")
    print(f"  Win Rate:           {results['win_rate']:.2f}%")
    print(f"  Avg Profit/Trade:   ${results['avg_profit_per_trade']:,.2f}")
    if 'max_drawdown_pct' in results:
        print(f"  Max Drawdown:       {results['max_drawdown_pct']:.2f}%")
    if 'drawdown_limit_reached' in results and results['drawdown_limit_reached']:
        print(f"\n  ⚠️  Drawdown Limit Reached: Trading was paused")
    print("=" * 60)

def save_backtest_results(results, filename='backtest_results.csv'):
    """Save backtest results to CSV."""
    df = pd.DataFrame(results['positions'])
    df.to_csv(filename, index=False)
    print(f"\nBacktest results saved to {filename}")

if __name__ == "__main__":
    try:
        # Configuration
        MODEL_PATH = 'lstm_price_difference_model.h5'
        DATA_FILE = 'processed_stock_data.csv'
        TIMESTEPS = 60
        TEST_SIZE = 0.2
        THRESHOLD = 0.5  # Threshold for trading signals (in normalized scale)
        INITIAL_CAPITAL = 10000
        MAX_DRAWDOWN = 0.05  # 5% maximum drawdown
        STOP_LOSS_PCT = 0.02  # 2% stop-loss per trade
        
        # Load model and prepare test data
        model, X_test, y_test_actual, test_dates, price_data = load_model_and_data(
            MODEL_PATH, DATA_FILE, TIMESTEPS, TEST_SIZE
        )
        
        if price_data is None:
            print("Error: Could not load original price data for backtesting.")
            print("Please ensure processed_stock_data.csv contains original price columns.")
            exit(1)
        
        # Step 1: Use the trained model to predict next day's price difference
        predictions_normalized = predict_next_day(model, X_test)
        
        # Step 2: Generate trading signals
        print(f"\nGenerating trading signals with threshold={THRESHOLD}...")
        signals = generate_trading_signals(predictions_normalized, THRESHOLD)
        
        # Print some example signals
        print("\nFirst 10 trading signals:")
        for i in range(min(10, len(signals))):
            print(f"  {test_dates[i].strftime('%Y-%m-%d')}: {signals[i]} "
                  f"(Predicted diff: {predictions_normalized[i]:.4f})")
        
        # Count signals
        signal_counts = pd.Series(signals).value_counts()
        print(f"\nSignal distribution:")
        for signal, count in signal_counts.items():
            print(f"  {signal}: {count} ({count/len(signals)*100:.1f}%)")
        
        # Step 3: Backtest the strategy with risk management
        actual_diffs = price_data['Price_Difference'].values
        
        results = backtest_strategy(
            predictions_normalized, 
            actual_diffs,
            price_data,
            threshold=THRESHOLD,
            initial_capital=INITIAL_CAPITAL,
            max_drawdown=MAX_DRAWDOWN,
            stop_loss_pct=STOP_LOSS_PCT
        )
        
        # Print results
        print_backtest_results(results)
        
        # Save results
        save_backtest_results(results)
        
        print("\nBacktesting completed successfully!")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease ensure:")
        print("1. You have run 'python process_stock_data.py' to generate processed data")
        print("2. You have run 'python train_model.py' to train and save the model")
    except ImportError as e:
        print(f"TensorFlow/Keras not available: {e}")
        print("\nTo use this script:")
        print("1. Install TensorFlow: pip install tensorflow")
        print("2. Note: TensorFlow requires Python 3.11 or 3.12 (not 3.14)")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
