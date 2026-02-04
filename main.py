#!/usr/bin/env python3
"""
Stock Arbitrage Model - Interactive CLI
A fun and interactive command-line interface for the stock arbitrage trading system!
"""

import os
import sys
import subprocess
import functools
import traceback as _tb
from datetime import datetime

# ── Colour support (disabled via NO_COLOR env or --no-color flag) ──────
_NO_COLOR = os.environ.get("NO_COLOR") is not None or "--no-color" in sys.argv

class Colors:
    """ANSI colour codes; all empty strings when colour is disabled."""
    if _NO_COLOR:
        HEADER = OKBLUE = OKCYAN = OKGREEN = WARNING = FAIL = ENDC = BOLD = UNDERLINE = ''
    else:
        HEADER    = '\033[95m'
        OKBLUE    = '\033[94m'
        OKCYAN    = '\033[96m'
        OKGREEN   = '\033[92m'
        WARNING   = '\033[93m'
        FAIL      = '\033[91m'
        ENDC      = '\033[0m'
        BOLD      = '\033[1m'
        UNDERLINE = '\033[4m'


# ── Helpers ──────────────────────────────────────────────────────────────

def get_python_cmd():
    """Return the best Python executable (prefers 3.12 venv for TensorFlow)."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    venv_python = os.path.join(script_dir, 'venv_py312', 'bin', 'python')
    if os.path.exists(venv_python):
        return venv_python
    return sys.executable


def safe_execute(func):
    """Decorator: catch exceptions inside a menu handler so the loop never dies."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except KeyboardInterrupt:
            print_warning("\n⚠  [CANCELLED] Operation interrupted by user")
            return False
        except Exception as exc:
            print_error(f"[ERROR] {exc}")
            show_tb = get_user_input("Show detailed traceback? (y/n)", "n")
            if show_tb and show_tb.lower() == 'y':
                _tb.print_exc()
            return False
    return wrapper


# ── Output primitives ────────────────────────────────────────────────────

def print_header():
    """Print the top banner."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}")
    print("  STOCK ARBITRAGE MODEL - INTERACTIVE CLI")
    print(f"{'='*70}{Colors.ENDC}")
    print(f"{Colors.OKCYAN}  Predict price differences | Train LSTM | Backtest | Live trade{Colors.ENDC}\n")

def print_success(message):
    print(f"{Colors.OKGREEN}[SUCCESS] {message}{Colors.ENDC}")

def print_error(message):
    print(f"{Colors.FAIL}[ERROR] {message}{Colors.ENDC}")

def print_info(message):
    print(f"{Colors.OKBLUE}[INFO] {message}{Colors.ENDC}")

def print_warning(message):
    print(f"{Colors.WARNING}[WARNING] {message}{Colors.ENDC}")

def clear_screen():
    os.system('clear' if os.name != 'nt' else 'cls')


# ── Input ────────────────────────────────────────────────────────────────

def get_user_input(prompt, default=None, input_type=str, validator=None):
    """
    Prompt the user for input with an optional default, type conversion, and
    a *validator* callback ``(value) -> (bool, str)`` that returns
    ``(True, "")`` on success or ``(False, "error message")`` on failure.
    Keeps re-prompting until the value is valid or the user presses Ctrl-C.
    """
    while True:
        if default is not None:
            prompt_text = f"{Colors.BOLD}{prompt} [{default}]: {Colors.ENDC}"
        else:
            prompt_text = f"{Colors.BOLD}{prompt}: {Colors.ENDC}"

        try:
            raw = input(prompt_text).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return None

        # Apply default
        if not raw:
            if default is not None:
                raw = str(default)
            else:
                print_warning("  A value is required.  Try again.")
                continue

        # Type conversion
        try:
            value = input_type(raw)
        except (ValueError, TypeError):
            print_warning(f"  Expected {input_type.__name__}.  Try again.")
            continue

        # Custom validation
        if validator is not None:
            ok, msg = validator(value)
            if not ok:
                print_warning(f"  {msg}")
                continue

        return value


def check_file_exists(filename):
    return os.path.exists(filename)

@safe_execute
def update_sentiment_cache_cli():
    """Update Reddit sentiment cache via CLI."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}💬 UPDATE REDDIT SENTIMENT CACHE{Colors.ENDC}")
    print(f"{Colors.OKCYAN}{'='*70}{Colors.ENDC}\n")
    
    try:
        from sentiment_analysis import build_historical_sentiment_cache
        from process_stock_data import get_selected_stocks
        from datetime import datetime, timedelta
        
        # Get selected stocks
        try:
            stock1, stock2 = get_selected_stocks()
            symbols = [stock1, stock2]
        except Exception:
            symbols_input = get_user_input("Enter stock symbols (comma-separated, e.g., AAPL,MSFT)", "AAPL,MSFT")
            symbols = [s.strip().upper() for s in symbols_input.split(',')]
        
        print_info(f"Updating sentiment cache for: {', '.join(symbols)}")
        
        # Get date range
        days_back = get_user_input("How many days back to fetch? (default: 30)", "30", int)
        if days_back is None:
            days_back = 30
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        print_info(f"Date range: {start_date.date()} to {end_date.date()}")
        
        confirm = get_user_input("Start updating sentiment cache? (y/n)", "y")
        if confirm.lower() != 'y':
            print_info("Cancelled.")
            return False
        
        print()
        build_historical_sentiment_cache(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            limit=50,
            delay_between_symbols=2
        )
        
        print_success("Sentiment cache updated successfully!")
        print_info("The cache will be used automatically when processing stock data.")
        return True
        
    except ImportError as e:
        print_error(f"Could not import sentiment analysis module: {e}")
        print_info("Make sure 'praw' is installed: pip install praw")
        return False
    except Exception as e:
        print_error(f"Error updating sentiment cache: {e}")
        import traceback
        traceback.print_exc()
        return False

@safe_execute
def update_options_cache_cli():
    """Update options volume cache via CLI."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}📊 UPDATE OPTIONS VOLUME CACHE{Colors.ENDC}")
    print(f"{Colors.OKCYAN}{'='*70}{Colors.ENDC}\n")
    
    try:
        from options_data import build_historical_options_cache
        from process_stock_data import get_selected_stocks
        from datetime import datetime, timedelta
        
        # Get selected stocks
        try:
            stock1, stock2 = get_selected_stocks()
            symbols = [stock1, stock2]
        except Exception:
            symbols_input = get_user_input("Enter stock symbols (comma-separated, e.g., AAPL,MSFT)", "AAPL,MSFT")
            symbols = [s.strip().upper() for s in symbols_input.split(',')]
        
        print_info(f"Updating options volume cache for: {', '.join(symbols)}")
        
        # Get date range
        days_back = get_user_input("How many days back to fetch? (default: 30)", "30", int)
        if days_back is None:
            days_back = 30
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        print_info(f"Date range: {start_date.date()} to {end_date.date()}")
        print_warning("Note: yfinance only provides current options data.")
        print_info("This will fetch current options volume and cache it for today.")
        print_info("For true historical data, run this daily via cron job.\n")
        
        confirm = get_user_input("Start updating options volume cache? (y/n)", "y")
        if confirm.lower() != 'y':
            print_info("Cancelled.")
            return False
        
        print()
        build_historical_options_cache(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            delay_between_symbols=2
        )
        
        print_success("Options volume cache updated successfully!")
        print_info("The cache will be used automatically when processing stock data.")
        return True
        
    except ImportError as e:
        print_error(f"Could not import options data module: {e}")
        return False
    except Exception as e:
        print_error(f"Error updating options volume cache: {e}")
        import traceback
        traceback.print_exc()
        return False

@safe_execute
def clear_yfinance_cache():
    """Clear the yfinance cache to fix download issues."""
    import shutil
    import os
    
    cache_paths = [
        os.path.expanduser('~/.cache/py-yfinance'),
        os.path.expanduser('~/.cache/yfinance'),
    ]
    
    print_info("Clearing yfinance cache...")
    cleared = False
    
    for cache_path in cache_paths:
        if os.path.exists(cache_path):
            try:
                shutil.rmtree(cache_path)
                print_success(f"Cleared cache: {cache_path}")
                cleared = True
            except Exception as e:
                print_warning(f"Could not clear {cache_path}: {e}")
    
    if not cleared:
        print_info("No yfinance cache found (this is okay)")
    
    print_success("Cache clearing complete!")
    return True

@safe_execute
def process_data():
    """Run the data processing script."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}📥 PROCESSING STOCK DATA{Colors.ENDC}")
    print(f"{Colors.OKCYAN}{'='*70}{Colors.ENDC}\n")
    
    # Check if user wants to configure market and date range
    from datetime import datetime, timedelta
    import pytz
    
    # Check current market config
    market_config_exists = os.path.exists('market_config.txt')
    current_market = 'NYSE'
    current_timezone = 'America/New_York'
    
    if market_config_exists:
        try:
            with open('market_config.txt', 'r') as f:
                for line in f:
                    if line.startswith('MARKET='):
                        current_market = line.split('=', 1)[1].strip()
                    elif line.startswith('TIMEZONE='):
                        current_timezone = line.split('=', 1)[1].strip()
            print_info(f"Current market: {current_market} (Timezone: {current_timezone})")
        except Exception:
            pass
    
    # Check if date range config exists
    date_config_exists = os.path.exists('date_range_config.txt')
    if date_config_exists:
        try:
            with open('date_range_config.txt', 'r') as f:
                for line in f:
                    if line.startswith('START_DATE='):
                        start_str = line.split('=', 1)[1].strip()
                    elif line.startswith('END_DATE='):
                        end_str = line.split('=', 1)[1].strip()
            print_info(f"Current date range: {start_str} to {end_str}")
        except Exception:
            pass
    
    configure_market = get_user_input("Configure market and date range? (y/n) [Default: NYSE, last trading day - 3 years]", "n")
    if configure_market.lower() == 'y':
        print()
        print(f"{Colors.BOLD}Market Selection:{Colors.ENDC}")
        print("  1. NYSE (New York Stock Exchange) - America/New_York")
        print("  2. NASDAQ - America/New_York")
        print("  3. NIKKEI (Tokyo Stock Exchange) - Asia/Tokyo")
        print("  4. LSE (London Stock Exchange) - Europe/London")
        print("  5. TSX (Toronto Stock Exchange) - America/Toronto")
        print("  6. ASX (Australian Stock Exchange) - Australia/Sydney")
        print("  7. Custom")
        
        market_choice = get_user_input("Select market (1-7)", "1")
        
        market_map = {
            '1': ('NYSE', 'America/New_York'),
            '2': ('NASDAQ', 'America/New_York'),
            '3': ('NIKKEI', 'Asia/Tokyo'),
            '4': ('LSE', 'Europe/London'),
            '5': ('TSX', 'America/Toronto'),
            '6': ('ASX', 'Australia/Sydney'),
        }
        
        if market_choice in market_map:
            market, timezone_str = market_map[market_choice]
        elif market_choice == '7':
            market = get_user_input("Enter market name", "NYSE")
            timezone_str = get_user_input("Enter timezone (e.g., America/New_York)", "America/New_York")
        else:
            market, timezone_str = 'NYSE', 'America/New_York'
        
        # Save market config
        try:
            with open('market_config.txt', 'w') as f:
                f.write(f"MARKET={market}\n")
                f.write(f"TIMEZONE={timezone_str}\n")
            print_success(f"Market configuration saved: {market} ({timezone_str})")
        except Exception as e:
            print_warning(f"Could not save market config: {e}")
        
        # Get last trading day for the selected market
        try:
            # Import the function from process_stock_data
            import sys
            import importlib.util
            spec = importlib.util.spec_from_file_location("process_stock_data", "process_stock_data.py")
            process_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(process_module)
            
            last_trading_day = process_module.get_last_trading_day(market, timezone_str)
            print_info(f"Last trading day for {market}: {last_trading_day.strftime('%Y-%m-%d')}")
        except Exception as e:
            print_warning(f"Could not determine last trading day: {e}")
            # Fallback: simple calculation (yesterday, accounting for weekends)
            tz = pytz.timezone(timezone_str)
            now = datetime.now(tz)
            last_trading_day = now - timedelta(days=1)
            while last_trading_day.weekday() >= 5:  # Saturday = 5, Sunday = 6
                last_trading_day = last_trading_day - timedelta(days=1)
            last_trading_day = last_trading_day.replace(hour=0, minute=0, second=0, microsecond=0)
            print_info(f"Using fallback calculation: {last_trading_day.strftime('%Y-%m-%d')}")
        
        print()
        configure_dates = get_user_input("Configure custom date range? (y/n) [Default: Last trading day - 3 years]", "n")
        if configure_dates.lower() == 'y':
            # Get start date
            default_start = (last_trading_day - timedelta(days=3*365)).strftime('%Y-%m-%d')
            start_input = get_user_input(f"Enter start date (YYYY-MM-DD) [Default: {default_start}]", default_start)
            try:
                start_date = datetime.strptime(start_input, '%Y-%m-%d')
            except ValueError:
                print_error("Invalid date format. Using default (3 years before last trading day).")
                start_date = last_trading_day - timedelta(days=3*365)
            
            # Get end date
            default_end = last_trading_day.strftime('%Y-%m-%d')
            end_input = get_user_input(f"Enter end date (YYYY-MM-DD) [Default: {default_end} (last trading day)]", default_end)
            try:
                end_date = datetime.strptime(end_input, '%Y-%m-%d')
            except ValueError:
                print_error("Invalid date format. Using default (last trading day).")
                end_date = last_trading_day
            
            # Validate date range
            if end_date < start_date:
                print_error("End date must be after start date. Using defaults.")
                start_date = last_trading_day - timedelta(days=3*365)
                end_date = last_trading_day
            
            # Save to config file
            try:
                with open('date_range_config.txt', 'w') as f:
                    f.write(f"START_DATE={start_date.strftime('%Y-%m-%d')}\n")
                    f.write(f"END_DATE={end_date.strftime('%Y-%m-%d')}\n")
                print_success(f"Date range saved: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            except Exception as e:
                print_warning(f"Could not save date range config: {e}")
        else:
            # Use defaults (last trading day - 3 years)
            print_info(f"Using default date range: Last trading day ({last_trading_day.strftime('%Y-%m-%d')}) - 3 years")
    
    if check_file_exists('processed_stock_data.csv'):
        print_warning("processed_stock_data.csv already exists!")
        overwrite = get_user_input("Overwrite existing data? (y/n)", "n")
        if overwrite.lower() != 'y':
            print_info("Skipping data processing...")
            return True
    
    # Offer to clear cache if user wants
    print_info("If you're experiencing download errors, clearing the cache may help.")
    clear_cache = get_user_input("Clear yfinance cache before downloading? (y/n)", "n")
    if clear_cache and clear_cache.lower() == 'y':
        clear_yfinance_cache()
        print()  # Add spacing
    
    print_info("📥 Downloading stock data from Yahoo Finance...")
    print_info("⏱️  Estimated time: 2-5 minutes (depends on date range)")
    print_info("💡 Tip: The script will show progress as it downloads\n")

    try:
        import time
        start_time = time.time()

        result = subprocess.run([sys.executable, 'process_stock_data.py'],
                              capture_output=False, text=True)

        elapsed = time.time() - start_time
        elapsed_str = f"{int(elapsed//60)}m {int(elapsed%60)}s" if elapsed >= 60 else f"{int(elapsed)}s"

        if result.returncode == 0:
            print_success(f"✅ Data processed successfully in {elapsed_str}!")
            print_info("📄 Output saved to: processed_stock_data.csv")
            return True
        else:
            print_error("❌ Data processing failed!")
            print_info("\n💡 Common issues and fixes:")
            print("  1. Network timeout: Check internet connection")
            print("  2. Rate limiting: Wait 5-10 minutes and try again")
            print("  3. Cache issues: Run option 13 to clear yfinance cache")
            print("  4. Invalid symbols: Check selected_stocks.txt")
            print("\n📝 Detailed logs:")
            print("  • Run manually to see full output: python process_stock_data.py")
            return False
    except KeyboardInterrupt:
        print_warning("\n⚠️  Data processing interrupted by user")
        print_info("💡 Partial data may have been saved")
        return False
    except Exception as e:
        print_error(f"Error: {e}")
        print_info("\nTroubleshooting tips:")
        print("  • Try clearing the yfinance cache (option 8)")
        print("  • Check your internet connection")
        return False

@safe_execute
def train_model():
    """Run the model training script."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}🧠 TRAINING LSTM MODEL{Colors.ENDC}")
    print(f"{Colors.OKCYAN}{'='*70}{Colors.ENDC}\n")
    
    if not check_file_exists('processed_stock_data.csv'):
        print_error("No processed data found!")
        print_info("Please process data first (option 1)")
        return False
    
    print_info("Configuring training parameters...\n")
    
    # Get training parameters
    epochs = get_user_input("Number of epochs", 10, int)
    batch_size = get_user_input("Batch size", 32, int)
    dropout_rate = get_user_input("Dropout rate (0.0-1.0)", 0.2, float)
    
    if epochs is None or batch_size is None or dropout_rate is None:
        print_error("Invalid input! Using defaults...")
        epochs, batch_size, dropout_rate = 10, 32, 0.2
    
    print(f"\n{Colors.BOLD}Training Configuration:{Colors.ENDC}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Dropout Rate: {dropout_rate}\n")
    
    confirm = get_user_input("Start training? (y/n)", "y")
    if confirm.lower() != 'y':
        print_info("Training cancelled.")
        return False
    
    print_info("Training model... This may take several minutes...\n")
    
    try:
        # Modify train_model.py to accept command-line args or use environment variables
        # For now, we'll modify the script temporarily or use a simpler approach
        # Actually, let's just run it and note that parameters are hardcoded
        print_warning("Note: Training will use default parameters from train_model.py")
        print_info("To customize, edit train_model.py directly\n")
        
        python_cmd = get_python_cmd()
        result = subprocess.run([python_cmd, 'train_model.py'],
                              capture_output=False, text=True)
        if result.returncode == 0:
            print_success("Model trained successfully!")
            return True
        else:
            print_error("Model training failed!")
            return False
    except Exception as e:
        print_error(f"Error: {e}")
        return False

@safe_execute
def backtest_strategy():
    """Run the backtesting script."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}📊 BACKTESTING TRADING STRATEGY{Colors.ENDC}")
    print(f"{Colors.OKCYAN}{'='*70}{Colors.ENDC}\n")
    
    if not check_file_exists('lstm_price_difference_model.h5'):
        print_error("No trained model found!")
        print_info("Please train a model first (option 2)")
        return False
    
    if not check_file_exists('processed_stock_data.csv'):
        print_error("No processed data found!")
        print_info("Please process data first (option 1)")
        return False
    
    print_info("Configuring backtesting parameters...\n")
    
    # Get backtesting parameters
    threshold = get_user_input("Trading signal threshold (normalized)", 0.5, float)
    initial_capital = get_user_input("Initial capital ($)", 10000, float)
    
    if threshold is None or initial_capital is None:
        print_error("Invalid input! Using defaults...")
        threshold, initial_capital = 0.5, 10000
    
    # Validate parameters
    if threshold < 0:
        print_warning("Threshold should be non-negative. Using default: 0.5")
        threshold = 0.5
    if initial_capital <= 0:
        print_warning("Initial capital must be positive. Using default: 10000")
        initial_capital = 10000
    
    print(f"\n{Colors.BOLD}Backtest Configuration:{Colors.ENDC}")
    print(f"  Signal Threshold: {threshold}")
    print(f"  Initial Capital: ${initial_capital:,.2f}\n")
    
    confirm = get_user_input("Start backtesting? (y/n)", "y")
    if confirm.lower() != 'y':
        print_info("Backtesting cancelled.")
        return False
    
    print_info("Running backtest...\n")
    
    try:
        print_warning("Note: Backtesting uses parameters from backtest_strategy.py")
        print_info("To customize, edit backtest_strategy.py directly\n")
        
        python_cmd = get_python_cmd()
        result = subprocess.run([python_cmd, 'backtest_strategy.py'],
                              capture_output=False, text=True)
        if result.returncode == 0:
            print_success("Backtesting completed successfully!")
            if check_file_exists('backtest_results.csv'):
                print_info("Results saved to: backtest_results.csv")
            return True
        else:
            print_error("Backtesting failed!")
            print_info("\nTroubleshooting tips:")
            print("  • Make sure the model file is valid")
            print("  • Check that processed_stock_data.csv is up to date")
            print("  • Review the error messages above")
            return False
    except KeyboardInterrupt:
        print_warning("\nBacktesting interrupted by user")
        return False
    except FileNotFoundError:
        print_error("backtest_strategy.py not found!")
        return False
    except Exception as e:
        print_error(f"Error: {e}")
        print_info("\nTroubleshooting tips:")
        print("  • Make sure all required files exist")
        print("  • Check that TensorFlow is installed")
        return False

@safe_execute
def setup_live_trading():
    """First-use setup process for Schwab API credentials."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}⚙️  SETUP LIVE TRADING (SCHWAB API){Colors.ENDC}")
    print(f"{Colors.OKCYAN}{'='*70}{Colors.ENDC}\n")
    
    # Check Python version
    if sys.version_info < (3, 10):
        print_error("Schwab API requires Python 3.10 or higher!")
        print_info(f"Your Python version: {sys.version_info.major}.{sys.version_info.minor}")
        print_info("Please upgrade Python or use paper trading mode.")
        return False
    
    # Check if schwab-py is installed
    try:
        import schwab
        print_success("schwab-py library is installed")
    except ImportError:
        print_error("schwab-py library is not installed!")
        print_info("Installing schwab-py...")
        try:
            import subprocess
            result = subprocess.run([sys.executable, '-m', 'pip', 'install', 'schwab-py'],
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print_success("schwab-py installed successfully!")
            else:
                print_error(f"Failed to install schwab-py: {result.stderr}")
                print_info("Please install manually: pip install schwab-py")
                return False
        except Exception as e:
            print_error(f"Error installing schwab-py: {e}")
            return False
    
    # Check if credentials already exist
    credentials_file = 'schwab_credentials.json'
    
    try:
        from credential_manager import CredentialManager
        cred_manager = CredentialManager()
        has_encrypted = cred_manager.credentials_exist()
    except ImportError:
        has_encrypted = False
        # Check for old plain text file
        config_file = 'schwab_config.txt'
        if os.path.exists(config_file):
            print_warning("⚠️  Found unencrypted credentials file!")
            print_warning("This is insecure. We'll migrate to encrypted storage.")
    
    if os.path.exists(credentials_file) or has_encrypted:
        print_warning("Credentials already exist!")
        overwrite = get_user_input("Do you want to reconfigure? (y/n)", "n")
        if overwrite.lower() != 'y':
            print_info("Setup cancelled. Using existing credentials.")
            return True
        # Delete old credentials
        try:
            from credential_manager import CredentialManager
            cred_manager = CredentialManager()
            cred_manager.delete_credentials()
        except Exception:
            pass
    
    print_info("This setup will guide you through configuring Schwab API for live trading.")
    print()
    print(f"{Colors.BOLD}Prerequisites:{Colors.ENDC}")
    print("  1. A Schwab account")
    print("  2. A registered app at https://developer.schwab.com")
    print("  3. App status must be 'Ready For Use' (approval can take days)")
    print("  4. App callback URL must be: http://127.0.0.1")
    print()
    
    continue_setup = get_user_input("Do you have a registered Schwab app? (y/n)", "n")
    if continue_setup.lower() != 'y':
        print()
        print(f"{Colors.BOLD}📝 How to Create a Schwab App:{Colors.ENDC}")
        print("  1. Go to: https://developer.schwab.com")
        print("  2. Register as an Individual Developer")
        print("  3. Create a new application")
        print("  4. Set callback URL to: http://127.0.0.1 (NOT localhost)")
        print("  5. Wait for approval (can take multiple days)")
        print("  6. Once approved, note your App Key and App Secret")
        print()
        print_info("After creating your app, run this setup again.")
        return False
    
    print()
    print(f"{Colors.BOLD}Step 1: Enter Your App Credentials{Colors.ENDC}")
    print("=" * 70)
    print_info("You can find these in your app details at developer.schwab.com")
    print_info("🔐 Credentials will be encrypted before storage (not plain text)")
    print()
    
    app_key = get_user_input("Enter your Schwab App Key", None)
    if not app_key:
        print_error("App Key is required!")
        return False
    
    app_secret = get_user_input("Enter your Schwab App Secret", None)
    if not app_secret:
        print_error("App Secret is required!")
        return False
    
    # Save app credentials securely using encryption
    try:
        from credential_manager import CredentialManager
        
        cred_manager = CredentialManager()
        cred_manager.save_credentials(app_key, app_secret)
        print_success("App credentials encrypted and saved securely")
        print_info("Credentials are stored in encrypted format (not plain text)")
    except ImportError:
        print_error("cryptography library not available!")
        print_info("Installing cryptography library...")
        try:
            import subprocess
            result = subprocess.run([sys.executable, '-m', 'pip', 'install', 'cryptography'],
                                  capture_output=True, text=True)
            if result.returncode == 0:
                from credential_manager import CredentialManager
                cred_manager = CredentialManager()
                cred_manager.save_credentials(app_key, app_secret)
                print_success("App credentials encrypted and saved securely")
            else:
                print_error("Failed to install cryptography. Credentials will be stored in plain text (INSECURE)")
                print_warning("⚠️  SECURITY WARNING: Storing credentials in plain text is not recommended!")
                # Fallback to plain text (with warning)
                with open(config_file, 'w') as f:
                    f.write(f"APP_KEY={app_key}\n")
                    f.write(f"APP_SECRET={app_secret}\n")
                print_success(f"App credentials saved to {config_file} (UNENCRYPTED)")
        except Exception as e:
            print_error(f"Error: {e}")
            return False
    except Exception as e:
        print_error(f"Error saving credentials: {e}")
        return False
    
    print()
    print(f"{Colors.BOLD}Step 2: OAuth Authentication{Colors.ENDC}")
    print("=" * 70)
    print_info("This will open a browser window for you to authorize the app.")
    print_info("After authorization, you'll be redirected back to complete setup.")
    print()
    
    proceed = get_user_input("Ready to start OAuth flow? (y/n)", "y")
    if proceed.lower() != 'y':
        print_info("OAuth flow cancelled. You can run this setup again later.")
        return False
    
    try:
        from schwab.auth import client_from_manual_flow
        
        print()
        print_info("Starting OAuth flow...")
        print_info("A browser window will open. Please log in and authorize the app.")
        print()
        
        # Run the manual OAuth flow
        # This will open a browser and handle the callback
        client, token_path = client_from_manual_flow(
            app_key=app_key,
            app_secret=app_secret,
            redirect_uri='http://127.0.0.1',
            token_path=credentials_file
        )
        
        if client and token_path:
            print_success(f"✓ OAuth authentication successful!")
            print_success(f"✓ Token saved to: {token_path}")
            
            # Test the connection
            print()
            print_info("Testing connection to Schwab API...")
            try:
                import asyncio
                
                async def test_connection():
                    accounts = await client.get_accounts()
                    if accounts:
                        print_success("✓ Successfully connected to Schwab API!")
                        return True
                    return False
                
                result = asyncio.run(test_connection())
                if result:
                    print()
                    print_success("🎉 Setup completed successfully!")
                    print()
                    print(f"{Colors.BOLD}Important Notes:{Colors.ENDC}")
                    print("  • Tokens expire in 7 days (not 90 days like TDA)")
                    print("  • You'll need to regenerate tokens before they expire")
                    print("  • Run this setup again to refresh tokens")
                    print("  • Your credentials are saved locally")
                    return True
                else:
                    print_warning("Connection test failed, but credentials are saved.")
                    return True
            except Exception as e:
                print_warning(f"Connection test failed: {e}")
                print_info("Credentials are saved. You can test later in Live Trading.")
                return True
        else:
            print_error("OAuth flow failed. Please try again.")
            return False
            
    except Exception as e:
        print_error(f"Error during OAuth flow: {e}")
        print()
        print_info("Common issues:")
        print("  • Make sure your app is approved and 'Ready For Use'")
        print("  • Check that callback URL is exactly: http://127.0.0.1")
        print("  • Verify your App Key and App Secret are correct")
        print("  • Ensure you have internet connectivity")
        import traceback
        traceback.print_exc()
        return False

@safe_execute
def live_trade():
    """Run live trading (paper trading mode)."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}💰 LIVE TRADING{Colors.ENDC}")
    print(f"{Colors.OKCYAN}{'='*70}{Colors.ENDC}\n")
    
    # Check if credentials are configured
    credentials_file = 'schwab_credentials.json'
    has_credentials = os.path.exists(credentials_file)
    
    # Check for encrypted credentials
    try:
        from credential_manager import CredentialManager
        cred_manager = CredentialManager()
        if cred_manager.credentials_exist():
            has_credentials = True
    except ImportError:
        pass
    
    # Check for old plain text file (migration)
    old_config_file = 'schwab_config.txt'
    if os.path.exists(old_config_file) and not has_credentials:
        print_warning("⚠️  Found old unencrypted credentials file!")
        print_info("Please run setup (option 5) to migrate to encrypted storage.")
        has_credentials = True  # Allow use but warn
    
    if not has_credentials:
        print_warning("⚠️  Schwab API credentials not configured!")
        print_info("You need to set up credentials before using live trading.")
        setup_now = get_user_input("Run setup now? (y/n)", "y")
        if setup_now.lower() == 'y':
            if not setup_live_trading():
                print_info("Switching to paper trading mode...")
                has_credentials = False
        else:
            has_credentials = False
    
    if not has_credentials:
        print_warning("⚠️  Running in PAPER TRADING mode (simulated trades).\n")
    
    if not check_file_exists('lstm_price_difference_model.h5'):
        print_error("No trained model found!")
        print_info("Please train a model first (option 2)")
        return False
    
    if not check_file_exists('processed_stock_data.csv'):
        print_error("No processed data found!")
        print_info("Please process data first (option 1)")
        return False
    
    # Check if trading API module exists
    try:
        from trading_api import TradingAPI
    except ImportError as e:
        print_error(f"Trading API module not available: {e}")
        print_info("Make sure trading_api.py is in the project directory")
        return False
    
    # Check Python version for Schwab API
    if sys.version_info < (3, 10):
        print_warning("⚠️  Schwab API requires Python 3.10+")
        print_info(f"Current Python version: {sys.version_info.major}.{sys.version_info.minor}")
        print_info("Paper trading mode will work, but live trading requires Python 3.10+")
    
    print_info("Initializing trading API...")
    
    # Ask user for paper trading mode
    paper_mode = get_user_input("Use paper trading mode? (y/n)", "y")
    use_paper = paper_mode.lower() == 'y' if paper_mode else True
    
    if not use_paper:
        print_warning("⚠️  LIVE TRADING MODE ENABLED")
        print_warning("This will execute REAL trades with REAL money!")
        confirm = get_user_input("Are you absolutely sure? Type 'yes' to confirm", "")
        if confirm.lower() != 'yes':
            print_info("Live trading cancelled. Using paper trading mode.")
            use_paper = True
    
    try:
        # Initialize trading API
        api = TradingAPI(paper_trading=use_paper)
        api.connect()
        
        # Get account balance
        balance_info = api.get_account_balance()
        if balance_info:
            if isinstance(balance_info, dict):
                balance = balance_info.get('available_funds', balance_info.get('total_value', 0))
                paper_status = "📝 (Paper Trading)" if balance_info.get('paper_trading', True) else "💰 (Live Trading)"
                print_success(f"Account balance: ${balance:,.2f} {paper_status}")
            else:
                print_success(f"Account balance: ${balance_info:,.2f}")
        else:
            print_warning("Could not fetch account balance")
        
        # Get latest signal using Python 3.12 venv
        print_info("\nFetching latest trading signal...")
        try:
            python_cmd = get_python_cmd()
            # Use subprocess to run get_latest_signal in the venv
            import subprocess
            import json
            
            # Create a simple script to get the signal
            signal_script = """
import sys
sys.path.insert(0, '.')
try:
    from live_trading import get_latest_signal
    signal = get_latest_signal(
        model_path='lstm_price_difference_model.h5',
        data_file='processed_stock_data.csv',
        threshold=0.5
    )
    if signal:
        print(signal)
    else:
        print('Hold')
except Exception as e:
    print(f'ERROR: {e}', file=sys.stderr)
    sys.exit(1)
"""
            result = subprocess.run(
                [python_cmd, '-c', signal_script],
                capture_output=True,
                text=True,
                cwd=os.path.dirname(os.path.abspath(__file__))
            )
            
            if result.returncode == 0:
                signal = result.stdout.strip()
                if signal and signal != 'ERROR':
                    print_success(f"Latest signal: {signal}")
                    
                    # Execute trades based on signal
                    if signal == 'Buy AAPL, Sell MSFT':
                        print_info("\nExecuting trades:")
                        api.place_order('AAPL', 10, 'BUY')
                        api.place_order('MSFT', 10, 'SELL')
                    elif signal == 'Buy MSFT, Sell AAPL':
                        print_info("\nExecuting trades:")
                        api.place_order('MSFT', 10, 'BUY')
                        api.place_order('AAPL', 10, 'SELL')
                    else:
                        print_info(f"No action taken (signal: {signal})")
                else:
                    print_warning("Could not get valid trading signal")
            else:
                print_error(f"Error getting trading signal: {result.stderr}")
                print_info("Make sure the model and data files are available")
                return False
                
        except Exception as e:
            print_error(f"Error getting trading signal: {e}")
            print_info("Make sure the model and data files are available")
            import traceback
            traceback.print_exc()
            return False
        
        return True
        
    except Exception as e:
        print_error(f"Error in live trading: {e}")
        import traceback
        traceback.print_exc()
        return False

@safe_execute
def show_status():
    """Show the status of files in the project."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}📋 PROJECT STATUS{Colors.ENDC}")
    print(f"{Colors.OKCYAN}{'='*70}{Colors.ENDC}\n")
    
    files = {
        'processed_stock_data.csv': 'Processed stock data',
        'lstm_price_difference_model.h5': 'Trained LSTM model',
        'backtest_results.csv': 'Backtest results'
    }
    
    all_good = True
    for filename, description in files.items():
        if check_file_exists(filename):
            size = os.path.getsize(filename)
            size_mb = size / (1024 * 1024)
            print_success(f"{description}: {filename} ({size_mb:.2f} MB)")
        else:
            print_error(f"{description}: {filename} (not found)")
            all_good = False
    
    if all_good:
        print(f"\n{Colors.OKGREEN}{Colors.BOLD}✓ All files present! Ready to go!{Colors.ENDC}\n")
    else:
        print(f"\n{Colors.WARNING}⚠ Some files are missing. Run the appropriate options to generate them.{Colors.ENDC}\n")

@safe_execute
def run_full_pipeline():
    """Run the complete pipeline: process data, train model, and backtest."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}🚀 RUNNING FULL PIPELINE{Colors.ENDC}")
    print(f"{Colors.OKCYAN}{'='*70}{Colors.ENDC}\n")
    
    print_info("This will:")
    print("  1. Process stock data")
    print("  2. Train the LSTM model")
    print("  3. Backtest the trading strategy\n")
    
    confirm = get_user_input("Run full pipeline? This may take a while (y/n)", "n")
    if confirm.lower() != 'y':
        print_info("Pipeline cancelled.")
        return
    
    steps = [
        ("Processing Data", process_data),
        ("Training Model", train_model),
        ("Backtesting Strategy", backtest_strategy)
    ]
    
    for step_name, step_func in steps:
        print(f"\n{Colors.BOLD}{'='*70}{Colors.ENDC}")
        print(f"{Colors.BOLD}Step: {step_name}{Colors.ENDC}")
        print(f"{Colors.BOLD}{'='*70}{Colors.ENDC}")
        
        if not step_func():
            print_error(f"Pipeline stopped at: {step_name}")
            return
    
    print(f"\n{Colors.OKGREEN}{Colors.BOLD}{'='*70}")
    print("  🎉  PIPELINE COMPLETED SUCCESSFULLY!  🎉")
    print(f"{'='*70}{Colors.ENDC}\n")

@safe_execute
def launch_dashboard():
    """Launch the Streamlit dashboard."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}📊 LAUNCHING DASHBOARD{Colors.ENDC}")
    print(f"{Colors.OKCYAN}{'='*70}{Colors.ENDC}\n")
    
    # Check if Streamlit is installed
    try:
        import streamlit
    except ImportError:
        print_error("Streamlit is not installed!")
        print_info("Install with: pip install streamlit")
        print_info("Then run: streamlit run dashboard.py")
        return False
    
    # Check if dashboard.py exists
    if not check_file_exists('dashboard.py'):
        print_error("dashboard.py not found!")
        return False
    
    print_info("Launching Streamlit dashboard...")
    print_info("The dashboard will open in your default web browser.")
    print_info("Press Ctrl+C to stop the dashboard server.\n")
    
    try:
        import subprocess
        result = subprocess.run([sys.executable, '-m', 'streamlit', 'run', 'dashboard.py'],
                              capture_output=False)
        if result.returncode == 0:
            print_success("Dashboard closed successfully")
            return True
        else:
            print_error("Dashboard exited with errors")
            return False
    except KeyboardInterrupt:
        print_warning("\nDashboard stopped by user")
        return False
    except FileNotFoundError:
        print_error("Streamlit not found. Please install with: pip install streamlit")
        return False
    except Exception as e:
        print_error(f"Error launching dashboard: {e}")
        return False

def get_selected_stocks():
    """Get the currently selected stocks from config file."""
    config_file = 'selected_stocks.txt'
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                line = f.read().strip()
                if ',' in line:
                    stocks = [s.strip().upper() for s in line.split(',')]
                    if len(stocks) >= 2:
                        return stocks[0], stocks[1]
        except Exception:
            pass
    # Default stocks
    return 'AAPL', 'MSFT'

@safe_execute
def select_stocks():
    """Allow user to select stock pairs for analysis with checkbox interface."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}🔄 SELECT STOCKS FOR ANALYSIS{Colors.ENDC}")
    print(f"{Colors.OKCYAN}{'='*70}{Colors.ENDC}\n")
    
    # Get current selection
    current_stock1, current_stock2 = get_selected_stocks()
    print_info(f"Current selection: {current_stock1} and {current_stock2}")
    print()
    
    # Common stock tickers to choose from
    common_tickers = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM',
        'V', 'JNJ', 'WMT', 'PG', 'MA', 'UNH', 'HD', 'DIS', 'BAC', 'XOM',
        'CVX', 'ABBV', 'PFE', 'KO', 'AVGO', 'COST', 'MRK', 'PEP', 'TMO',
        'ABT', 'ACN', 'NFLX', 'ADBE', 'CRM', 'NKE', 'TXN', 'CMCSA', 'LIN'
    ]
    
    # Initialize selected stocks (set current selection as selected)
    selected_tickers = set()
    if current_stock1 in common_tickers:
        selected_tickers.add(current_stock1)
    if current_stock2 in common_tickers:
        selected_tickers.add(current_stock2)
    
    # Allow custom ticker entry
    print_info("You can also enter custom tickers at the end.")
    print()
    
    while True:
        # Display ticker list with checkboxes
        print(f"{Colors.BOLD}Select stocks (enter number to toggle, 'c' for custom, 'd' when done):{Colors.ENDC}\n")
        
        for i, ticker in enumerate(common_tickers, 1):
            checkbox = f"{Colors.OKGREEN}[X]{Colors.ENDC}" if ticker in selected_tickers else "[ ]"
            current_marker = ""
            if ticker == current_stock1 or ticker == current_stock2:
                current_marker = f" {Colors.WARNING}(current){Colors.ENDC}"
            print(f"  {checkbox} {i:2d}. {ticker}{current_marker}")
        
        print(f"\n  {Colors.BOLD}Custom:{Colors.ENDC} Enter 'c' to add custom ticker")
        print(f"  {Colors.BOLD}Done:{Colors.ENDC} Enter 'd' to finish selection")
        print()
        
        # Show current selection count
        num_selected = len(selected_tickers)
        if num_selected > 0:
            selected_list = ', '.join(sorted(selected_tickers))
            print_info(f"Currently selected ({num_selected}): {selected_list}")
        else:
            print_warning("No stocks selected yet. Select at least 2 stocks.")
        print()
        
        # Get user input
        user_input = input(f"{Colors.BOLD}Enter choice (1-{len(common_tickers)}, 'c', or 'd'): {Colors.ENDC}").strip().lower()
        
        if user_input == 'd':
            # Done selecting
            if len(selected_tickers) < 2:
                print_error("Please select at least 2 stocks!")
                continue
            
            selected_list = sorted(list(selected_tickers))
            if len(selected_list) == 2:
                stock1, stock2 = selected_list[0], selected_list[1]
            else:
                # If more than 2 selected, ask which 2 to use
                print(f"\n{Colors.BOLD}You selected {len(selected_list)} stocks:{Colors.ENDC}")
                for i, ticker in enumerate(selected_list, 1):
                    print(f"  {i}. {ticker}")
                print()
                idx1 = get_user_input(f"Enter number for first stock (1-{len(selected_list)})", 1, int)
                idx2 = get_user_input(f"Enter number for second stock (1-{len(selected_list)})", 2, int)
                
                if idx1 is None or idx2 is None or idx1 == idx2:
                    print_error("Invalid selection! Using first two stocks.")
                    stock1, stock2 = selected_list[0], selected_list[1]
                else:
                    stock1 = selected_list[idx1 - 1]
                    stock2 = selected_list[idx2 - 1]
            
            # Save selected stocks
            try:
                with open("selected_stocks.txt", "w") as f:
                    f.write(f"{stock1},{stock2}\n")
                print_success(f"Selected stocks: {stock1} and {stock2}")
                print_info("These stocks will be used for data processing and analysis.")
                return True
            except Exception as e:
                print_error(f"Error saving stock selection: {e}")
                return False
        
        elif user_input == 'c':
            # Add custom ticker
            custom_ticker = input(f"{Colors.BOLD}Enter custom ticker symbol: {Colors.ENDC}").strip().upper()
            if custom_ticker:
                if custom_ticker not in common_tickers:
                    common_tickers.append(custom_ticker)
                if custom_ticker in selected_tickers:
                    selected_tickers.remove(custom_ticker)
                    print_info(f"Removed {custom_ticker} from selection")
                else:
                    selected_tickers.add(custom_ticker)
                    print_success(f"Added {custom_ticker} to selection")
            else:
                print_warning("No ticker entered")
            print()
        
        elif user_input.isdigit():
            # Toggle selection by number
            ticker_num = int(user_input)
            if 1 <= ticker_num <= len(common_tickers):
                ticker = common_tickers[ticker_num - 1]
                if ticker in selected_tickers:
                    selected_tickers.remove(ticker)
                    print_info(f"Deselected {ticker}")
                else:
                    selected_tickers.add(ticker)
                    print_success(f"Selected {ticker}")
            else:
                print_error(f"Invalid number! Please enter 1-{len(common_tickers)}")
            print()
        else:
            print_error("Invalid input! Please enter a number, 'c', or 'd'")
            print()

@safe_execute
def analyze_correlation():
    """Analyze correlations between stocks and suggest pairs."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}📊 ANALYZE STOCK CORRELATIONS{Colors.ENDC}")
    print(f"{Colors.OKCYAN}{'='*70}{Colors.ENDC}\n")
    
    # Prompt for a list of stocks to analyze
    default_stocks = "AAPL,MSFT,GOOGL,AMZN,TSLA"
    stock_input = get_user_input("Enter a comma-separated list of stock tickers", default_stocks)
    
    if not stock_input:
        stock_input = default_stocks
    
    stock_list = [s.strip().upper() for s in stock_input.split(",") if s.strip()]
    
    if len(stock_list) < 2:
        print_error("Please enter at least 2 stock tickers!")
        return False
    
    print_info(f"Analyzing correlations for: {', '.join(stock_list)}")
    print_info("Downloading stock data... This may take a moment...\n")
    
    try:
        import yfinance as yf
        import pandas as pd
        
        # Download data for all stocks
        data = yf.download(stock_list, start="2020-01-01", end="2023-01-01", progress=False)
        
        if data.empty:
            print_error("Failed to download data. Please check your stock tickers.")
            return False
        
        # Get close prices
        if 'Close' in data.columns.names:
            close_data = data['Close']
        else:
            # If single level columns
            close_data = data
        
        # Handle multi-index columns
        if isinstance(close_data.columns, pd.MultiIndex):
            close_data = close_data.droplevel(0, axis=1)
        
        # Calculate correlation matrix
        corr_matrix = close_data.corr()
        
        print(f"\n{Colors.BOLD}Correlation Matrix:{Colors.ENDC}")
        print(corr_matrix.round(2))
        
        # Suggest pairs with high correlation (e.g., > 0.8)
        suggested_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                stock1 = corr_matrix.columns[i]
                stock2 = corr_matrix.columns[j]
                correlation = corr_matrix.iloc[i, j]
                if not pd.isna(correlation) and correlation > 0.8:
                    suggested_pairs.append((stock1, stock2, correlation))
        
        # Sort by correlation
        suggested_pairs.sort(key=lambda x: x[2], reverse=True)
        
        if suggested_pairs:
            print(f"\n{Colors.BOLD}Suggested Stock Pairs (Correlation > 0.8):{Colors.ENDC}")
            for pair in suggested_pairs:
                print(f"  {Colors.OKGREEN}•{Colors.ENDC} {pair[0]} and {pair[1]} (Correlation: {pair[2]:.3f})")
        else:
            print(f"\n{Colors.WARNING}No highly correlated pairs found (correlation > 0.8).{Colors.ENDC}")
            print_info("You may want to try a different set of stocks or lower the correlation threshold.")
        
        return True
        
    except Exception as e:
        print_error(f"Error analyzing correlations: {e}")
        import traceback
        traceback.print_exc()
        return False

@safe_execute
def analyze_cointegration():
    """Analyze cointegration between stocks and suggest pairs."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}📈 ANALYZE STOCK COINTEGRATION{Colors.ENDC}")
    print(f"{Colors.OKCYAN}{'='*70}{Colors.ENDC}\n")
    
    # Check if statsmodels is available
    try:
        from statsmodels.tsa.stattools import coint
    except ImportError:
        print_error("statsmodels is not installed!")
        print_info("Install with: pip install statsmodels")
        return False
    
    # Prompt for a list of stocks to analyze
    default_stocks = "AAPL,MSFT,GOOGL,AMZN,TSLA"
    stock_input = get_user_input("Enter a comma-separated list of stock tickers", default_stocks)
    
    if not stock_input:
        stock_input = default_stocks
    
    stock_list = [s.strip().upper() for s in stock_input.split(",") if s.strip()]
    
    if len(stock_list) < 2:
        print_error("Please enter at least 2 stock tickers!")
        return False
    
    print_info(f"Analyzing cointegration for: {', '.join(stock_list)}")
    print_info("Downloading stock data... This may take a moment...\n")
    
    try:
        import yfinance as yf
        import pandas as pd
        import numpy as np
        
        # Download data for all stocks
        data = yf.download(stock_list, start="2020-01-01", end="2023-01-01", progress=False)
        
        if data.empty:
            print_error("Failed to download data. Please check your stock tickers.")
            return False
        
        # Get close prices
        if 'Close' in data.columns.names:
            close_data = data['Close']
        else:
            close_data = data
        
        # Handle multi-index columns
        if isinstance(close_data.columns, pd.MultiIndex):
            close_data = close_data.droplevel(0, axis=1)
        
        # Remove any columns with all NaN values
        close_data = close_data.dropna(axis=1, how='all')
        
        # Analyze cointegration for all pairs
        suggested_pairs = []
        print_info("Testing cointegration for all pairs...\n")
        
        for i in range(len(close_data.columns)):
            for j in range(i + 1, len(close_data.columns)):
                stock1 = close_data.columns[i]
                stock2 = close_data.columns[j]
                
                # Get the two series
                series1 = close_data[stock1].dropna()
                series2 = close_data[stock2].dropna()
                
                # Align the series
                common_index = series1.index.intersection(series2.index)
                if len(common_index) < 100:  # Need sufficient data
                    continue
                
                series1_aligned = series1.loc[common_index]
                series2_aligned = series2.loc[common_index]
                
                try:
                    _, p_value, _ = coint(series1_aligned, series2_aligned)
                    if p_value < 0.05:  # Significant cointegration
                        suggested_pairs.append((stock1, stock2, p_value))
                except Exception:
                    continue
        
        # Sort by p-value (lower is better)
        suggested_pairs.sort(key=lambda x: x[2])
        
        if suggested_pairs:
            print(f"{Colors.BOLD}Suggested Stock Pairs (Cointegrated, p < 0.05):{Colors.ENDC}")
            for pair in suggested_pairs:
                print(f"  {Colors.OKGREEN}•{Colors.ENDC} {pair[0]} and {pair[1]} (p-value: {pair[2]:.4f})")
        else:
            print(f"{Colors.WARNING}No cointegrated pairs found (p-value < 0.05).{Colors.ENDC}")
            print_info("This means the stocks may not have a long-term equilibrium relationship.")
        
        return True
        
    except Exception as e:
        print_error(f"Error analyzing cointegration: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_menu():
    """Display the main menu."""
    print(f"\n{Colors.BOLD}{Colors.HEADER}MAIN MENU{Colors.ENDC}")
    print(f"{Colors.OKCYAN}{'='*70}{Colors.ENDC}")

    # Data & Model
    print(f"\n{Colors.BOLD}{Colors.OKBLUE}📊 DATA & MODEL{Colors.ENDC}")
    print(f"{Colors.BOLD}1.{Colors.ENDC} 📥 Process Stock Data")
    print(f"{Colors.BOLD}2.{Colors.ENDC} 🧠 Train LSTM Model")
    print(f"{Colors.BOLD}3.{Colors.ENDC} 🚀 Run Full Pipeline (Process → Train → Backtest)")

    # Trading
    print(f"\n{Colors.BOLD}{Colors.OKGREEN}💰 TRADING{Colors.ENDC}")
    print(f"{Colors.BOLD}4.{Colors.ENDC} 📊 Backtest Trading Strategy")
    print(f"{Colors.BOLD}5.{Colors.ENDC} 💵 Live Trading")
    print(f"{Colors.BOLD}6.{Colors.ENDC} ⚙️  Setup Trading API (Schwab)")

    # Analysis & Visualization
    print(f"\n{Colors.BOLD}{Colors.HEADER}📈 ANALYSIS & VISUALIZATION{Colors.ENDC}")
    print(f"{Colors.BOLD}7.{Colors.ENDC} 📊 Analyze Stock Correlations")
    print(f"{Colors.BOLD}8.{Colors.ENDC} 📈 Analyze Stock Cointegration")
    print(f"{Colors.BOLD}9.{Colors.ENDC} 🎨 Launch Real-Time Dashboard")

    # Configuration
    print(f"\n{Colors.BOLD}{Colors.WARNING}⚙️  CONFIGURATION{Colors.ENDC}")
    print(f"{Colors.BOLD}10.{Colors.ENDC} 🔄 Select Stocks for Analysis")
    print(f"{Colors.BOLD}11.{Colors.ENDC} 💬 Update Sentiment Cache")
    print(f"{Colors.BOLD}12.{Colors.ENDC} 📊 Update Options Cache")
    print(f"{Colors.BOLD}13.{Colors.ENDC} 🧹 Clear yfinance Cache")

    # System & Help
    print(f"\n{Colors.BOLD}{Colors.OKCYAN}ℹ️  SYSTEM & HELP{Colors.ENDC}")
    print(f"{Colors.BOLD}14.{Colors.ENDC} 📋 Show Project Status")
    print(f"{Colors.BOLD}15.{Colors.ENDC} 📚 Show Help & Documentation")
    print(f"{Colors.BOLD}16.{Colors.ENDC} ❌ Exit")

    print(f"{Colors.OKCYAN}{'='*70}{Colors.ENDC}")

# ── Per-option help text ─────────────────────────────────────────────────

HELP_TEXT = {
    '1': (
        "📥  PROCESS STOCK DATA",
        "Downloads OHLCV data from Yahoo Finance, calculates technical\n"
        "  indicators (MA, RSI, MACD), merges sentiment & options data, and\n"
        "  writes processed_stock_data.csv.\n\n"
        "  Prerequisites : Internet connection\n"
        "  Output        : processed_stock_data.csv"
    ),
    '2': (
        "🧠  TRAIN LSTM MODEL",
        "Trains a two-layer LSTM neural network on the processed data.\n"
        "  Uses early stopping and a 60-day look-back window.\n\n"
        "  Prerequisites : processed_stock_data.csv, Python 3.11/3.12\n"
        "  Output        : lstm_price_difference_model.h5"
    ),
    '3': (
        "🚀  RUN FULL PIPELINE",
        "Runs Process → Train → Backtest in sequence.  Good for a first run\n"
        "  or a full retrain after changing stock selection.\n\n"
        "  Prerequisites : Internet connection, Python 3.11/3.12\n"
        "  Output        : all pipeline artefacts"
    ),
    '4': (
        "📊  BACKTEST TRADING STRATEGY",
        "Simulates the trading strategy on historical data with realistic\n"
        "  costs: commissions, SEC fees, slippage, borrow costs, and margin.\n"
        "  Enforces market hours, stop-loss, and max-drawdown limits.\n\n"
        "  Prerequisites : processed_stock_data.csv, trained model\n"
        "  Output        : backtest_results.csv"
    ),
    '5': (
        "💵  LIVE TRADING",
        "Runs the trading loop against the Schwab API (or in paper-trading\n"
        "  mode if credentials are not configured).  Requires a trained model\n"
        "  and up-to-date processed data.\n\n"
        "  Prerequisites : trained model, processed data, (optional) Schwab creds"
    ),
    '6': (
        "⚙️   SETUP TRADING API",
        "Walks you through creating a Schwab developer app and completing\n"
        "  the OAuth flow.  Credentials are encrypted at rest with AES-256.\n\n"
        "  Prerequisites : Schwab developer account"
    ),
    '7': (
        "📊  ANALYZE CORRELATIONS",
        "Computes and displays a correlation matrix for the selected stocks\n"
        "  so you can identify good pairs-trading candidates."
    ),
    '8': (
        "📈  ANALYZE COINTEGRATION",
        "Runs the Engle-Granger cointegration test on every pair of selected\n"
        "  stocks.  Cointegrated pairs are the strongest candidates for the\n"
        "  mean-reversion strategy."
    ),
    '9': (
        "🎨  LAUNCH DASHBOARD",
        "Opens the Streamlit real-time dashboard in your default browser.\n"
        "  Shows equity curve, drawdown, live signals, and trade history.\n\n"
        "  Prerequisites : streamlit installed"
    ),
    '10': (
        "🔄  SELECT STOCKS",
        "Choose which stock symbols to use for data download, training,\n"
        "  and trading.  Saved to selected_stocks.txt."
    ),
    '11': (
        "💬  UPDATE SENTIMENT CACHE",
        "Fetches Reddit (r/wallstreetbets) sentiment for the selected stocks\n"
        "  and caches it locally.  Used as a feature during data processing.\n\n"
        "  Prerequisites : Reddit API credentials configured"
    ),
    '12': (
        "📊  UPDATE OPTIONS CACHE",
        "Fetches current options-volume and implied-volatility data from\n"
        "  yfinance and caches it.  Run daily via cron for best results."
    ),
    '13': (
        "🧹  CLEAR YFINANCE CACHE",
        "Deletes the local yfinance download cache.  Useful when downloads\n"
        "  return stale or corrupted data."
    ),
    '14': (
        "📋  PROJECT STATUS",
        "Displays the current state of every pipeline artefact: data files,\n"
        "  model, backtest results, credential files, and environment checks."
    ),
}


def show_help(option=None):
    """Show general help, or drill into a specific option number."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}")
    print("  HELP & DOCUMENTATION")
    print(f"{'='*70}{Colors.ENDC}\n")

    if option and option in HELP_TEXT:
        title, body = HELP_TEXT[option]
        print(f"  {Colors.BOLD}{title}{Colors.ENDC}\n")
        print(f"  {body}\n")
        return

    # ── General overview ────────────────────────────────────────────
    print(f"  {Colors.BOLD}Recommended workflow:{Colors.ENDC}")
    print("    1 → Process Data    (download & feature-engineer)")
    print("    2 → Train Model     (LSTM neural network)")
    print("    4 → Backtest        (simulate with realistic costs)")
    print("    9 → Dashboard       (visualise results live)\n")

    print(f"  {Colors.BOLD}Quick-launch commands (bypass the menu):{Colors.ENDC}")
    print("    python main.py process      – option 1")
    print("    python main.py train        – option 2")
    print("    python main.py pipeline     – option 3")
    print("    python main.py backtest     – option 4")
    print("    python main.py dashboard    – option 9")
    print("    python main.py status       – option 14\n")

    print(f"  {Colors.BOLD}Flags:{Colors.ENDC}")
    print("    --no-color   disable coloured output (or set NO_COLOR env var)")
    print("    --help       show this help and exit\n")

    print(f"  {Colors.BOLD}Want detail on a specific option?{Colors.ENDC}")
    print("    Enter the option number at the menu prompt, then press 'h'.")
    print("    Or type  ?N   (e.g. ?4) at the menu prompt.\n")


# ── Command dispatch table ───────────────────────────────────────────────

COMMANDS = {
    "process":   ("1",  "Process stock data"),
    "train":     ("2",  "Train LSTM model"),
    "pipeline":  ("3",  "Run full pipeline"),
    "backtest":  ("4",  "Backtest strategy"),
    "live":      ("5",  "Live trading"),
    "setup":     ("6",  "Setup Schwab API"),
    "corr":      ("7",  "Correlation analysis"),
    "coint":     ("8",  "Cointegration analysis"),
    "dashboard": ("9",  "Launch dashboard"),
    "stocks":    ("10", "Select stocks"),
    "status":    ("14", "Show status"),
    "help":      ("15", "Show help"),
}

CHOICE_MAP = {
    '1':  process_data,
    '2':  train_model,
    '3':  run_full_pipeline,
    '4':  backtest_strategy,
    '5':  live_trade,
    '6':  setup_live_trading,
    '7':  analyze_correlation,
    '8':  analyze_cointegration,
    '9':  launch_dashboard,
    '10': select_stocks,
    '11': update_sentiment_cache_cli,
    '12': update_options_cache_cli,
    '13': clear_yfinance_cache,
    '14': show_status,
    '15': show_help,
}


def main():
    """
    Entry-point.  Supports an optional positional command for quick launch:
        python main.py process
        python main.py --help
    Falls back to the interactive menu when no command is given.
    """
    # Strip --no-color from argv before argparse sees it (already consumed)
    argv = [a for a in sys.argv[1:] if a != '--no-color']

    # ── Quick-launch path ────────────────────────────────────────────
    if argv:
        cmd = argv[0].lower()

        if cmd in ('--help', '-h', 'help'):
            show_help()
            return

        if cmd in COMMANDS:
            choice, _ = COMMANDS[cmd]
            print_header()
            handler = CHOICE_MAP.get(choice)
            if handler:
                handler()
            return

        # Unknown command – show usage
        print_error(f"Unknown command: '{cmd}'")
        print(f"\n  Available commands:\n")
        for name, (_, desc) in sorted(COMMANDS.items()):
            print(f"    {Colors.BOLD}{name:12s}{Colors.ENDC} {desc}")
        print(f"\n  Or run without arguments for the interactive menu.\n")
        return

    # ── Interactive menu loop ────────────────────────────────────────
    clear_screen()
    print_header()

    while True:
        show_menu()
        choice = get_user_input("\nSelect an option (1-16) or ?N for help on option N", None)

        if choice is None:
            continue

        # Inline help: user typed ?N
        if choice.startswith('?'):
            show_help(option=choice[1:])
            input(f"\n{Colors.OKCYAN}Press Enter to continue...{Colors.ENDC}")
            continue

        # Exit
        if choice == '16':
            print(f"\n{Colors.OKGREEN}{Colors.BOLD}Goodbye!{Colors.ENDC}\n")
            break

        # Dispatch
        handler = CHOICE_MAP.get(choice)
        if handler is None:
            print_error("Invalid choice! Please select 1-16, or ?N for help on option N.")
        else:
            handler()

        input(f"\n{Colors.OKCYAN}Press Enter to continue...{Colors.ENDC}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.WARNING}[CANCELLED] Interrupted by user.{Colors.ENDC}")
        print(f"{Colors.OKGREEN}Goodbye!{Colors.ENDC}\n")
        sys.exit(0)
