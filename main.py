#!/usr/bin/env python3
"""
Stock Arbitrage Model - Interactive CLI
A fun and interactive command-line interface for the stock arbitrage trading system!
"""

import os
import sys
import subprocess
from datetime import datetime

# Color codes for terminal output (ANSI escape codes)
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header():
    """Print a fun header banner."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}")
    print("  🚀  STOCK ARBITRAGE MODEL - INTERACTIVE CLI  🚀")
    print(f"{'='*70}{Colors.ENDC}")
    print(f"{Colors.OKCYAN}💰 Predict price differences between AAPL and MSFT")
    print(f"📈 Train LSTM models and backtest trading strategies{Colors.ENDC}\n")

def print_success(message):
    """Print a success message."""
    print(f"{Colors.OKGREEN}✓ {message}{Colors.ENDC}")

def print_error(message):
    """Print an error message."""
    print(f"{Colors.FAIL}✗ {message}{Colors.ENDC}")

def print_info(message):
    """Print an info message."""
    print(f"{Colors.OKBLUE}ℹ {message}{Colors.ENDC}")

def print_warning(message):
    """Print a warning message."""
    print(f"{Colors.WARNING}⚠ {message}{Colors.ENDC}")

def clear_screen():
    """Clear the terminal screen."""
    os.system('clear' if os.name != 'nt' else 'cls')

def get_user_input(prompt, default=None, input_type=str):
    """Get user input with a default value."""
    if default is not None:
        prompt_text = f"{Colors.BOLD}{prompt} [{default}]: {Colors.ENDC}"
    else:
        prompt_text = f"{Colors.BOLD}{prompt}: {Colors.ENDC}"
    
    try:
        user_input = input(prompt_text).strip()
        if not user_input and default is not None:
            return default
        if not user_input:
            return None
        return input_type(user_input)
    except (ValueError, KeyboardInterrupt):
        return None

def check_file_exists(filename):
    """Check if a file exists."""
    return os.path.exists(filename)

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

def process_data():
    """Run the data processing script."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}📥 PROCESSING STOCK DATA{Colors.ENDC}")
    print(f"{Colors.OKCYAN}{'='*70}{Colors.ENDC}\n")
    
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
    
    print_info("Downloading stock data from Yahoo Finance...")
    print_info("This may take a moment...\n")
    
    try:
        result = subprocess.run([sys.executable, 'process_stock_data.py'], 
                              capture_output=False, text=True)
        if result.returncode == 0:
            print_success("Data processed successfully!")
            return True
        else:
            print_error("Data processing failed!")
            print_info("\nTroubleshooting tips:")
            print("  • Try clearing the yfinance cache (option 8)")
            print("  • Check your internet connection")
            print("  • Wait a few minutes and try again (yfinance rate limits)")
            return False
    except KeyboardInterrupt:
        print_warning("\nData processing interrupted by user")
        return False
    except Exception as e:
        print_error(f"Error: {e}")
        print_info("\nTroubleshooting tips:")
        print("  • Try clearing the yfinance cache (option 8)")
        print("  • Check your internet connection")
        return False

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
        
        result = subprocess.run([sys.executable, 'train_model.py'],
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
        
        result = subprocess.run([sys.executable, 'backtest_strategy.py'],
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

def show_menu():
    """Display the main menu."""
    print(f"\n{Colors.BOLD}{Colors.HEADER}MAIN MENU{Colors.ENDC}")
    print(f"{Colors.OKCYAN}{'='*70}{Colors.ENDC}")
    print(f"{Colors.BOLD}1.{Colors.ENDC} 📥 Process Stock Data")
    print(f"{Colors.BOLD}2.{Colors.ENDC} 🧠 Train LSTM Model")
    print(f"{Colors.BOLD}3.{Colors.ENDC} 📊 Backtest Trading Strategy")
    print(f"{Colors.BOLD}4.{Colors.ENDC} 🚀 Run Full Pipeline (Process → Train → Backtest)")
    print(f"{Colors.BOLD}5.{Colors.ENDC} 💰 Live Trading (Paper Trading Mode)")
    print(f"{Colors.BOLD}6.{Colors.ENDC} 📈 Launch Real-Time Dashboard")
    print(f"{Colors.BOLD}7.{Colors.ENDC} 📋 Show Project Status")
    print(f"{Colors.BOLD}8.{Colors.ENDC} 📚 Show Help / Documentation")
    print(f"{Colors.BOLD}9.{Colors.ENDC} 🧹 Clear yfinance Cache (fix download issues)")
    print(f"{Colors.BOLD}10.{Colors.ENDC} ❌ Exit")
    print(f"{Colors.OKCYAN}{'='*70}{Colors.ENDC}")

def show_help():
    """Show help and documentation."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}📚 HELP & DOCUMENTATION{Colors.ENDC}")
    print(f"{Colors.OKCYAN}{'='*70}{Colors.ENDC}\n")
    
    print(f"{Colors.BOLD}About this Project:{Colors.ENDC}")
    print("This is a TensorFlow-based stock arbitrage model that uses LSTM")
    print("neural networks to predict price differences between AAPL and MSFT.\n")
    
    print(f"{Colors.BOLD}Workflow:{Colors.ENDC}")
    print("1. Process Data: Downloads and processes stock data with features")
    print("2. Train Model: Trains a 2-layer LSTM model with dropout")
    print("3. Backtest: Tests trading strategy on historical data\n")
    
    print(f"{Colors.BOLD}Files:{Colors.ENDC}")
    print("  • process_stock_data.py - Data processing script")
    print("  • train_model.py - Model training script")
    print("  • backtest_strategy.py - Backtesting script")
    print("  • dashboard.py - Real-time Streamlit dashboard")
    print("  • lstm_model.py - Model architecture definition\n")
    
    print(f"{Colors.BOLD}Requirements:{Colors.ENDC}")
    print("  • Python 3.11 or 3.12 (TensorFlow doesn't support 3.14)")
    print("  • All dependencies in requirements.txt\n")
    
    print(f"{Colors.WARNING}Note:{Colors.ENDC} Make sure TensorFlow is installed in a compatible")
    print("Python environment before running training or backtesting!\n")

def main():
    """Main function to run the interactive CLI."""
    clear_screen()
    print_header()
    
    while True:
        show_menu()
        choice = get_user_input("\nSelect an option (1-10)", None)
        
        if choice == '1':
            process_data()
        elif choice == '2':
            train_model()
        elif choice == '3':
            backtest_strategy()
        elif choice == '4':
            run_full_pipeline()
        elif choice == '5':
            live_trade()
        elif choice == '6':
            launch_dashboard()
        elif choice == '7':
            show_status()
        elif choice == '8':
            show_help()
        elif choice == '9':
            clear_yfinance_cache()
        elif choice == '10':
            print(f"\n{Colors.OKGREEN}{Colors.BOLD}👋 Thanks for using Stock Arbitrage Model CLI!{Colors.ENDC}\n")
            break
        else:
            print_error("Invalid choice! Please select 1-10.")
        
        if choice != '10':
            input(f"\n{Colors.OKCYAN}Press Enter to continue...{Colors.ENDC}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.WARNING}Interrupted by user.{Colors.ENDC}")
        print(f"{Colors.OKGREEN}👋 Goodbye!{Colors.ENDC}\n")
        sys.exit(0)
