"""
Real-Time Stock Arbitrage Dashboard
Displays equity curve, positions, trades, and latest model signals
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import os
import time
import sys
import subprocess

# Try importing TensorFlow/Keras for model predictions
try:
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Stock Arbitrage Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .positive {
        color: #28a745;
    }
    .negative {
        color: #dc3545;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=60)  # Cache for 60 seconds
def load_backtest_results(filename='backtest_results.csv'):
    """Load backtest results from CSV."""
    try:
        if os.path.exists(filename):
            df = pd.read_csv(filename, parse_dates=['date'])
            return df
        return None
    except Exception as e:
        st.error(f"❌ Error loading backtest results: {e}")
        with st.expander("🔍 Troubleshooting"):
            st.markdown("""
            **Possible causes**:
            - File is corrupted or incomplete
            - CSV format doesn't match expected structure
            - File is being written by another process

            **How to fix**:
            1. Check file exists: `ls -lh backtest_results.csv`
            2. Verify it's not empty: `wc -l backtest_results.csv`
            3. Re-run backtest if needed: `python backtest_strategy.py`
            """)
        return None

def load_equity_curve_from_results(results_df):
    """Extract equity curve from backtest results."""
    if results_df is None or results_df.empty:
        return None
    
    # Calculate cumulative profit
    results_df['cumulative_profit'] = results_df['profit'].cumsum()
    
    # Assume initial capital of 10000 (can be made configurable)
    initial_capital = 10000
    equity_curve = initial_capital + results_df['cumulative_profit']
    
    equity_df = pd.DataFrame({
        'date': results_df['date'],
        'equity': equity_curve
    })
    
    return equity_df

def get_latest_signal(model_path='lstm_price_difference_model.h5',
                     data_file='processed_stock_data.csv',
                     threshold=0.5):
    """Get the latest trading signal from the model."""
    if not TF_AVAILABLE:
        return None, "TensorFlow not available"
    
    try:
        from live_trading import get_latest_signal
        signal = get_latest_signal(model_path, data_file, threshold=threshold)
        return signal, None
    except Exception as e:
        return None, str(e)

def format_currency(value):
    """Format value as currency."""
    return f"${value:,.2f}"

def format_percentage(value):
    """Format value as percentage."""
    return f"{value:.2f}%"

def main():
    """Main dashboard function."""
    
    # Header
    st.markdown('<h1 class="main-header">📈 Stock Arbitrage Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Auto-refresh option
        auto_refresh = st.checkbox("Auto-refresh (30s)", value=False)
        
        if auto_refresh:
            refresh_interval = st.slider("Refresh interval (seconds)", 10, 60, 30)
        
        # Manual refresh button
        if st.button("🔄 Refresh Now"):
            st.rerun()
        
        st.markdown("---")
        
        # File paths with live existence checks
        st.subheader("📁 Data Files")
        results_file = st.text_input("Backtest Results CSV", value="backtest_results.csv")
        st.caption(f"{'✅ Found' if os.path.exists(results_file) else '❌ Not found'}")

        model_file = st.text_input("Model File", value="lstm_price_difference_model.h5")
        st.caption(f"{'✅ Found' if os.path.exists(model_file) else '❌ Not found'}")

        data_file = st.text_input("Processed Data CSV", value="processed_stock_data.csv")
        st.caption(f"{'✅ Found' if os.path.exists(data_file) else '❌ Not found'}")

        # Signal threshold
        signal_threshold = st.slider("Signal Threshold", 0.1, 1.0, 0.5, 0.1)

        st.markdown("---")
        st.caption(f"Last refreshed: {datetime.now().strftime('%H:%M:%S')}")
        st.info("💡 Dashboard displays backtest results and latest model signals")
    
    # Main content
    col1, col2, col3, col4 = st.columns(4)

    # Load backtest results with loading spinner
    with st.spinner("📊 Loading backtest results..."):
        results_df = load_backtest_results(results_file)

    if results_df is None or results_df.empty:
        st.warning("⚠️ No backtest results found")
        st.info("👉 Run a backtest to generate data for this dashboard")
        
        st.markdown("### 🚀 Run Backtest")
        st.info("Run a backtest to generate data for the dashboard. This will process data, train the model, and generate backtest results.")
        
        col_run1, col_run2 = st.columns([1, 1])
        
        with col_run1:
            st.markdown("#### Quick Run (Use Default Settings)")
            if st.button("📊 Run Backtest with Defaults", type="primary", use_container_width=True):
                progress_text = st.empty()
                progress_bar = st.progress(0)

                try:
                    # Use Python 3.12 venv if available for TensorFlow support
                    python_cmd = sys.executable
                    venv_python = os.path.join(os.getcwd(), 'venv_py312', 'bin', 'python')
                    if os.path.exists(venv_python):
                        python_cmd = venv_python

                    progress_text.text("🚀 Starting backtest...")
                    progress_bar.progress(10)

                    progress_text.text("📊 Running backtest (this may take 5-10 minutes)...")
                    progress_bar.progress(30)

                    result = subprocess.run(
                        [python_cmd, 'backtest_strategy.py'],
                        capture_output=True,
                        text=True,
                        timeout=600  # 10 minute timeout
                    )

                    progress_bar.progress(90)

                    if result.returncode == 0:
                        progress_bar.progress(100)
                        st.success("✅ Backtest completed successfully!")
                        st.balloons()
                        st.info("📊 Dashboard will refresh in 3 seconds to show new results...")
                        import time
                        time.sleep(3)
                        st.rerun()  # Refresh to load new data
                    else:
                        st.error("❌ Backtest failed")
                        with st.expander("📋 Error Details"):
                            st.code(result.stderr if result.stderr else result.stdout)
                        with st.expander("🔍 Troubleshooting"):
                            st.markdown("""
                            **Common causes**:
                            - Missing dependencies (TensorFlow)
                            - Insufficient data
                            - Model file missing

                            **How to fix**:
                            1. Check prerequisites above
                            2. Run manually to see detailed errors: `python backtest_strategy.py`
                            3. Check logs in `logs/` directory
                            """)
                except subprocess.TimeoutExpired:
                    st.error("⏱️ Backtest timed out (>10 minutes)")
                    with st.expander("🔍 What to do"):
                        st.markdown("""
                        **Why this happens**:
                        - Large dataset (>1 year of data)
                        - Slow CPU
                        - Insufficient RAM

                        **Solutions**:
                        1. Run manually: `python backtest_strategy.py`
                        2. Reduce date range in data
                        3. Use a faster machine
                        """)
                except Exception as e:
                    st.error(f"❌ Unexpected error: {e}")
                    st.info("Try running manually: `python backtest_strategy.py`")
        
        with col_run2:
            st.markdown("#### Manual Run")
            st.info("To run manually, use:")
            st.code("python backtest_strategy.py", language="bash")
            st.markdown("Or use the CLI:")
            st.code("python main.py\n# Select option 3: Backtest Trading Strategy", language="bash")
        
        st.markdown("---")
        st.markdown("### 📋 Prerequisites")
        st.markdown("""
        Before running backtests, ensure you have:
        - ✅ Processed stock data (`processed_stock_data.csv`)
        - ✅ Trained model (`lstm_price_difference_model.h5`)
        - ✅ TensorFlow installed (requires Python 3.11 or 3.12)
        """)
        
        # Check prerequisites
        st.markdown("#### 🔍 Prerequisites Check")
        prereq_col1, prereq_col2 = st.columns(2)
        
        with prereq_col1:
            data_exists = os.path.exists(data_file)
            model_exists = os.path.exists(model_file)
            st.markdown(f"- Processed Data: {'✅' if data_exists else '❌'} `{data_file}`")
            st.markdown(f"- Trained Model: {'✅' if model_exists else '❌'} `{model_file}`")
        
        with prereq_col2:
            try:
                import tensorflow
                tf_available = True
            except ImportError:
                tf_available = False
            
            st.markdown(f"- TensorFlow: {'✅ Available' if tf_available else '❌ Not Available'}")
            if not tf_available:
                st.caption("Note: TensorFlow requires Python 3.11 or 3.12")
        
        return
    
    # ── Calculate all metrics once ──────────────────────────────────
    initial_capital = 10000
    total_profit = results_df['profit'].sum()
    final_capital = initial_capital + total_profit
    total_return_pct = (total_profit / initial_capital) * 100

    num_trades = len(results_df[results_df['signal'] != 'Hold'])
    winning_trades = len(results_df[results_df['profit'] > 0])
    losing_trades = len(results_df[results_df['profit'] < 0])
    win_rate = (winning_trades / num_trades * 100) if num_trades > 0 else 0
    avg_profit = total_profit / num_trades if num_trades > 0 else 0

    # ── Top row: four key numbers ────────────────────────────────────
    # delta must be numeric (or a formatted string starting with + / -)
    # for Streamlit to colour it automatically
    delta_str = f"+{format_percentage(total_return_pct)}" if total_return_pct >= 0 \
                else format_percentage(total_return_pct)

    with col1:
        st.metric("💰 Final Capital", format_currency(final_capital), delta=delta_str)
    with col2:
        st.metric("📈 Total Return", format_percentage(total_return_pct), delta=delta_str)
    with col3:
        st.metric("🎯 Win Rate", format_percentage(win_rate),
                  delta=f"{winning_trades}W / {losing_trades}L")
    with col4:
        st.metric("📊 Total Trades", str(num_trades),
                  delta=f"Avg {format_currency(avg_profit)}/trade")

    # ── Secondary metrics in a collapsible expander ─────────────────
    with st.expander("📋 Detailed Metrics", expanded=False):
        d1, d2, d3, d4 = st.columns(4)
        with d1:
            st.metric("Starting Capital", format_currency(initial_capital))
        with d2:
            st.metric("Total Profit / Loss", format_currency(total_profit))
        with d3:
            st.metric("Winning Trades", str(winning_trades))
        with d4:
            st.metric("Losing Trades", str(losing_trades))

    st.markdown("---")
    
    # ── Equity Curve with drawdown overlay ──────────────────────────
    st.subheader("📊 Equity Curve & Drawdown")

    equity_df = load_equity_curve_from_results(results_df)

    if equity_df is not None and not equity_df.empty:
        dates = equity_df['date']
        equity = equity_df['equity']

        running_max = equity.expanding().max()
        drawdown_pct = (equity - running_max) / running_max * 100
        max_drawdown = drawdown_pct.min()

        fig = go.Figure()

        # Equity line (primary y-axis)
        fig.add_trace(go.Scatter(
            x=dates, y=equity,
            name="Equity",
            line=dict(color="#1f77b4", width=2),
            yaxis="y"
        ))

        # Drawdown shading (secondary y-axis)
        fig.add_trace(go.Scatter(
            x=dates, y=drawdown_pct,
            name="Drawdown %",
            fill="tozeroy",
            fillcolor="rgba(220, 53, 69, 0.25)",
            line=dict(color="rgba(220, 53, 69, 0.6)", width=1),
            yaxis="y2"
        ))

        fig.update_layout(
            height=380,
            margin=dict(l=10, r=10, t=30, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=-0.15),
            hovermode="x unified",
            yaxis=dict(
                title_text="Equity ($)",
                tickprefix="$",
                showgrid=True,
                gridcolor="#eee"
            ),
            yaxis2=dict(
                title_text="Drawdown (%)",
                ticksuffix="%",
                overlaying="y",
                side="right",
                range=[min(drawdown_pct.min() * 1.5, -1), 1],
                showgrid=False
            ),
            xaxis=dict(showgrid=False)
        )

        st.plotly_chart(fig, use_container_width=True)

        # Summary stats below chart
        c1, c2, c3 = st.columns(3)
        c1.metric("Max Drawdown", format_percentage(max_drawdown))
        c2.metric("Current Equity", format_currency(equity.iloc[-1]))
        c3.metric("Peak Equity", format_currency(running_max.iloc[-1]))

    st.markdown("---")
    
    # Latest Signals and Positions
    col_signal, col_positions = st.columns(2)
    
    with col_signal:
        st.subheader("🎯 Latest Trading Signals")
        
        # Get latest signals from model
        if TF_AVAILABLE and os.path.exists(model_file) and os.path.exists(data_file):
            with st.spinner("🔮 Generating latest trading signal..."):
                latest_signal, error = get_latest_signal(model_file, data_file, signal_threshold)

                if error:
                    st.error(f"❌ Error generating signal: {error}")
                    with st.expander("🔍 Troubleshooting"):
                        st.markdown("""
                        **Common issues**:
                        - Model file is corrupted
                        - Data file format has changed
                        - Not enough recent data (need 60+ days)

                        **Quick fixes**:
                        1. Retrain model: `python train_model.py`
                        2. Update data: `./run_daily_data_update.sh`
                        3. Check model file exists and is >1MB
                        """)
                elif latest_signal:
                    # Display signal with color coding
                    if latest_signal == 'Buy AAPL, Sell MSFT':
                        st.success(f"🟢 **{latest_signal}**")
                    elif latest_signal == 'Buy MSFT, Sell AAPL':
                        st.info(f"🔵 **{latest_signal}**")
                    else:
                        st.warning(f"🟡 **{latest_signal}**")
                else:
                    st.info("No signal available")
        else:
            st.warning("Model not available. Displaying latest signal from backtest results.")
            if not results_df.empty:
                latest_backtest_signal = results_df['signal'].iloc[-1]
                st.info(f"Latest backtest signal: **{latest_backtest_signal}**")
        
        # Show recent signals from backtest
        st.markdown("#### Recent Signals (Last 10)")
        if not results_df.empty:
            recent_signals = results_df[['date', 'signal', 'predicted_diff', 'profit']].tail(10)
            recent_signals['profit'] = recent_signals['profit'].apply(format_currency)
            recent_signals['predicted_diff'] = recent_signals['predicted_diff'].apply(lambda x: f"{x:.4f}")
            recent_signals['date'] = recent_signals['date'].dt.strftime('%Y-%m-%d')
            st.dataframe(recent_signals, use_container_width=True, hide_index=True)
    
    with col_positions:
        st.subheader("💼 Recent Trades")
        
        if not results_df.empty:
            # Filter non-zero profit trades
            trades_df = results_df[results_df['profit'] != 0].copy()
            
            if not trades_df.empty:
                # Get last 10 trades
                recent_trades = trades_df.tail(10)[['date', 'signal', 'aapl_price', 'msft_price', 'profit']].copy()
                recent_trades['date'] = recent_trades['date'].dt.strftime('%Y-%m-%d')
                recent_trades['aapl_price'] = recent_trades['aapl_price'].apply(format_currency)
                recent_trades['msft_price'] = recent_trades['msft_price'].apply(format_currency)
                
                # Format profit with color
                def format_profit(value):
                    if value > 0:
                        return f"🟢 {format_currency(value)}"
                    elif value < 0:
                        return f"🔴 {format_currency(value)}"
                    else:
                        return format_currency(value)
                
                recent_trades['profit'] = recent_trades['profit'].apply(format_profit)
                
                st.dataframe(recent_trades, use_container_width=True, hide_index=True)
                
                # Show profit breakdown
                st.markdown("#### Profit Breakdown")
                profit_by_signal = trades_df.groupby('signal')['profit'].agg(['sum', 'count', 'mean'])
                profit_by_signal['sum'] = profit_by_signal['sum'].apply(format_currency)
                profit_by_signal['mean'] = profit_by_signal['mean'].apply(format_currency)
                profit_by_signal.columns = ['Total Profit', 'Trade Count', 'Avg Profit']
                st.dataframe(profit_by_signal, use_container_width=True)
            else:
                st.info("No trades executed yet.")
    
    st.markdown("---")
    
    # Detailed Trade History
    st.subheader("📋 Complete Trade History")
    
    # Filters
    col_filter1, col_filter2, col_filter3 = st.columns(3)
    
    with col_filter1:
        signal_filter = st.multiselect("Filter by Signal", 
                                      options=results_df['signal'].unique(),
                                      default=results_df['signal'].unique())
    
    with col_filter2:
        date_range = st.date_input("Date Range",
                                   value=(results_df['date'].min(), results_df['date'].max()),
                                   min_value=results_df['date'].min(),
                                   max_value=results_df['date'].max())
    
    with col_filter3:
        show_holds = st.checkbox("Show Hold signals", value=False)
    
    # Apply filters
    filtered_df = results_df[results_df['signal'].isin(signal_filter)].copy()
    
    if isinstance(date_range, tuple) and len(date_range) == 2:
        filtered_df = filtered_df[
            (filtered_df['date'] >= pd.Timestamp(date_range[0])) &
            (filtered_df['date'] <= pd.Timestamp(date_range[1]))
        ]
    
    if not show_holds:
        filtered_df = filtered_df[filtered_df['signal'] != 'Hold']
    
    # Display filtered results
    display_df = filtered_df[['date', 'signal', 'aapl_price', 'msft_price', 
                             'predicted_diff', 'actual_diff', 'profit', 'return_pct']].copy()
    
    # Format columns
    display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
    display_df['aapl_price'] = display_df['aapl_price'].apply(format_currency)
    display_df['msft_price'] = display_df['msft_price'].apply(format_currency)
    display_df['profit'] = display_df['profit'].apply(format_currency)
    display_df['return_pct'] = display_df['return_pct'].apply(format_percentage)
    display_df['predicted_diff'] = display_df['predicted_diff'].apply(lambda x: f"{x:.4f}")
    display_df['actual_diff'] = display_df['actual_diff'].apply(lambda x: f"{x:.2f}")
    
    st.dataframe(display_df, use_container_width=True, height=400)
    
    # Download button
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Filtered Results (CSV)",
        data=csv,
        file_name=f"filtered_backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
    
    # Auto-refresh logic
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()

if __name__ == "__main__":
    main()
