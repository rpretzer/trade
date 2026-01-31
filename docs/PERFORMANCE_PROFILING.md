# Performance Profiling Guide

Guide to using performance profiling tools to identify and resolve bottlenecks in the trading system.

---

## Quick Start

### Basic Timing

Use the `@profile_time` decorator to measure function execution time:

```python
from performance_profiling import profile_time

@profile_time(log=True)
def process_data():
    # Your code here
    pass

# Runs and prints: [PERF] process_data took 0.123s
process_data()
```

### Memory Profiling

Use `@profile_memory` to track memory usage:

```python
from performance_profiling import profile_memory

@profile_memory(log=True)
def load_large_dataset():
    data = pd.read_csv("large_file.csv")
    return data

# Prints: [PERF] load_large_dataset took 2.5s, peak memory: 150.25 MB
```

### Profile Code Blocks

Use the `profile_block` context manager for sections of code:

```python
from performance_profiling import profile_block

def workflow():
    with profile_block("data_loading", log=True):
        df = pd.read_csv("data.csv")

    with profile_block("feature_engineering", log=True):
        features = calculate_features(df)

    with profile_block("model_inference", log=True):
        predictions = model.predict(features)
```

---

## Complete Performance Analysis

### 1. Add Profiling to Critical Functions

Identify critical paths in your code and add decorators:

```python
from performance_profiling import profile_time

@profile_time()  # track=True by default
def fetch_stock_data(symbol):
    # ... download logic
    pass

@profile_time()
def calculate_indicators(df):
    # ... indicator calculation
    pass

@profile_time()
def generate_prediction(features):
    # ... model inference
    pass
```

### 2. Run Your Workflow

Execute your trading workflow normally. The profiling decorators will collect timing data automatically:

```python
# Your normal workflow
for symbol in ["AAPL", "MSFT", "GOOGL"]:
    data = fetch_stock_data(symbol)
    indicators = calculate_indicators(data)
    prediction = generate_prediction(indicators)
```

### 3. Analyze Performance

After running your workflow, analyze the collected metrics:

```python
from performance_profiling import run_performance_analysis

# Print complete performance analysis
run_performance_analysis(slow_threshold=1.0)
```

**Output**:
```
================================================================================
PERFORMANCE SUMMARY
================================================================================

Performance Report: fetch_stock_data
  Total Calls: 3
  Total Time: 4.500s
  Avg Time: 1.500s
  Min Time: 1.200s
  Max Time: 1.900s
  95th Percentile: 1.850s

Performance Report: calculate_indicators
  Total Calls: 3
  Total Time: 0.300s
  Avg Time: 0.100s
  Min Time: 0.095s
  Max Time: 0.110s
  95th Percentile: 0.108s

================================================================================
PERFORMANCE BOTTLENECKS DETECTED
================================================================================

1. fetch_stock_data
   Calls: 3
   Avg Time: 1.500s
   Total Time: 4.500s
   Issues:
   - Slow average: 1.500s (threshold: 1.0s)
   - High total time: 4.5s across 3 calls
```

### 4. Save Report to File

Save the analysis for later review:

```python
from performance_profiling import save_performance_report

save_performance_report("logs/performance_report.txt")
```

---

## Advanced Profiling

### Detailed cProfile Analysis

For deep analysis of complex functions:

```python
from performance_profiling import profile_cprofile

def expensive_workflow():
    # ... complex operations
    pass

# Profile with cProfile
with profile_cprofile("logs/cprofile_output.txt", top_n=30):
    expensive_workflow()
```

This generates a detailed profile showing:
- Time spent in each function call
- Number of calls to each function
- Cumulative time including subcalls

### Disable/Enable Tracking

Control when profiling happens:

```python
from performance_profiling import PerformanceTracker

tracker = PerformanceTracker.get_instance()

# Disable profiling during startup
tracker.disable()
initialize_system()

# Enable for the main workflow
tracker.enable()
run_trading_workflow()

# Generate report
tracker.print_summary()
```

### Manual Timing Measurements

Record custom timing events:

```python
from performance_profiling import PerformanceTracker, TimingResult
from datetime import datetime
import time

tracker = PerformanceTracker.get_instance()

start = time.perf_counter()
# ... your code
duration = time.perf_counter() - start

result = TimingResult(
    function_name="custom_operation",
    duration_seconds=duration,
    timestamp=datetime.now()
)
tracker.record(result)
```

---

## Bottleneck Detection

The `BottleneckDetector` automatically identifies performance issues:

```python
from performance_profiling import BottleneckDetector, PerformanceTracker

tracker = PerformanceTracker.get_instance()
detector = BottleneckDetector(slow_threshold_seconds=0.5)

# After running your workflow
detector.analyze(tracker)
detector.print_report()
```

**What it detects**:

1. **Slow Functions**: Functions with average time > threshold
2. **High Variance**: Functions with unpredictable performance (max >> avg)
3. **High Total Time**: Functions consuming significant cumulative time

---

## Real-World Example: Profiling Backtest

```python
from performance_profiling import profile_time, profile_block, run_performance_analysis

@profile_time()
def load_historical_data(symbols, start_date, end_date):
    with profile_block("download_data"):
        data = download_from_api(symbols, start_date, end_date)

    with profile_block("process_data"):
        processed = process_stock_data(data)

    return processed

@profile_time()
def train_model(data):
    with profile_block("create_sequences"):
        X, y = create_sequences(data)

    with profile_block("model_training"):
        model = build_model()
        model.fit(X, y, epochs=100)

    return model

@profile_time()
def run_backtest(model, data):
    with profile_block("generate_predictions"):
        predictions = model.predict(data)

    with profile_block("simulate_trades"):
        results = simulate_trading(predictions, data)

    return results

# Run the full workflow
if __name__ == "__main__":
    # Execute workflow
    data = load_historical_data(["AAPL", "MSFT"], "2023-01-01", "2024-01-01")
    model = train_model(data)
    results = run_backtest(model, data)

    # Analyze performance
    print("\n" + "="*80)
    print("BACKTEST PERFORMANCE ANALYSIS")
    print("="*80 + "\n")
    run_performance_analysis(slow_threshold=1.0)
```

**Expected Output**:
```
================================================================================
BACKTEST PERFORMANCE ANALYSIS
================================================================================

Performance Report: load_historical_data
  Total Calls: 1
  Total Time: 15.234s
  Avg Time: 15.234s

Performance Report: download_data
  Total Calls: 1
  Total Time: 12.100s

Performance Report: train_model
  Total Calls: 1
  Total Time: 45.678s

Performance Report: model_training
  Total Calls: 1
  Total Time: 42.500s

================================================================================
PERFORMANCE BOTTLENECKS DETECTED
================================================================================

1. model_training
   Issues:
   - Slow average: 42.500s (threshold: 1.0s)
   - High total time: 42.5s across 1 calls

2. download_data
   Issues:
   - Slow average: 12.100s (threshold: 1.0s)
```

**Action Items** from this analysis:
- **model_training** (42.5s): Consider reducing epochs or using early stopping
- **download_data** (12.1s): Cache data or use batch downloads

---

## Integration with Existing Code

### Backtest Strategy

Add profiling to `backtest_strategy.py`:

```python
from performance_profiling import profile_time, profile_block

@profile_time()
def backtest_strategy(data, initial_capital=10000):
    with profile_block("signal_generation"):
        signals = generate_trading_signals(data)

    with profile_block("trade_execution"):
        results = execute_trades(signals, initial_capital)

    with profile_block("risk_calculations"):
        metrics = calculate_risk_metrics(results)

    return results, metrics
```

### Live Trading

Add profiling to `live_trading.py`:

```python
from performance_profiling import profile_time

@profile_time(log=True)  # Log each execution
def get_latest_signal():
    data = fetch_latest_data()
    features = prepare_features(data)
    prediction = model.predict(features)
    return prediction

# At the end of your trading day
from performance_profiling import save_performance_report
save_performance_report(f"logs/trading_performance_{date.today()}.txt")
```

### Model Training

Profile training to optimize hyperparameters:

```python
from performance_profiling import profile_time, profile_memory

@profile_memory(log=True)
def create_sequences(data, timesteps=60):
    # This will show memory usage
    X, y = [], []
    for i in range(timesteps, len(data)):
        X.append(data[i-timesteps:i])
        y.append(data[i, -1])
    return np.array(X), np.array(y)

@profile_time(log=True)
def train_lstm_model(X, y, epochs=100):
    model = build_model()
    model.fit(X, y, epochs=epochs, batch_size=32)
    return model
```

---

## Performance Optimization Tips

Based on profiling results, common optimizations:

### 1. Slow Data Loading (>5s)

**Before**:
```python
def load_data():
    df = pd.read_csv("large_file.csv")
    return df
```

**After**:
```python
def load_data():
    # Use chunking for large files
    chunks = pd.read_csv("large_file.csv", chunksize=10000)
    df = pd.concat(chunks, ignore_index=True)
    return df
```

### 2. Repeated Calculations

**Before**:
```python
for symbol in symbols:
    data = download_data(symbol)  # Slow
    indicators = calculate_indicators(data)  # Slow
```

**After**:
```python
# Batch download
all_data = download_batch(symbols)  # Single slow operation

# Parallel processing
with ThreadPoolExecutor(max_workers=4) as executor:
    results = executor.map(calculate_indicators, all_data)
```

### 3. Inefficient DataFrame Operations

**Before**:
```python
for i in range(len(df)):
    df.loc[i, 'ma'] = df.loc[i-20:i, 'close'].mean()  # Very slow
```

**After**:
```python
df['ma'] = df['close'].rolling(window=20).mean()  # Vectorized
```

---

## Continuous Monitoring

Set up automated performance monitoring:

```python
# At the end of each trading day
import schedule
from performance_profiling import save_performance_report, PerformanceTracker

def daily_performance_report():
    tracker = PerformanceTracker.get_instance()

    # Save report
    filename = f"logs/daily_perf_{date.today().strftime('%Y%m%d')}.txt"
    save_performance_report(filename, tracker)

    # Clear for next day
    tracker.clear()

# Schedule daily
schedule.every().day.at("16:30").do(daily_performance_report)
```

---

## Troubleshooting

### Issue: No performance data collected

**Cause**: Profiling disabled or functions not decorated

**Solution**:
```python
tracker = PerformanceTracker.get_instance()
print(f"Tracker enabled: {tracker.enabled}")
print(f"Reports count: {len(tracker.get_all_reports())}")
```

### Issue: Memory profiling crashes

**Cause**: `tracemalloc` incompatible with some libraries

**Solution**: Use `@profile_time` instead of `@profile_memory`

### Issue: Performance overhead from profiling

**Cause**: Too much profiling in production

**Solution**: Disable in production, enable only for debugging:
```python
import os
if os.getenv("ENABLE_PROFILING") != "true":
    tracker = PerformanceTracker.get_instance()
    tracker.disable()
```

---

## API Reference

### Decorators

- `@profile_time(track=True, log=False)` - Time function execution
- `@profile_memory(track=True, log=False)` - Time + memory usage

### Context Managers

- `profile_block(name, track=True, log=True)` - Profile code block
- `profile_cprofile(output_file, top_n=20)` - Detailed cProfile

### Analysis Functions

- `run_performance_analysis(tracker, slow_threshold)` - Complete analysis
- `save_performance_report(output_file, tracker)` - Save to file

### Classes

- `PerformanceTracker` - Collect and aggregate measurements
- `BottleneckDetector` - Identify performance issues
- `TimingResult` - Individual timing measurement
- `PerformanceReport` - Aggregated statistics

---

## Related Documentation

- [Architecture](ARCHITECTURE.md) - System architecture
- [Development Guide](DEVELOPMENT.md) - Development practices
- [Deployment Guide](DEPLOYMENT.md) - Deployment procedures

---

**Last Updated**: 2026-01-31
