"""
Performance Profiling Utilities

Provides tools for measuring and analyzing performance of the trading system.
Includes timing, memory profiling, and bottleneck identification.
"""

import time
import functools
import logging
from typing import Callable, Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from contextlib import contextmanager
import tracemalloc
import cProfile
import pstats
import io
from pathlib import Path


logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class TimingResult:
    """Result of a timing measurement."""

    function_name: str
    duration_seconds: float
    timestamp: datetime
    args_summary: Optional[str] = None
    memory_delta_mb: Optional[float] = None


@dataclass
class PerformanceReport:
    """Aggregated performance statistics."""

    function_name: str
    total_calls: int = 0
    total_time_seconds: float = 0.0
    min_time_seconds: float = float('inf')
    max_time_seconds: float = 0.0
    avg_time_seconds: float = 0.0
    percentile_95_seconds: float = 0.0
    measurements: List[TimingResult] = field(default_factory=list)

    def add_measurement(self, result: TimingResult):
        """Add a timing measurement to the report."""
        self.measurements.append(result)
        self.total_calls += 1
        self.total_time_seconds += result.duration_seconds
        self.min_time_seconds = min(self.min_time_seconds, result.duration_seconds)
        self.max_time_seconds = max(self.max_time_seconds, result.duration_seconds)
        self.avg_time_seconds = self.total_time_seconds / self.total_calls

        # Calculate 95th percentile
        sorted_times = sorted([m.duration_seconds for m in self.measurements])
        idx = int(len(sorted_times) * 0.95)
        if idx < len(sorted_times):
            self.percentile_95_seconds = sorted_times[idx]

    def __str__(self) -> str:
        """Format report as string."""
        return (
            f"Performance Report: {self.function_name}\n"
            f"  Total Calls: {self.total_calls}\n"
            f"  Total Time: {self.total_time_seconds:.3f}s\n"
            f"  Avg Time: {self.avg_time_seconds:.3f}s\n"
            f"  Min Time: {self.min_time_seconds:.3f}s\n"
            f"  Max Time: {self.max_time_seconds:.3f}s\n"
            f"  95th Percentile: {self.percentile_95_seconds:.3f}s"
        )


# ============================================================================
# Performance Tracker
# ============================================================================


class PerformanceTracker:
    """
    Global performance tracker for collecting timing measurements.

    Usage:
        tracker = PerformanceTracker.get_instance()
        tracker.record(result)
        report = tracker.get_report("function_name")
    """

    _instance = None

    def __init__(self):
        self.reports: Dict[str, PerformanceReport] = {}
        self.enabled = True

    @classmethod
    def get_instance(cls) -> 'PerformanceTracker':
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def record(self, result: TimingResult):
        """Record a timing result."""
        if not self.enabled:
            return

        if result.function_name not in self.reports:
            self.reports[result.function_name] = PerformanceReport(
                function_name=result.function_name
            )

        self.reports[result.function_name].add_measurement(result)

    def get_report(self, function_name: str) -> Optional[PerformanceReport]:
        """Get performance report for a function."""
        return self.reports.get(function_name)

    def get_all_reports(self) -> List[PerformanceReport]:
        """Get all performance reports."""
        return list(self.reports.values())

    def clear(self):
        """Clear all measurements."""
        self.reports.clear()

    def disable(self):
        """Disable performance tracking."""
        self.enabled = False

    def enable(self):
        """Enable performance tracking."""
        self.enabled = True

    def print_summary(self):
        """Print summary of all performance reports."""
        print("\n" + "=" * 80)
        print("PERFORMANCE SUMMARY")
        print("=" * 80)

        # Sort by total time descending
        sorted_reports = sorted(
            self.reports.values(),
            key=lambda r: r.total_time_seconds,
            reverse=True
        )

        for report in sorted_reports:
            print(f"\n{report}")

        print("\n" + "=" * 80)


# ============================================================================
# Decorators
# ============================================================================


def profile_time(track: bool = True, log: bool = False):
    """
    Decorator to measure function execution time.

    Args:
        track: Whether to record measurements in global tracker
        log: Whether to log timing to logger

    Usage:
        @profile_time()
        def my_function():
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()

            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.perf_counter() - start_time

                # Create timing result
                timing_result = TimingResult(
                    function_name=func.__name__,
                    duration_seconds=duration,
                    timestamp=datetime.now(),
                    args_summary=f"args={len(args)}, kwargs={len(kwargs)}"
                )

                # Record to tracker
                if track:
                    tracker = PerformanceTracker.get_instance()
                    tracker.record(timing_result)

                # Log if requested
                if log:
                    logger.info(
                        f"[PERF] {func.__name__} took {duration:.3f}s"
                    )

        return wrapper
    return decorator


def profile_memory(track: bool = True, log: bool = False):
    """
    Decorator to measure function memory usage.

    Args:
        track: Whether to record measurements in global tracker
        log: Whether to log memory usage to logger

    Usage:
        @profile_memory()
        def my_function():
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tracemalloc.start()
            start_time = time.perf_counter()

            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.perf_counter() - start_time
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()

                memory_mb = peak / (1024 * 1024)

                # Create timing result with memory
                timing_result = TimingResult(
                    function_name=func.__name__,
                    duration_seconds=duration,
                    timestamp=datetime.now(),
                    memory_delta_mb=memory_mb
                )

                # Record to tracker
                if track:
                    tracker = PerformanceTracker.get_instance()
                    tracker.record(timing_result)

                # Log if requested
                if log:
                    logger.info(
                        f"[PERF] {func.__name__} took {duration:.3f}s, "
                        f"peak memory: {memory_mb:.2f} MB"
                    )

        return wrapper
    return decorator


# ============================================================================
# Context Managers
# ============================================================================


@contextmanager
def profile_block(name: str, track: bool = True, log: bool = True):
    """
    Context manager to profile a code block.

    Args:
        name: Name for this profiling block
        track: Whether to record in global tracker
        log: Whether to log timing

    Usage:
        with profile_block("data_loading"):
            df = pd.read_csv("data.csv")
    """
    start_time = time.perf_counter()

    try:
        yield
    finally:
        duration = time.perf_counter() - start_time

        # Create timing result
        timing_result = TimingResult(
            function_name=name,
            duration_seconds=duration,
            timestamp=datetime.now()
        )

        # Record to tracker
        if track:
            tracker = PerformanceTracker.get_instance()
            tracker.record(timing_result)

        # Log if requested
        if log:
            logger.info(f"[PERF] {name} took {duration:.3f}s")


@contextmanager
def profile_cprofile(output_file: Optional[str] = None, top_n: int = 20):
    """
    Context manager for detailed cProfile profiling.

    Args:
        output_file: Optional file to save profile stats
        top_n: Number of top functions to print

    Usage:
        with profile_cprofile("profile.txt", top_n=30):
            expensive_function()
    """
    profiler = cProfile.Profile()
    profiler.enable()

    try:
        yield profiler
    finally:
        profiler.disable()

        # Create stats
        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream)
        stats.strip_dirs()
        stats.sort_stats('cumulative')

        # Print top N
        print("\n" + "=" * 80)
        print(f"cProfile Results (Top {top_n})")
        print("=" * 80)
        stats.print_stats(top_n)
        print(stream.getvalue())

        # Save to file if requested
        if output_file:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                stats = pstats.Stats(profiler, stream=f)
                stats.strip_dirs()
                stats.sort_stats('cumulative')
                stats.print_stats()
            print(f"\nFull profile saved to: {output_file}")


# ============================================================================
# Bottleneck Detection
# ============================================================================


class BottleneckDetector:
    """
    Identifies performance bottlenecks in the system.

    Usage:
        detector = BottleneckDetector()
        detector.analyze(performance_tracker)
        bottlenecks = detector.get_bottlenecks()
    """

    def __init__(self, slow_threshold_seconds: float = 1.0):
        """
        Initialize bottleneck detector.

        Args:
            slow_threshold_seconds: Functions slower than this are flagged
        """
        self.slow_threshold = slow_threshold_seconds
        self.bottlenecks: List[Dict[str, Any]] = []

    def analyze(self, tracker: PerformanceTracker):
        """
        Analyze performance data to identify bottlenecks.

        Args:
            tracker: PerformanceTracker instance with measurements
        """
        self.bottlenecks.clear()

        for report in tracker.get_all_reports():
            issues = []

            # Check if function is slow on average
            if report.avg_time_seconds > self.slow_threshold:
                issues.append(
                    f"Slow average: {report.avg_time_seconds:.3f}s "
                    f"(threshold: {self.slow_threshold}s)"
                )

            # Check for high variance (unpredictable performance)
            if report.total_calls > 10:
                variance_ratio = report.max_time_seconds / report.avg_time_seconds
                if variance_ratio > 3.0:
                    issues.append(
                        f"High variance: max {report.max_time_seconds:.3f}s "
                        f"vs avg {report.avg_time_seconds:.3f}s "
                        f"(ratio: {variance_ratio:.1f}x)"
                    )

            # Check if function consumes significant total time
            if report.total_time_seconds > 10.0:
                issues.append(
                    f"High total time: {report.total_time_seconds:.1f}s "
                    f"across {report.total_calls} calls"
                )

            if issues:
                self.bottlenecks.append({
                    'function': report.function_name,
                    'issues': issues,
                    'avg_time': report.avg_time_seconds,
                    'total_time': report.total_time_seconds,
                    'calls': report.total_calls
                })

        # Sort by total time (most impactful first)
        self.bottlenecks.sort(key=lambda x: x['total_time'], reverse=True)

    def get_bottlenecks(self) -> List[Dict[str, Any]]:
        """Get list of identified bottlenecks."""
        return self.bottlenecks

    def print_report(self):
        """Print bottleneck analysis report."""
        if not self.bottlenecks:
            print("\n✅ No performance bottlenecks detected!")
            return

        print("\n" + "=" * 80)
        print("PERFORMANCE BOTTLENECKS DETECTED")
        print("=" * 80)

        for i, bottleneck in enumerate(self.bottlenecks, 1):
            print(f"\n{i}. {bottleneck['function']}")
            print(f"   Calls: {bottleneck['calls']}")
            print(f"   Avg Time: {bottleneck['avg_time']:.3f}s")
            print(f"   Total Time: {bottleneck['total_time']:.3f}s")
            print("   Issues:")
            for issue in bottleneck['issues']:
                print(f"   - {issue}")

        print("\n" + "=" * 80)


# ============================================================================
# Convenience Functions
# ============================================================================


def run_performance_analysis(
    tracker: Optional[PerformanceTracker] = None,
    slow_threshold: float = 1.0
):
    """
    Run complete performance analysis.

    Args:
        tracker: PerformanceTracker to analyze (default: global instance)
        slow_threshold: Threshold for flagging slow functions (seconds)
    """
    if tracker is None:
        tracker = PerformanceTracker.get_instance()

    # Print performance summary
    tracker.print_summary()

    # Detect bottlenecks
    detector = BottleneckDetector(slow_threshold_seconds=slow_threshold)
    detector.analyze(tracker)
    detector.print_report()


def save_performance_report(
    output_file: str,
    tracker: Optional[PerformanceTracker] = None
):
    """
    Save performance report to file.

    Args:
        output_file: Path to output file
        tracker: PerformanceTracker to analyze (default: global instance)
    """
    if tracker is None:
        tracker = PerformanceTracker.get_instance()

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("PERFORMANCE REPORT\n")
        f.write(f"Generated: {datetime.now()}\n")
        f.write("=" * 80 + "\n\n")

        # Write individual reports
        sorted_reports = sorted(
            tracker.get_all_reports(),
            key=lambda r: r.total_time_seconds,
            reverse=True
        )

        for report in sorted_reports:
            f.write(str(report) + "\n\n")

        # Write bottleneck analysis
        detector = BottleneckDetector()
        detector.analyze(tracker)

        f.write("\n" + "=" * 80 + "\n")
        f.write("BOTTLENECK ANALYSIS\n")
        f.write("=" * 80 + "\n\n")

        bottlenecks = detector.get_bottlenecks()
        if not bottlenecks:
            f.write("✅ No performance bottlenecks detected!\n")
        else:
            for i, bottleneck in enumerate(bottlenecks, 1):
                f.write(f"{i}. {bottleneck['function']}\n")
                f.write(f"   Calls: {bottleneck['calls']}\n")
                f.write(f"   Avg Time: {bottleneck['avg_time']:.3f}s\n")
                f.write(f"   Total Time: {bottleneck['total_time']:.3f}s\n")
                f.write("   Issues:\n")
                for issue in bottleneck['issues']:
                    f.write(f"   - {issue}\n")
                f.write("\n")

    print(f"Performance report saved to: {output_file}")


# ============================================================================
# Example Usage
# ============================================================================


if __name__ == "__main__":
    # Example 1: Using decorators
    @profile_time(log=True)
    def slow_function():
        time.sleep(0.1)
        return "done"

    @profile_memory(log=True)
    def memory_intensive():
        data = [i for i in range(1000000)]
        return sum(data)

    # Example 2: Using context managers
    def example_workflow():
        with profile_block("data_loading"):
            time.sleep(0.05)

        with profile_block("processing"):
            time.sleep(0.1)

        with profile_block("saving"):
            time.sleep(0.02)

    # Run examples
    print("Running performance profiling examples...\n")

    for _ in range(5):
        slow_function()
        memory_intensive()
        example_workflow()

    # Print analysis
    run_performance_analysis(slow_threshold=0.05)
