"""
Tests for Performance Profiling

Tests timing decorators, memory profiling, context managers, and bottleneck detection.
"""

import pytest
import time
from datetime import datetime
from performance_profiling import (
    TimingResult,
    PerformanceReport,
    PerformanceTracker,
    BottleneckDetector,
    profile_time,
    profile_memory,
    profile_block,
    profile_cprofile,
    run_performance_analysis,
    save_performance_report
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def fresh_tracker():
    """Provide a fresh PerformanceTracker for each test."""
    tracker = PerformanceTracker.get_instance()
    tracker.clear()
    yield tracker
    tracker.clear()


@pytest.fixture
def sample_timing_results():
    """Provide sample timing results for testing."""
    return [
        TimingResult(
            function_name="fast_function",
            duration_seconds=0.01,
            timestamp=datetime.now()
        ),
        TimingResult(
            function_name="slow_function",
            duration_seconds=2.0,
            timestamp=datetime.now()
        ),
        TimingResult(
            function_name="slow_function",
            duration_seconds=2.5,
            timestamp=datetime.now()
        ),
    ]


# ============================================================================
# TimingResult Tests
# ============================================================================


def test_timing_result_creation():
    """Test TimingResult creation."""
    result = TimingResult(
        function_name="test_func",
        duration_seconds=1.5,
        timestamp=datetime.now(),
        args_summary="args=2, kwargs=1",
        memory_delta_mb=10.5
    )

    assert result.function_name == "test_func"
    assert result.duration_seconds == 1.5
    assert result.args_summary == "args=2, kwargs=1"
    assert result.memory_delta_mb == 10.5


def test_timing_result_minimal():
    """Test TimingResult with minimal fields."""
    result = TimingResult(
        function_name="test_func",
        duration_seconds=0.5,
        timestamp=datetime.now()
    )

    assert result.function_name == "test_func"
    assert result.duration_seconds == 0.5
    assert result.args_summary is None
    assert result.memory_delta_mb is None


# ============================================================================
# PerformanceReport Tests
# ============================================================================


def test_performance_report_add_measurement():
    """Test adding measurements to PerformanceReport."""
    report = PerformanceReport(function_name="test_func")

    result1 = TimingResult("test_func", 1.0, datetime.now())
    result2 = TimingResult("test_func", 2.0, datetime.now())

    report.add_measurement(result1)
    assert report.total_calls == 1
    assert report.total_time_seconds == 1.0
    assert report.avg_time_seconds == 1.0

    report.add_measurement(result2)
    assert report.total_calls == 2
    assert report.total_time_seconds == 3.0
    assert report.avg_time_seconds == 1.5


def test_performance_report_statistics():
    """Test statistical calculations in PerformanceReport."""
    report = PerformanceReport(function_name="test_func")

    # Add measurements with known distribution
    for duration in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]:
        result = TimingResult("test_func", duration, datetime.now())
        report.add_measurement(result)

    assert report.total_calls == 10
    assert report.min_time_seconds == 0.5
    assert report.max_time_seconds == 5.0
    assert report.avg_time_seconds == 2.75
    # 95th percentile of [0.5, 1.0, ..., 5.0] should be around 4.5-5.0
    assert report.percentile_95_seconds >= 4.0


def test_performance_report_str():
    """Test string representation of PerformanceReport."""
    report = PerformanceReport(function_name="test_func")
    result = TimingResult("test_func", 1.0, datetime.now())
    report.add_measurement(result)

    report_str = str(report)
    assert "test_func" in report_str
    assert "Total Calls: 1" in report_str
    assert "1.000s" in report_str


# ============================================================================
# PerformanceTracker Tests
# ============================================================================


def test_tracker_singleton():
    """Test PerformanceTracker singleton pattern."""
    tracker1 = PerformanceTracker.get_instance()
    tracker2 = PerformanceTracker.get_instance()
    assert tracker1 is tracker2


def test_tracker_record(fresh_tracker):
    """Test recording measurements in tracker."""
    result = TimingResult("test_func", 1.0, datetime.now())
    fresh_tracker.record(result)

    report = fresh_tracker.get_report("test_func")
    assert report is not None
    assert report.total_calls == 1
    assert report.total_time_seconds == 1.0


def test_tracker_multiple_functions(fresh_tracker):
    """Test tracking multiple different functions."""
    result1 = TimingResult("func1", 1.0, datetime.now())
    result2 = TimingResult("func2", 2.0, datetime.now())

    fresh_tracker.record(result1)
    fresh_tracker.record(result2)

    reports = fresh_tracker.get_all_reports()
    assert len(reports) == 2

    func1_report = fresh_tracker.get_report("func1")
    func2_report = fresh_tracker.get_report("func2")

    assert func1_report.total_time_seconds == 1.0
    assert func2_report.total_time_seconds == 2.0


def test_tracker_clear(fresh_tracker):
    """Test clearing tracker measurements."""
    result = TimingResult("test_func", 1.0, datetime.now())
    fresh_tracker.record(result)

    assert len(fresh_tracker.get_all_reports()) == 1

    fresh_tracker.clear()
    assert len(fresh_tracker.get_all_reports()) == 0


def test_tracker_disable_enable(fresh_tracker):
    """Test disabling and enabling tracker."""
    fresh_tracker.enable()
    result1 = TimingResult("test_func", 1.0, datetime.now())
    fresh_tracker.record(result1)

    fresh_tracker.disable()
    result2 = TimingResult("test_func", 2.0, datetime.now())
    fresh_tracker.record(result2)

    # Only first measurement should be recorded
    report = fresh_tracker.get_report("test_func")
    assert report.total_calls == 1

    fresh_tracker.enable()
    result3 = TimingResult("test_func", 3.0, datetime.now())
    fresh_tracker.record(result3)

    # Now should have 2 measurements
    report = fresh_tracker.get_report("test_func")
    assert report.total_calls == 2


# ============================================================================
# Decorator Tests
# ============================================================================


def test_profile_time_decorator():
    """Test @profile_time decorator."""
    tracker = PerformanceTracker.get_instance()
    tracker.clear()

    @profile_time(track=True, log=False)
    def test_func():
        time.sleep(0.01)
        return "done"

    result = test_func()
    assert result == "done"

    # Check that timing was recorded
    report = tracker.get_report("test_func")
    assert report is not None
    assert report.total_calls >= 1
    assert report.avg_time_seconds >= 0.01


def test_profile_time_no_tracking():
    """Test @profile_time with tracking disabled."""
    tracker = PerformanceTracker.get_instance()
    tracker.clear()

    @profile_time(track=False, log=False)
    def test_func():
        return "done"

    test_func()

    # Should not be tracked
    report = tracker.get_report("test_func")
    # Note: May still exist if other tests ran it
    # Just ensure it's not incremented
    initial_count = report.total_calls if report else 0

    test_func()
    report = tracker.get_report("test_func")
    final_count = report.total_calls if report else 0

    # Count should not increase
    assert final_count == initial_count


def test_profile_memory_decorator():
    """Test @profile_memory decorator."""
    tracker = PerformanceTracker.get_instance()
    tracker.clear()

    @profile_memory(track=True, log=False)
    def test_func():
        # Allocate some memory
        data = [i for i in range(10000)]
        return sum(data)

    result = test_func()
    assert result == sum(range(10000))

    # Check that memory was tracked
    report = tracker.get_report("test_func")
    assert report is not None
    assert len(report.measurements) >= 1
    # At least one measurement should have memory data
    assert any(m.memory_delta_mb is not None for m in report.measurements)


# ============================================================================
# Context Manager Tests
# ============================================================================


def test_profile_block_context():
    """Test profile_block context manager."""
    tracker = PerformanceTracker.get_instance()
    tracker.clear()

    with profile_block("test_block", track=True, log=False):
        time.sleep(0.01)

    report = tracker.get_report("test_block")
    assert report is not None
    assert report.total_calls >= 1
    assert report.avg_time_seconds >= 0.01


def test_profile_block_exception_handling():
    """Test profile_block handles exceptions properly."""
    tracker = PerformanceTracker.get_instance()
    tracker.clear()

    try:
        with profile_block("error_block", track=True, log=False):
            raise ValueError("Test error")
    except ValueError:
        pass

    # Should still record timing even if exception occurred
    report = tracker.get_report("error_block")
    assert report is not None
    assert report.total_calls >= 1


def test_profile_cprofile_context(tmp_path):
    """Test profile_cprofile context manager."""
    output_file = tmp_path / "profile.txt"

    def expensive_function():
        total = 0
        for i in range(100000):
            total += i
        return total

    with profile_cprofile(str(output_file), top_n=5):
        expensive_function()

    # Check that profile file was created
    assert output_file.exists()
    content = output_file.read_text()
    assert len(content) > 0


# ============================================================================
# BottleneckDetector Tests
# ============================================================================


def test_bottleneck_detector_slow_function(fresh_tracker, sample_timing_results):
    """Test bottleneck detection for slow functions."""
    # Add results to tracker
    for result in sample_timing_results:
        fresh_tracker.record(result)

    # Detect bottlenecks with 1.0s threshold
    detector = BottleneckDetector(slow_threshold_seconds=1.0)
    detector.analyze(fresh_tracker)

    bottlenecks = detector.get_bottlenecks()
    assert len(bottlenecks) >= 1

    # slow_function should be detected
    slow_func_bottleneck = next(
        (b for b in bottlenecks if b['function'] == 'slow_function'),
        None
    )
    assert slow_func_bottleneck is not None
    assert slow_func_bottleneck['avg_time'] > 1.0


def test_bottleneck_detector_high_variance(fresh_tracker):
    """Test bottleneck detection for high variance."""
    # Add measurements with high variance (1s to 10s)
    for duration in [1.0] * 10 + [10.0] * 2:
        result = TimingResult("variable_func", duration, datetime.now())
        fresh_tracker.record(result)

    detector = BottleneckDetector(slow_threshold_seconds=0.5)
    detector.analyze(fresh_tracker)

    bottlenecks = detector.get_bottlenecks()

    # Should detect high variance
    variable_bottleneck = next(
        (b for b in bottlenecks if b['function'] == 'variable_func'),
        None
    )
    assert variable_bottleneck is not None
    assert any("variance" in issue.lower() for issue in variable_bottleneck['issues'])


def test_bottleneck_detector_no_bottlenecks(fresh_tracker):
    """Test bottleneck detector with fast functions."""
    # Add only fast measurements
    for _ in range(10):
        result = TimingResult("fast_func", 0.01, datetime.now())
        fresh_tracker.record(result)

    detector = BottleneckDetector(slow_threshold_seconds=1.0)
    detector.analyze(fresh_tracker)

    bottlenecks = detector.get_bottlenecks()
    assert len(bottlenecks) == 0


# ============================================================================
# Integration Tests
# ============================================================================


def test_run_performance_analysis(fresh_tracker, sample_timing_results, capsys):
    """Test complete performance analysis workflow."""
    for result in sample_timing_results:
        fresh_tracker.record(result)

    run_performance_analysis(tracker=fresh_tracker, slow_threshold=1.0)

    captured = capsys.readouterr()
    assert "PERFORMANCE SUMMARY" in captured.out
    assert "BOTTLENECKS" in captured.out or "No performance bottlenecks" in captured.out


def test_save_performance_report(fresh_tracker, sample_timing_results, tmp_path):
    """Test saving performance report to file."""
    for result in sample_timing_results:
        fresh_tracker.record(result)

    output_file = tmp_path / "performance_report.txt"
    save_performance_report(str(output_file), tracker=fresh_tracker)

    assert output_file.exists()
    content = output_file.read_text()
    assert "PERFORMANCE REPORT" in content
    assert "BOTTLENECK ANALYSIS" in content
    assert "fast_function" in content or "slow_function" in content


def test_real_world_workflow():
    """Test performance profiling in realistic workflow."""
    tracker = PerformanceTracker.get_instance()
    tracker.clear()

    @profile_time(track=True, log=False)
    def load_data():
        time.sleep(0.01)
        return [1, 2, 3, 4, 5]

    @profile_time(track=True, log=False)
    def process_data(data):
        time.sleep(0.02)
        return [x * 2 for x in data]

    @profile_time(track=True, log=False)
    def save_results(results):
        time.sleep(0.01)
        return len(results)

    # Run workflow multiple times
    for _ in range(5):
        data = load_data()
        processed = process_data(data)
        save_results(processed)

    # Analyze performance
    reports = tracker.get_all_reports()
    assert len(reports) == 3

    # Each function should have been called 5 times
    for report in reports:
        assert report.total_calls == 5

    # process_data should be slowest
    process_report = tracker.get_report("process_data")
    load_report = tracker.get_report("load_data")
    assert process_report.avg_time_seconds > load_report.avg_time_seconds


# ============================================================================
# Edge Cases
# ============================================================================


def test_empty_tracker_analysis(fresh_tracker):
    """Test analysis with empty tracker."""
    detector = BottleneckDetector()
    detector.analyze(fresh_tracker)

    bottlenecks = detector.get_bottlenecks()
    assert len(bottlenecks) == 0


def test_single_measurement_statistics():
    """Test statistics with only one measurement."""
    report = PerformanceReport(function_name="single")
    result = TimingResult("single", 1.5, datetime.now())
    report.add_measurement(result)

    assert report.total_calls == 1
    assert report.avg_time_seconds == 1.5
    assert report.min_time_seconds == 1.5
    assert report.max_time_seconds == 1.5
    assert report.percentile_95_seconds == 1.5


def test_nested_profiling():
    """Test nested profile_block calls."""
    tracker = PerformanceTracker.get_instance()
    tracker.clear()

    with profile_block("outer", track=True, log=False):
        time.sleep(0.01)
        with profile_block("inner", track=True, log=False):
            time.sleep(0.01)

    outer_report = tracker.get_report("outer")
    inner_report = tracker.get_report("inner")

    assert outer_report is not None
    assert inner_report is not None
    # Outer should take longer than inner
    assert outer_report.avg_time_seconds >= inner_report.avg_time_seconds
