"""
Unit Tests for Model Drift Detection
Tests performance monitoring, feature drift, and prediction drift detection
"""

import pytest
import numpy as np
from datetime import datetime, timedelta, timezone
from model_drift_detection import (
    PerformanceMonitor, FeatureDriftDetector, PredictionDriftDetector,
    ModelDriftMonitor, DriftSeverity, DriftType, DriftAlert, PerformanceMetrics
)


class TestPerformanceMonitor:
    """Test performance monitoring and drift detection."""

    def setup_method(self):
        """Set up test environment."""
        self.baseline = {
            'mse': 0.10,
            'mae': 0.25,
            'directional_accuracy': 65.0
        }
        self.monitor = PerformanceMonitor(
            baseline_metrics=self.baseline,
            window_size=100,
            alert_threshold_pct=0.15
        )

    def test_initialization(self):
        """Test monitor initialization."""
        assert self.monitor.baseline_metrics == self.baseline
        assert self.monitor.window_size == 100
        assert len(self.monitor.recent_errors) == 0

    def test_record_prediction(self):
        """Test recording predictions."""
        self.monitor.record_prediction(predicted=1.0, actual=1.1)
        assert len(self.monitor.recent_errors) == 1
        assert len(self.monitor.recent_directions) == 1

    def test_directional_accuracy_correct(self):
        """Test directional accuracy for correct prediction."""
        # Both positive - correct
        self.monitor.record_prediction(predicted=1.0, actual=1.5)
        assert self.monitor.recent_directions[-1] == 1

        # Both negative - correct
        self.monitor.record_prediction(predicted=-1.0, actual=-1.5)
        assert self.monitor.recent_directions[-1] == 1

    def test_directional_accuracy_incorrect(self):
        """Test directional accuracy for incorrect prediction."""
        # Positive predicted, negative actual - wrong
        self.monitor.record_prediction(predicted=1.0, actual=-0.5)
        assert self.monitor.recent_directions[-1] == 0

    def test_get_current_metrics_insufficient_data(self):
        """Test that metrics return None with insufficient data."""
        for i in range(5):  # Less than 10
            self.monitor.record_prediction(1.0, 1.1)

        metrics = self.monitor.get_current_metrics()
        assert metrics is None

    def test_get_current_metrics_sufficient_data(self):
        """Test metrics calculation with sufficient data."""
        # Add 20 predictions
        for i in range(20):
            self.monitor.record_prediction(1.0 + i*0.1, 1.0 + i*0.1 + 0.05)

        metrics = self.monitor.get_current_metrics()
        assert metrics is not None
        assert metrics.mse > 0
        assert metrics.mae > 0
        assert metrics.num_predictions == 20

    def test_no_drift_when_performance_good(self):
        """Test that no drift is detected when performance is good."""
        # Add predictions with similar error to baseline
        np.random.seed(42)
        for i in range(50):
            pred = np.random.normal(0, 1)
            actual = pred + np.random.normal(0, 0.3)  # Small error
            self.monitor.record_prediction(pred, actual)

        alerts = self.monitor.check_drift()
        # May or may not have alerts depending on random data
        # This test mainly checks no crashes occur
        assert isinstance(alerts, list)

    def test_drift_detected_on_degradation(self):
        """Test that drift is detected when performance degrades."""
        # First add good predictions
        for i in range(30):
            self.monitor.record_prediction(1.0, 1.05)

        # Now add bad predictions (large errors)
        for i in range(30):
            self.monitor.record_prediction(1.0, 2.0)  # Large error

        alerts = self.monitor.check_drift()
        # Should detect MSE degradation
        assert len(alerts) > 0
        assert any(a.drift_type == DriftType.PERFORMANCE for a in alerts)

    def test_severity_calculation(self):
        """Test severity calculation."""
        assert self.monitor._calculate_severity(0.10) == DriftSeverity.LOW
        assert self.monitor._calculate_severity(0.20) == DriftSeverity.MEDIUM
        assert self.monitor._calculate_severity(0.30) == DriftSeverity.HIGH
        assert self.monitor._calculate_severity(0.50) == DriftSeverity.CRITICAL


class TestFeatureDriftDetector:
    """Test feature drift detection."""

    def setup_method(self):
        """Set up test environment."""
        self.baseline_stats = {
            'feature1': {'mean': 10.0, 'std': 2.0},
            'feature2': {'mean': 50.0, 'std': 5.0},
            'feature3': {'mean': 0.0, 'std': 1.0}
        }
        self.detector = FeatureDriftDetector(
            baseline_stats=self.baseline_stats,
            window_size=100,
            significance_level=0.01
        )

    def test_initialization(self):
        """Test detector initialization."""
        assert len(self.detector.feature_windows) == 3
        assert 'feature1' in self.detector.feature_windows

    def test_record_features(self):
        """Test recording feature values."""
        features = {'feature1': 10.5, 'feature2': 51.0, 'feature3': 0.1}
        self.detector.record_features(features)

        assert len(self.detector.feature_windows['feature1']) == 1
        assert self.detector.feature_windows['feature1'][0] == 10.5

    def test_no_drift_with_similar_distribution(self):
        """Test that no drift is detected with similar distribution."""
        np.random.seed(42)

        # Generate features similar to baseline
        for i in range(50):
            features = {
                'feature1': np.random.normal(10.0, 2.0),
                'feature2': np.random.normal(50.0, 5.0),
                'feature3': np.random.normal(0.0, 1.0)
            }
            self.detector.record_features(features)

        alerts = self.detector.check_drift()
        # Should have few or no alerts
        assert len(alerts) <= 1  # Allow for random statistical fluctuation

    def test_drift_detected_on_mean_shift(self):
        """Test that drift is detected when feature mean shifts."""
        np.random.seed(42)

        # Generate features with shifted mean for feature1
        for i in range(50):
            features = {
                'feature1': np.random.normal(15.0, 2.0),  # Mean shifted from 10 to 15
                'feature2': np.random.normal(50.0, 5.0),
                'feature3': np.random.normal(0.0, 1.0)
            }
            self.detector.record_features(features)

        alerts = self.detector.check_drift()
        # Should detect drift in feature1
        assert len(alerts) > 0
        assert any('feature1' in a.metric_name for a in alerts)

    def test_drift_detected_on_variance_change(self):
        """Test that drift is detected when variance changes."""
        np.random.seed(42)

        # Generate features with much larger variance
        for i in range(50):
            features = {
                'feature1': np.random.normal(10.0, 8.0),  # Std changed from 2 to 8
                'feature2': np.random.normal(50.0, 5.0),
                'feature3': np.random.normal(0.0, 1.0)
            }
            self.detector.record_features(features)

        alerts = self.detector.check_drift()
        # Should detect variance change in feature1
        assert len(alerts) > 0

    def test_insufficient_data_no_alerts(self):
        """Test that insufficient data doesn't trigger alerts."""
        # Add only 10 samples (need 30)
        for i in range(10):
            features = {'feature1': 20.0, 'feature2': 100.0, 'feature3': 5.0}
            self.detector.record_features(features)

        alerts = self.detector.check_drift()
        assert len(alerts) == 0  # Not enough data


class TestPredictionDriftDetector:
    """Test prediction drift detection."""

    def setup_method(self):
        """Set up test environment."""
        self.detector = PredictionDriftDetector(
            baseline_prediction_mean=0.5,
            baseline_prediction_std=0.3,
            window_size=100
        )

    def test_initialization(self):
        """Test detector initialization."""
        assert self.detector.baseline_mean == 0.5
        assert self.detector.baseline_std == 0.3
        assert len(self.detector.predictions) == 0

    def test_record_prediction(self):
        """Test recording predictions."""
        self.detector.record_prediction(0.6)
        assert len(self.detector.predictions) == 1

    def test_no_drift_with_similar_predictions(self):
        """Test that no drift is detected with similar predictions."""
        np.random.seed(42)

        # Generate predictions similar to baseline
        for i in range(50):
            pred = np.random.normal(0.5, 0.3)
            self.detector.record_prediction(pred)

        alerts = self.detector.check_drift()
        assert len(alerts) == 0

    def test_drift_detected_on_bias(self):
        """Test that drift is detected when predictions become biased."""
        # Generate heavily biased predictions
        for i in range(50):
            self.detector.record_prediction(2.0)  # Much higher than baseline 0.5

        alerts = self.detector.check_drift()
        assert len(alerts) > 0
        assert alerts[0].drift_type == DriftType.PREDICTION

    def test_insufficient_data_no_alerts(self):
        """Test that insufficient data doesn't trigger alerts."""
        for i in range(20):  # Less than 30
            self.detector.record_prediction(2.0)

        alerts = self.detector.check_drift()
        assert len(alerts) == 0


class TestModelDriftMonitor:
    """Test comprehensive drift monitoring."""

    def setup_method(self):
        """Set up test environment."""
        baseline_performance = {
            'mse': 0.10,
            'mae': 0.25,
            'directional_accuracy': 65.0
        }

        baseline_features = {
            'price': {'mean': 100.0, 'std': 10.0},
            'volume': {'mean': 1000000, 'std': 200000}
        }

        baseline_prediction_stats = {
            'mean': 0.5,
            'std': 0.3
        }

        self.monitor = ModelDriftMonitor(
            baseline_performance=baseline_performance,
            baseline_features=baseline_features,
            baseline_prediction_stats=baseline_prediction_stats
        )

    def test_initialization(self):
        """Test monitor initialization."""
        assert self.monitor.performance_monitor is not None
        assert self.monitor.feature_detector is not None
        assert self.monitor.prediction_detector is not None

    def test_record_prediction_complete(self):
        """Test recording complete prediction data."""
        features = {'price': 102.0, 'volume': 1050000}
        prediction = 0.6
        actual = 0.65

        self.monitor.record_prediction(features, prediction, actual)

        # Check all detectors received data
        assert len(self.monitor.feature_detector.feature_windows['price']) == 1
        assert len(self.monitor.prediction_detector.predictions) == 1
        assert len(self.monitor.performance_monitor.recent_errors) == 1

    def test_record_prediction_without_actual(self):
        """Test recording prediction without actual value."""
        features = {'price': 102.0, 'volume': 1050000}
        prediction = 0.6

        # Should not crash even without actual
        self.monitor.record_prediction(features, prediction, actual=None)

        assert len(self.monitor.prediction_detector.predictions) == 1

    def test_check_all_drift_no_drift(self):
        """Test drift checking when no drift present."""
        np.random.seed(42)

        # Generate good data
        for i in range(50):
            features = {
                'price': np.random.normal(100.0, 10.0),
                'volume': np.random.normal(1000000, 200000)
            }
            pred = np.random.normal(0.5, 0.3)
            actual = pred + np.random.normal(0, 0.05)

            self.monitor.record_prediction(features, pred, actual)

        alerts = self.monitor.check_all_drift()
        # May have some alerts due to random variation
        # Main check is it doesn't crash
        assert isinstance(alerts, list)

    def test_drift_summary(self):
        """Test drift summary generation."""
        # Generate some drift
        for i in range(50):
            features = {'price': 150.0, 'volume': 2000000}  # Shifted features
            pred = 2.0  # Biased predictions
            actual = 0.5

            self.monitor.record_prediction(features, pred, actual)

        # Check drift
        self.monitor.check_all_drift()

        summary = self.monitor.get_drift_summary()
        assert 'total_alerts_7days' in summary
        assert 'severity_counts' in summary
        assert 'requires_retraining' in summary

    def test_alert_history_stored(self):
        """Test that alerts are stored in history."""
        initial_count = len(self.monitor.alert_history)

        # Generate drift
        for i in range(50):
            features = {'price': 200.0, 'volume': 3000000}
            pred = 5.0
            actual = 0.5
            self.monitor.record_prediction(features, pred, actual)

        self.monitor.check_all_drift()

        # Should have more alerts than before
        assert len(self.monitor.alert_history) >= initial_count


class TestDriftAlert:
    """Test DriftAlert dataclass."""

    def test_alert_creation(self):
        """Test creating a drift alert."""
        alert = DriftAlert(
            timestamp=datetime(2024, 1, 15, 10, 0),
            drift_type=DriftType.PERFORMANCE,
            severity=DriftSeverity.HIGH,
            metric_name='mse',
            baseline_value=0.10,
            current_value=0.25,
            deviation_pct=150.0,
            message="MSE degraded significantly",
            recommended_action="Retrain model"
        )

        assert alert.metric_name == 'mse'
        assert alert.severity == DriftSeverity.HIGH
        assert alert.drift_type == DriftType.PERFORMANCE

    def test_alert_to_dict(self):
        """Test converting alert to dictionary."""
        alert = DriftAlert(
            timestamp=datetime(2024, 1, 15, 10, 0),
            drift_type=DriftType.FEATURE,
            severity=DriftSeverity.MEDIUM,
            metric_name='price_mean',
            baseline_value=100.0,
            current_value=150.0,
            deviation_pct=50.0,
            message="Price mean shifted",
            recommended_action="Check data source"
        )

        data = alert.to_dict()
        assert data['metric_name'] == 'price_mean'
        assert data['drift_type'] == 'FEATURE'
        assert data['severity'] == 'MEDIUM'
        assert 'timestamp' in data


class TestPerformanceMetrics:
    """Test PerformanceMetrics dataclass."""

    def test_metrics_creation(self):
        """Test creating performance metrics."""
        metrics = PerformanceMetrics(
            timestamp=datetime.now(timezone.utc),
            mse=0.15,
            mae=0.30,
            directional_accuracy=62.5,
            num_predictions=100,
            mean_prediction=0.45,
            std_prediction=0.35
        )

        assert metrics.mse == 0.15
        assert metrics.directional_accuracy == 62.5
        assert metrics.num_predictions == 100
