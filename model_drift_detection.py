"""
Model Drift Detection Module
Monitors ML model performance degradation and feature distribution shifts
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
from scipy import stats

logger = logging.getLogger(__name__)


class DriftSeverity(Enum):
    """Severity level of detected drift."""
    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class DriftType(Enum):
    """Type of drift detected."""
    PERFORMANCE = "PERFORMANCE"  # Model accuracy degrading
    FEATURE = "FEATURE"  # Input feature distribution changed
    PREDICTION = "PREDICTION"  # Output distribution changed
    CONCEPT = "CONCEPT"  # Relationship between features and target changed


@dataclass
class DriftAlert:
    """Alert for detected model drift."""
    timestamp: datetime
    drift_type: DriftType
    severity: DriftSeverity
    metric_name: str
    baseline_value: float
    current_value: float
    deviation_pct: float
    message: str
    recommended_action: str

    def to_dict(self) -> Dict:
        """Convert to dictionary for logging."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'drift_type': self.drift_type.value,
            'severity': self.severity.value,
            'metric_name': self.metric_name,
            'baseline_value': self.baseline_value,
            'current_value': self.current_value,
            'deviation_pct': self.deviation_pct,
            'message': self.message,
            'recommended_action': self.recommended_action
        }


@dataclass
class PerformanceMetrics:
    """Model performance metrics snapshot."""
    timestamp: datetime
    mse: float
    mae: float
    directional_accuracy: float  # % predictions with correct direction
    num_predictions: int

    # Additional context
    mean_prediction: float = 0.0
    std_prediction: float = 0.0


class PerformanceMonitor:
    """
    Monitors model prediction accuracy over time.

    Detects when performance degrades below baseline.
    """

    def __init__(
        self,
        baseline_metrics: Dict[str, float],
        window_size: int = 100,
        alert_threshold_pct: float = 0.15  # 15% degradation triggers alert
    ):
        """
        Initialize performance monitor.

        Args:
            baseline_metrics: Baseline metrics from validation (mse, mae, accuracy)
            window_size: Number of recent predictions to track
            alert_threshold_pct: % degradation that triggers alert
        """
        self.baseline_metrics = baseline_metrics
        self.window_size = window_size
        self.alert_threshold_pct = alert_threshold_pct

        # Rolling windows for metrics
        self.recent_errors = deque(maxlen=window_size)
        self.recent_directions = deque(maxlen=window_size)  # 1 if correct, 0 if wrong
        self.history: List[PerformanceMetrics] = []

    def record_prediction(
        self,
        predicted: float,
        actual: float,
        timestamp: Optional[datetime] = None
    ):
        """
        Record a prediction and its actual outcome.

        Args:
            predicted: Predicted value
            actual: Actual value
            timestamp: Prediction timestamp
        """
        if timestamp is None:
            timestamp = datetime.utcnow()

        error = abs(predicted - actual)
        self.recent_errors.append(error)

        # Check directional accuracy (for price difference prediction)
        # Correct if signs match
        direction_correct = (predicted * actual) >= 0
        self.recent_directions.append(1 if direction_correct else 0)

    def get_current_metrics(self) -> Optional[PerformanceMetrics]:
        """
        Calculate current performance metrics.

        Returns:
            Current metrics or None if insufficient data
        """
        if len(self.recent_errors) < 10:  # Need at least 10 predictions
            return None

        errors = np.array(self.recent_errors)

        return PerformanceMetrics(
            timestamp=datetime.utcnow(),
            mse=np.mean(errors ** 2),
            mae=np.mean(errors),
            directional_accuracy=np.mean(self.recent_directions) * 100,
            num_predictions=len(self.recent_errors)
        )

    def check_drift(self) -> List[DriftAlert]:
        """
        Check for performance drift.

        Returns:
            List of drift alerts (empty if no drift)
        """
        current = self.get_current_metrics()
        if current is None:
            return []

        alerts = []

        # Check MSE degradation
        if 'mse' in self.baseline_metrics:
            baseline_mse = self.baseline_metrics['mse']
            degradation = (current.mse - baseline_mse) / baseline_mse

            if degradation > self.alert_threshold_pct:
                severity = self._calculate_severity(degradation)

                alerts.append(DriftAlert(
                    timestamp=current.timestamp,
                    drift_type=DriftType.PERFORMANCE,
                    severity=severity,
                    metric_name='mse',
                    baseline_value=baseline_mse,
                    current_value=current.mse,
                    deviation_pct=degradation * 100,
                    message=f"MSE degraded by {degradation*100:.1f}%",
                    recommended_action=self._get_recommendation(severity)
                ))

        # Check directional accuracy
        if 'directional_accuracy' in self.baseline_metrics:
            baseline_acc = self.baseline_metrics['directional_accuracy']
            degradation = (baseline_acc - current.directional_accuracy) / baseline_acc

            if degradation > self.alert_threshold_pct:
                severity = self._calculate_severity(degradation)

                alerts.append(DriftAlert(
                    timestamp=current.timestamp,
                    drift_type=DriftType.PERFORMANCE,
                    severity=severity,
                    metric_name='directional_accuracy',
                    baseline_value=baseline_acc,
                    current_value=current.directional_accuracy,
                    deviation_pct=-degradation * 100,  # Negative = worse
                    message=f"Directional accuracy dropped by {degradation*100:.1f}%",
                    recommended_action=self._get_recommendation(severity)
                ))

        # Store snapshot
        if current:
            self.history.append(current)

        return alerts

    def _calculate_severity(self, degradation: float) -> DriftSeverity:
        """Calculate drift severity based on degradation."""
        if degradation < 0.15:
            return DriftSeverity.LOW
        elif degradation < 0.25:
            return DriftSeverity.MEDIUM
        elif degradation < 0.40:
            return DriftSeverity.HIGH
        else:
            return DriftSeverity.CRITICAL

    def _get_recommendation(self, severity: DriftSeverity) -> str:
        """Get recommended action based on severity."""
        if severity == DriftSeverity.CRITICAL:
            return "IMMEDIATE: Stop trading, retrain model with recent data"
        elif severity == DriftSeverity.HIGH:
            return "URGENT: Schedule model retraining within 24 hours"
        elif severity == DriftSeverity.MEDIUM:
            return "Schedule model retraining within 1 week"
        else:
            return "Monitor closely, consider retraining if trend continues"


class FeatureDriftDetector:
    """
    Detects distribution shifts in input features.

    Uses statistical tests to compare current feature distributions
    to baseline distributions from training/validation.
    """

    def __init__(
        self,
        baseline_stats: Dict[str, Dict[str, float]],
        window_size: int = 100,
        significance_level: float = 0.01  # p-value threshold
    ):
        """
        Initialize feature drift detector.

        Args:
            baseline_stats: Baseline feature statistics (mean, std, min, max)
            window_size: Number of recent samples to track
            significance_level: p-value threshold for statistical tests
        """
        self.baseline_stats = baseline_stats
        self.window_size = window_size
        self.significance_level = significance_level

        # Rolling windows for each feature
        self.feature_windows: Dict[str, deque] = {
            feature: deque(maxlen=window_size)
            for feature in baseline_stats.keys()
        }

    def record_features(self, features: Dict[str, float]):
        """
        Record feature values for drift detection.

        Args:
            features: Dictionary of feature name -> value
        """
        for feature, value in features.items():
            if feature in self.feature_windows:
                self.feature_windows[feature].append(value)

    def check_drift(self) -> List[DriftAlert]:
        """
        Check for feature distribution drift using statistical tests.

        Returns:
            List of drift alerts
        """
        alerts = []

        for feature, values in self.feature_windows.items():
            if len(values) < 30:  # Need enough samples for statistical test
                continue

            if feature not in self.baseline_stats:
                continue

            baseline = self.baseline_stats[feature]
            current_values = np.array(values)

            # Calculate current statistics
            current_mean = np.mean(current_values)
            current_std = np.std(current_values)

            # Test 1: Mean shift (z-test)
            baseline_mean = baseline.get('mean', 0)
            baseline_std = baseline.get('std', 1)

            if baseline_std > 0:
                z_score = abs(current_mean - baseline_mean) / (baseline_std / np.sqrt(len(values)))

                # Two-tailed test
                p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

                if p_value < self.significance_level:
                    deviation_pct = ((current_mean - baseline_mean) / baseline_mean) * 100 if baseline_mean != 0 else 0

                    severity = self._calculate_severity_from_pvalue(p_value)

                    alerts.append(DriftAlert(
                        timestamp=datetime.utcnow(),
                        drift_type=DriftType.FEATURE,
                        severity=severity,
                        metric_name=f"{feature}_mean",
                        baseline_value=baseline_mean,
                        current_value=current_mean,
                        deviation_pct=deviation_pct,
                        message=f"Feature '{feature}' mean shifted significantly (p={p_value:.4f})",
                        recommended_action="Investigate data pipeline, check for upstream changes"
                    ))

            # Test 2: Variance change (F-test/Levene's test)
            if baseline_std > 0:
                variance_ratio = current_std / baseline_std

                # Significant if variance changed by > 50%
                if variance_ratio > 1.5 or variance_ratio < 0.67:
                    alerts.append(DriftAlert(
                        timestamp=datetime.utcnow(),
                        drift_type=DriftType.FEATURE,
                        severity=DriftSeverity.MEDIUM,
                        metric_name=f"{feature}_std",
                        baseline_value=baseline_std,
                        current_value=current_std,
                        deviation_pct=((current_std - baseline_std) / baseline_std) * 100,
                        message=f"Feature '{feature}' variance changed significantly",
                        recommended_action="Check for data quality issues or market regime change"
                    ))

        return alerts

    def _calculate_severity_from_pvalue(self, p_value: float) -> DriftSeverity:
        """Calculate severity based on p-value."""
        if p_value < 0.0001:
            return DriftSeverity.CRITICAL
        elif p_value < 0.001:
            return DriftSeverity.HIGH
        elif p_value < 0.01:
            return DriftSeverity.MEDIUM
        else:
            return DriftSeverity.LOW


class PredictionDriftDetector:
    """
    Detects shifts in model output distribution.

    Monitors if model predictions are becoming extreme or biased.
    """

    def __init__(
        self,
        baseline_prediction_mean: float,
        baseline_prediction_std: float,
        window_size: int = 100
    ):
        """
        Initialize prediction drift detector.

        Args:
            baseline_prediction_mean: Expected mean of predictions
            baseline_prediction_std: Expected std of predictions
            window_size: Number of recent predictions to track
        """
        self.baseline_mean = baseline_prediction_mean
        self.baseline_std = baseline_prediction_std
        self.window_size = window_size
        self.predictions = deque(maxlen=window_size)

    def record_prediction(self, prediction: float):
        """Record a model prediction."""
        self.predictions.append(prediction)

    def check_drift(self) -> List[DriftAlert]:
        """
        Check for prediction distribution drift.

        Returns:
            List of drift alerts
        """
        if len(self.predictions) < 30:
            return []

        alerts = []
        preds = np.array(self.predictions)

        current_mean = np.mean(preds)
        current_std = np.std(preds)

        # Check for bias (mean shift)
        mean_shift = abs(current_mean - self.baseline_mean)
        if self.baseline_std > 0:
            z_score = mean_shift / (self.baseline_std / np.sqrt(len(preds)))

            if z_score > 3:  # 3 sigma event
                alerts.append(DriftAlert(
                    timestamp=datetime.utcnow(),
                    drift_type=DriftType.PREDICTION,
                    severity=DriftSeverity.HIGH,
                    metric_name='prediction_mean',
                    baseline_value=self.baseline_mean,
                    current_value=current_mean,
                    deviation_pct=((current_mean - self.baseline_mean) / self.baseline_std) * 100,
                    message=f"Prediction distribution shifted (bias detected)",
                    recommended_action="Model may be biased, investigate training data"
                ))

        return alerts


class ModelDriftMonitor:
    """
    Comprehensive model drift monitoring system.

    Combines performance, feature, and prediction drift detection.
    """

    def __init__(
        self,
        baseline_performance: Dict[str, float],
        baseline_features: Dict[str, Dict[str, float]],
        baseline_prediction_stats: Dict[str, float]
    ):
        """
        Initialize drift monitor.

        Args:
            baseline_performance: Baseline performance metrics
            baseline_features: Baseline feature statistics
            baseline_prediction_stats: Baseline prediction statistics
        """
        self.performance_monitor = PerformanceMonitor(baseline_performance)
        self.feature_detector = FeatureDriftDetector(baseline_features)
        self.prediction_detector = PredictionDriftDetector(
            baseline_prediction_stats.get('mean', 0),
            baseline_prediction_stats.get('std', 1)
        )

        self.alert_history: List[DriftAlert] = []

    def record_prediction(
        self,
        features: Dict[str, float],
        prediction: float,
        actual: Optional[float] = None,
        timestamp: Optional[datetime] = None
    ):
        """
        Record a prediction and check for drift.

        Args:
            features: Input features
            prediction: Model prediction
            actual: Actual outcome (if available)
            timestamp: Prediction timestamp
        """
        # Record with all detectors
        self.feature_detector.record_features(features)
        self.prediction_detector.record_prediction(prediction)

        if actual is not None:
            self.performance_monitor.record_prediction(prediction, actual, timestamp)

    def check_all_drift(self) -> List[DriftAlert]:
        """
        Run all drift checks.

        Returns:
            Combined list of all drift alerts
        """
        all_alerts = []

        # Check performance drift
        all_alerts.extend(self.performance_monitor.check_drift())

        # Check feature drift
        all_alerts.extend(self.feature_detector.check_drift())

        # Check prediction drift
        all_alerts.extend(self.prediction_detector.check_drift())

        # Log and store alerts
        for alert in all_alerts:
            logger.warning(f"DRIFT ALERT: {alert.message}")
            self.alert_history.append(alert)

        return all_alerts

    def get_drift_summary(self) -> Dict:
        """
        Get summary of drift status.

        Returns:
            Dictionary with drift summary
        """
        recent_alerts = [
            a for a in self.alert_history
            if (datetime.utcnow() - a.timestamp).days < 7
        ]

        severity_counts = {}
        for severity in DriftSeverity:
            severity_counts[severity.value] = sum(
                1 for a in recent_alerts if a.severity == severity
            )

        return {
            'total_alerts_7days': len(recent_alerts),
            'severity_counts': severity_counts,
            'latest_alert': recent_alerts[-1].to_dict() if recent_alerts else None,
            'requires_retraining': any(
                a.severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL]
                for a in recent_alerts
            )
        }
