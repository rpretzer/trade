"""
Model Management Module
Handles model versioning, validation, A/B testing, and deployment
"""

import json
import logging
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum

from exceptions import ModelLoadError, ModelVersionMismatchError
from audit_logging import audit_log, AuditEventType

logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    """Model deployment status."""
    TRAINING = "TRAINING"
    VALIDATING = "VALIDATING"
    TESTING = "TESTING"
    SHADOW = "SHADOW"  # Running alongside production
    CANARY = "CANARY"  # Serving small % of traffic
    PRODUCTION = "PRODUCTION"
    DEPRECATED = "DEPRECATED"
    ARCHIVED = "ARCHIVED"


@dataclass
class ModelMetrics:
    """Model performance metrics."""
    # Statistical metrics
    mse: float
    mae: float
    r_squared: float

    # Trading-specific metrics
    directional_accuracy: float  # % predictions with correct direction
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float

    # Distribution metrics
    prediction_mean: float
    prediction_std: float

    # Validation info
    validation_samples: int
    validation_date_range: str

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ModelVersion:
    """Model version metadata."""
    version: str  # Semantic version (e.g., "1.2.3")
    model_type: str  # e.g., "LSTM", "XGBoost", "RandomForest"
    created_at: str
    created_by: str

    # Model configuration
    config: Dict[str, Any]
    features: List[str]
    target: str

    # Training info
    training_samples: int
    training_date_range: str

    # Performance metrics
    train_metrics: Dict[str, float]
    validation_metrics: Dict[str, float]

    # File paths
    model_path: str

    # Optional fields (with defaults)
    test_metrics: Optional[Dict[str, float]] = None
    scaler_path: Optional[str] = None
    status: ModelStatus = ModelStatus.TRAINING
    deployed_at: Optional[str] = None
    traffic_percentage: float = 0.0
    model_checksum: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        data = asdict(self)
        data['status'] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: Dict) -> 'ModelVersion':
        """Create from dictionary."""
        data = data.copy()
        if 'status' in data:
            data['status'] = ModelStatus(data['status'])
        return cls(**data)


class WalkForwardValidator:
    """
    Walk-forward validation for time series models.

    Splits data into multiple windows with proper temporal ordering.
    """

    @staticmethod
    def create_splits(
        data: pd.DataFrame,
        n_splits: int = 5,
        train_size: float = 0.6,
        val_size: float = 0.2,
        test_size: float = 0.2
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
        """
        Create walk-forward validation splits.

        Args:
            data: Time series data (must be sorted by date)
            n_splits: Number of validation windows
            train_size: Fraction for training
            val_size: Fraction for validation
            test_size: Fraction for testing

        Returns:
            List of (train, val, test) DataFrame tuples
        """
        assert train_size + val_size + test_size == 1.0, "Sizes must sum to 1.0"

        total_len = len(data)
        window_size = total_len // (n_splits + 2)  # Reserve space for initial training

        splits = []

        for i in range(n_splits):
            # Calculate indices for this split
            start_idx = i * window_size
            train_end = int(start_idx + window_size * (train_size / (train_size + val_size + test_size)) * 3)
            val_end = int(train_end + window_size * (val_size / (train_size + val_size + test_size)) * 3)
            test_end = min(val_end + window_size, total_len)

            if test_end - val_end < 10:  # Need at least 10 samples for test
                break

            train = data.iloc[start_idx:train_end]
            val = data.iloc[train_end:val_end]
            test = data.iloc[val_end:test_end]

            splits.append((train, val, test))

            logger.info(
                f"Split {i+1}: Train={len(train)}, Val={len(val)}, Test={len(test)}"
            )

        return splits

    @staticmethod
    def validate_temporal_ordering(
        train: pd.DataFrame,
        val: pd.DataFrame,
        test: pd.DataFrame
    ) -> bool:
        """
        Ensure no data leakage - validation and test come after training.

        Args:
            train: Training data
            val: Validation data
            test: Test data

        Returns:
            True if ordering is correct
        """
        if train.index.max() >= val.index.min():
            logger.error("Data leakage: Training data overlaps with validation")
            return False

        if val.index.max() >= test.index.min():
            logger.error("Data leakage: Validation data overlaps with test")
            return False

        return True


class ModelRegistry:
    """
    Central registry for model versions.

    Tracks all model versions, metrics, and deployment status.
    """

    def __init__(self, registry_path: str = "model_registry.json"):
        """
        Initialize model registry.

        Args:
            registry_path: Path to registry JSON file
        """
        self.registry_path = Path(registry_path)
        self.models: Dict[str, ModelVersion] = {}
        self._load_registry()

    def _load_registry(self):
        """Load registry from disk."""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, 'r') as f:
                    data = json.load(f)

                for version_str, model_data in data.items():
                    self.models[version_str] = ModelVersion.from_dict(model_data)

                logger.info(f"Loaded {len(self.models)} models from registry")
            except Exception as e:
                logger.error(f"Error loading registry: {e}")

    def _save_registry(self):
        """Save registry to disk."""
        try:
            data = {
                version: model.to_dict()
                for version, model in self.models.items()
            }

            with open(self.registry_path, 'w') as f:
                json.dump(data, f, indent=2)

            logger.info(f"Saved registry with {len(self.models)} models")
        except Exception as e:
            logger.error(f"Error saving registry: {e}")
            raise

    def register_model(
        self,
        version: ModelVersion,
        user: str = 'system'
    ) -> bool:
        """
        Register a new model version.

        Args:
            version: Model version to register
            user: User registering the model

        Returns:
            True if successful
        """
        if version.version in self.models:
            logger.warning(f"Model version {version.version} already exists")
            return False

        self.models[version.version] = version
        self._save_registry()

        # Audit log
        audit_log(
            event_type=AuditEventType.MODEL_DEPLOYED,
            user=user,
            action="register_model",
            resource=version.version,
            status="SUCCESS",
            details={
                'model_type': version.model_type,
                'status': version.status.value,
                'train_metrics': version.train_metrics
            }
        )

        logger.info(f"Registered model version {version.version}")
        return True

    def get_model(self, version: str) -> Optional[ModelVersion]:
        """Get model by version."""
        return self.models.get(version)

    def get_production_model(self) -> Optional[ModelVersion]:
        """Get current production model."""
        prod_models = [
            m for m in self.models.values()
            if m.status == ModelStatus.PRODUCTION
        ]

        if not prod_models:
            return None

        # Return most recent
        return max(prod_models, key=lambda m: m.created_at)

    def get_canary_models(self) -> List[ModelVersion]:
        """Get all canary models."""
        return [
            m for m in self.models.values()
            if m.status == ModelStatus.CANARY
        ]

    def update_model_status(
        self,
        version: str,
        status: ModelStatus,
        user: str = 'system'
    ):
        """
        Update model status.

        Args:
            version: Model version
            status: New status
            user: User making the change
        """
        if version not in self.models:
            raise ModelVersionMismatchError(f"Model version {version} not found")

        old_status = self.models[version].status
        self.models[version].status = status

        if status == ModelStatus.PRODUCTION:
            self.models[version].deployed_at = datetime.now(timezone.utc).isoformat()

        self._save_registry()

        # Audit log
        audit_log(
            event_type=AuditEventType.MODEL_UPDATED,
            user=user,
            action="update_status",
            resource=version,
            status="SUCCESS",
            details={
                'old_status': old_status.value,
                'new_status': status.value
            }
        )

        logger.info(f"Updated model {version} status: {old_status.value} → {status.value}")

    def list_models(
        self,
        status: Optional[ModelStatus] = None
    ) -> List[ModelVersion]:
        """
        List all models, optionally filtered by status.

        Args:
            status: Filter by status

        Returns:
            List of model versions
        """
        if status:
            return [m for m in self.models.values() if m.status == status]
        return list(self.models.values())


class ABTestController:
    """
    A/B testing controller for model comparison.

    Routes predictions to different models and tracks performance.
    """

    def __init__(self, registry: ModelRegistry):
        """
        Initialize A/B test controller.

        Args:
            registry: Model registry
        """
        self.registry = registry
        self.prediction_counts: Dict[str, int] = {}

    def select_model(self, request_id: Optional[str] = None) -> ModelVersion:
        """
        Select which model to use for a prediction.

        Uses traffic allocation percentages for canary models.

        Args:
            request_id: Optional request ID for consistent routing

        Returns:
            Selected model version
        """
        # Get production and canary models
        prod_model = self.registry.get_production_model()
        canary_models = self.registry.get_canary_models()

        if not prod_model:
            raise ModelLoadError("No production model available")

        if not canary_models:
            # No canary models, always use production
            self.prediction_counts[prod_model.version] = \
                self.prediction_counts.get(prod_model.version, 0) + 1
            return prod_model

        # Calculate traffic allocation
        total_canary_traffic = sum(m.traffic_percentage for m in canary_models)

        # Generate random number or hash request_id for consistent routing
        if request_id:
            # Consistent hashing for same request ID
            hash_val = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
            rand = (hash_val % 10000) / 10000.0
        else:
            rand = np.random.random()

        # Route to canary if rand < total_canary_traffic
        if total_canary_traffic > 0 and rand < total_canary_traffic:
            # Select which canary
            cumulative = 0.0
            for canary in canary_models:
                cumulative += canary.traffic_percentage
                if rand < cumulative:
                    self.prediction_counts[canary.version] = \
                        self.prediction_counts.get(canary.version, 0) + 1
                    return canary

        # Default to production (always update count)
        self.prediction_counts[prod_model.version] = \
            self.prediction_counts.get(prod_model.version, 0) + 1
        return prod_model

    def get_traffic_stats(self) -> Dict[str, Dict]:
        """Get traffic distribution statistics."""
        total = sum(self.prediction_counts.values())

        if total == 0:
            return {}

        return {
            version: {
                'count': count,
                'percentage': count / total
            }
            for version, count in self.prediction_counts.items()
        }


def calculate_model_checksum(model_path: str) -> str:
    """
    Calculate checksum of model file for integrity verification.

    Args:
        model_path: Path to model file

    Returns:
        SHA-256 checksum
    """
    sha256 = hashlib.sha256()

    with open(model_path, 'rb') as f:
        while chunk := f.read(8192):
            sha256.update(chunk)

    return sha256.hexdigest()


def verify_model_integrity(model_path: str, expected_checksum: str) -> bool:
    """
    Verify model file hasn't been tampered with.

    Args:
        model_path: Path to model file
        expected_checksum: Expected checksum

    Returns:
        True if integrity check passes
    """
    actual_checksum = calculate_model_checksum(model_path)

    if actual_checksum != expected_checksum:
        logger.error(
            f"Model integrity check failed: "
            f"expected {expected_checksum[:16]}..., "
            f"got {actual_checksum[:16]}..."
        )
        return False

    return True
