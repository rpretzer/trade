"""
Unit Tests for Model Management Module
Tests model versioning, walk-forward validation, and A/B testing
"""

import pytest
import json
import tempfile
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from model_management import (
    ModelVersion, ModelStatus, ModelRegistry, WalkForwardValidator,
    ABTestController, calculate_model_checksum, verify_model_integrity
)


class TestWalkForwardValidator:
    """Test walk-forward validation."""

    def test_create_splits_basic(self):
        """Test creating basic validation splits."""
        # Create time series data
        dates = pd.date_range('2020-01-01', periods=1000, freq='D')
        data = pd.DataFrame({
            'value': np.random.randn(1000)
        }, index=dates)

        splits = WalkForwardValidator.create_splits(
            data,
            n_splits=3,
            train_size=0.6,
            val_size=0.2,
            test_size=0.2
        )

        # Should have 3 splits
        assert len(splits) == 3

        # Each split should have train, val, test
        for train, val, test in splits:
            assert len(train) > 0
            assert len(val) > 0
            assert len(test) > 0

    def test_temporal_ordering(self):
        """Test that splits maintain temporal ordering."""
        dates = pd.date_range('2020-01-01', periods=500, freq='D')
        data = pd.DataFrame({
            'value': np.random.randn(500)
        }, index=dates)

        splits = WalkForwardValidator.create_splits(data, n_splits=2)

        for train, val, test in splits:
            # Validation should come after training
            assert train.index.max() < val.index.min()

            # Test should come after validation
            assert val.index.max() < test.index.min()

    def test_validate_temporal_ordering_valid(self):
        """Test temporal ordering validation with valid data."""
        dates1 = pd.date_range('2020-01-01', periods=100, freq='D')
        dates2 = pd.date_range('2020-04-11', periods=50, freq='D')
        dates3 = pd.date_range('2020-06-01', periods=50, freq='D')

        train = pd.DataFrame({'value': np.random.randn(100)}, index=dates1)
        val = pd.DataFrame({'value': np.random.randn(50)}, index=dates2)
        test = pd.DataFrame({'value': np.random.randn(50)}, index=dates3)

        is_valid = WalkForwardValidator.validate_temporal_ordering(train, val, test)
        assert is_valid is True

    def test_validate_temporal_ordering_invalid(self):
        """Test temporal ordering validation detects leakage."""
        dates1 = pd.date_range('2020-01-01', periods=100, freq='D')
        dates2 = pd.date_range('2020-03-01', periods=50, freq='D')  # Overlaps with train
        dates3 = pd.date_range('2020-05-01', periods=50, freq='D')

        train = pd.DataFrame({'value': np.random.randn(100)}, index=dates1)
        val = pd.DataFrame({'value': np.random.randn(50)}, index=dates2)
        test = pd.DataFrame({'value': np.random.randn(50)}, index=dates3)

        is_valid = WalkForwardValidator.validate_temporal_ordering(train, val, test)
        assert is_valid is False

    def test_splits_proportions(self):
        """Test that splits respect train/val/test proportions."""
        dates = pd.date_range('2020-01-01', periods=1000, freq='D')
        data = pd.DataFrame({'value': np.random.randn(1000)}, index=dates)

        splits = WalkForwardValidator.create_splits(
            data,
            n_splits=1,
            train_size=0.6,
            val_size=0.2,
            test_size=0.2
        )

        train, val, test = splits[0]
        total = len(train) + len(val) + len(test)

        # Check proportions (with some tolerance)
        train_ratio = len(train) / total
        val_ratio = len(val) / total
        test_ratio = len(test) / total

        assert 0.55 < train_ratio < 0.65
        assert 0.15 < val_ratio < 0.25
        assert 0.15 < test_ratio < 0.25


class TestModelVersion:
    """Test ModelVersion dataclass."""

    def test_model_version_creation(self):
        """Test creating a model version."""
        version = ModelVersion(
            version="1.0.0",
            model_type="LSTM",
            created_at=datetime.now().isoformat(),
            created_by="test_user",
            config={'layers': [50, 30], 'dropout': 0.2},
            features=['price', 'volume'],
            target='price_diff',
            training_samples=1000,
            training_date_range="2020-01-01 to 2020-12-31",
            train_metrics={'mse': 0.01, 'mae': 0.05},
            validation_metrics={'mse': 0.02, 'mae': 0.06},
            model_path="/path/to/model.h5",
            status=ModelStatus.TRAINING
        )

        assert version.version == "1.0.0"
        assert version.model_type == "LSTM"
        assert version.status == ModelStatus.TRAINING

    def test_model_version_to_dict(self):
        """Test converting model version to dictionary."""
        version = ModelVersion(
            version="1.0.0",
            model_type="LSTM",
            created_at=datetime.now().isoformat(),
            created_by="test_user",
            config={},
            features=['price'],
            target='price_diff',
            training_samples=1000,
            training_date_range="2020-01-01 to 2020-12-31",
            train_metrics={},
            validation_metrics={},
            model_path="/path/to/model.h5"
        )

        data = version.to_dict()

        assert isinstance(data, dict)
        assert data['version'] == "1.0.0"
        assert data['status'] == ModelStatus.TRAINING.value

    def test_model_version_from_dict(self):
        """Test creating model version from dictionary."""
        data = {
            'version': "1.0.0",
            'model_type': "LSTM",
            'created_at': datetime.now().isoformat(),
            'created_by': "test_user",
            'config': {},
            'features': ['price'],
            'target': 'price_diff',
            'training_samples': 1000,
            'training_date_range': "2020-01-01 to 2020-12-31",
            'train_metrics': {},
            'validation_metrics': {},
            'model_path': "/path/to/model.h5",
            'status': 'PRODUCTION'
        }

        version = ModelVersion.from_dict(data)

        assert version.version == "1.0.0"
        assert version.status == ModelStatus.PRODUCTION


class TestModelRegistry:
    """Test model registry."""

    def setup_method(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.registry_path = Path(self.test_dir) / "test_registry.json"

    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_registry_initialization(self):
        """Test registry initialization."""
        registry = ModelRegistry(registry_path=str(self.registry_path))

        assert len(registry.models) == 0

    def test_register_model(self):
        """Test registering a model."""
        registry = ModelRegistry(registry_path=str(self.registry_path))

        version = ModelVersion(
            version="1.0.0",
            model_type="LSTM",
            created_at=datetime.now().isoformat(),
            created_by="test_user",
            config={},
            features=['price'],
            target='price_diff',
            training_samples=1000,
            training_date_range="2020-01-01 to 2020-12-31",
            train_metrics={'mse': 0.01},
            validation_metrics={'mse': 0.02},
            model_path="/path/to/model.h5"
        )

        result = registry.register_model(version)

        assert result is True
        assert "1.0.0" in registry.models

    def test_register_duplicate_version(self):
        """Test that duplicate versions are rejected."""
        registry = ModelRegistry(registry_path=str(self.registry_path))

        version = ModelVersion(
            version="1.0.0",
            model_type="LSTM",
            created_at=datetime.now().isoformat(),
            created_by="test_user",
            config={},
            features=['price'],
            target='price_diff',
            training_samples=1000,
            training_date_range="2020-01-01 to 2020-12-31",
            train_metrics={},
            validation_metrics={},
            model_path="/path/to/model.h5"
        )

        registry.register_model(version)
        result = registry.register_model(version)

        assert result is False

    def test_get_production_model(self):
        """Test getting production model."""
        registry = ModelRegistry(registry_path=str(self.registry_path))

        # Register training model
        v1 = ModelVersion(
            version="1.0.0",
            model_type="LSTM",
            created_at=datetime.now().isoformat(),
            created_by="test_user",
            config={},
            features=['price'],
            target='price_diff',
            training_samples=1000,
            training_date_range="2020-01-01 to 2020-12-31",
            train_metrics={},
            validation_metrics={},
            model_path="/path/to/model1.h5",
            status=ModelStatus.TRAINING
        )
        registry.register_model(v1)

        # Register production model
        v2 = ModelVersion(
            version="2.0.0",
            model_type="LSTM",
            created_at=datetime.now().isoformat(),
            created_by="test_user",
            config={},
            features=['price'],
            target='price_diff',
            training_samples=1000,
            training_date_range="2020-01-01 to 2020-12-31",
            train_metrics={},
            validation_metrics={},
            model_path="/path/to/model2.h5",
            status=ModelStatus.PRODUCTION
        )
        registry.register_model(v2)

        prod_model = registry.get_production_model()

        assert prod_model is not None
        assert prod_model.version == "2.0.0"

    def test_update_model_status(self):
        """Test updating model status."""
        registry = ModelRegistry(registry_path=str(self.registry_path))

        version = ModelVersion(
            version="1.0.0",
            model_type="LSTM",
            created_at=datetime.now().isoformat(),
            created_by="test_user",
            config={},
            features=['price'],
            target='price_diff',
            training_samples=1000,
            training_date_range="2020-01-01 to 2020-12-31",
            train_metrics={},
            validation_metrics={},
            model_path="/path/to/model.h5",
            status=ModelStatus.TRAINING
        )
        registry.register_model(version)

        registry.update_model_status("1.0.0", ModelStatus.PRODUCTION)

        model = registry.get_model("1.0.0")
        assert model.status == ModelStatus.PRODUCTION
        assert model.deployed_at is not None

    def test_list_models_by_status(self):
        """Test listing models filtered by status."""
        registry = ModelRegistry(registry_path=str(self.registry_path))

        # Register multiple models with different statuses
        for i, status in enumerate([ModelStatus.TRAINING, ModelStatus.PRODUCTION, ModelStatus.CANARY]):
            version = ModelVersion(
                version=f"{i}.0.0",
                model_type="LSTM",
                created_at=datetime.now().isoformat(),
                created_by="test_user",
                config={},
                features=['price'],
                target='price_diff',
                training_samples=1000,
                training_date_range="2020-01-01 to 2020-12-31",
                train_metrics={},
                validation_metrics={},
                model_path=f"/path/to/model{i}.h5",
                status=status
            )
            registry.register_model(version)

        prod_models = registry.list_models(status=ModelStatus.PRODUCTION)
        assert len(prod_models) == 1

    def test_registry_persistence(self):
        """Test that registry persists to disk."""
        # Create registry and add model
        registry1 = ModelRegistry(registry_path=str(self.registry_path))

        version = ModelVersion(
            version="1.0.0",
            model_type="LSTM",
            created_at=datetime.now().isoformat(),
            created_by="test_user",
            config={},
            features=['price'],
            target='price_diff',
            training_samples=1000,
            training_date_range="2020-01-01 to 2020-12-31",
            train_metrics={},
            validation_metrics={},
            model_path="/path/to/model.h5"
        )
        registry1.register_model(version)

        # Create new registry instance (simulating restart)
        registry2 = ModelRegistry(registry_path=str(self.registry_path))

        # Should load the saved model
        assert "1.0.0" in registry2.models


class TestABTestController:
    """Test A/B testing controller."""

    def test_select_model_production_only(self):
        """Test model selection with only production model."""
        registry = ModelRegistry(registry_path=tempfile.mktemp())

        prod_version = ModelVersion(
            version="1.0.0",
            model_type="LSTM",
            created_at=datetime.now().isoformat(),
            created_by="test_user",
            config={},
            features=['price'],
            target='price_diff',
            training_samples=1000,
            training_date_range="2020-01-01 to 2020-12-31",
            train_metrics={},
            validation_metrics={},
            model_path="/path/to/model.h5",
            status=ModelStatus.PRODUCTION
        )
        registry.register_model(prod_version)

        controller = ABTestController(registry)

        # Should always return production model
        for _ in range(10):
            model = controller.select_model()
            assert model.version == "1.0.0"

    def test_select_model_with_canary(self):
        """Test model selection with canary deployment."""
        registry = ModelRegistry(registry_path=tempfile.mktemp())

        # Production model
        prod_version = ModelVersion(
            version="1.0.0",
            model_type="LSTM",
            created_at=datetime.now().isoformat(),
            created_by="test_user",
            config={},
            features=['price'],
            target='price_diff',
            training_samples=1000,
            training_date_range="2020-01-01 to 2020-12-31",
            train_metrics={},
            validation_metrics={},
            model_path="/path/to/model1.h5",
            status=ModelStatus.PRODUCTION,
            traffic_percentage=0.9
        )
        registry.register_model(prod_version)

        # Canary model (10% traffic)
        canary_version = ModelVersion(
            version="2.0.0",
            model_type="LSTM",
            created_at=datetime.now().isoformat(),
            created_by="test_user",
            config={},
            features=['price'],
            target='price_diff',
            training_samples=1000,
            training_date_range="2020-01-01 to 2020-12-31",
            train_metrics={},
            validation_metrics={},
            model_path="/path/to/model2.h5",
            status=ModelStatus.CANARY,
            traffic_percentage=0.1
        )
        registry.register_model(canary_version)

        controller = ABTestController(registry)

        # Make many selections
        selections = []
        for _ in range(1000):
            model = controller.select_model()
            selections.append(model.version)

        # Check distribution (should be approximately 90/10)
        canary_count = selections.count("2.0.0")
        canary_ratio = canary_count / 1000

        assert 0.05 < canary_ratio < 0.15  # Allow some variance

    def test_get_traffic_stats(self):
        """Test getting traffic statistics."""
        temp_file = tempfile.mktemp()
        registry = ModelRegistry(registry_path=temp_file)

        prod_version = ModelVersion(
            version="1.0.0",
            model_type="LSTM",
            created_at=datetime.now().isoformat(),
            created_by="test_user",
            config={},
            features=['price'],
            target='price_diff',
            training_samples=1000,
            training_date_range="2020-01-01 to 2020-12-31",
            train_metrics={},
            validation_metrics={},
            model_path="/path/to/model.h5",
            status=ModelStatus.PRODUCTION
        )
        registry.register_model(prod_version, user='test_user')

        controller = ABTestController(registry)

        # Make some selections
        for _ in range(100):
            model = controller.select_model()
            assert model is not None  # Verify we get a model

        stats = controller.get_traffic_stats()

        assert "1.0.0" in stats, f"Expected '1.0.0' in stats, got {stats}"
        assert stats["1.0.0"]['count'] == 100
        assert stats["1.0.0"]['percentage'] == 1.0

        # Clean up
        try:
            Path(temp_file).unlink()
        except:
            pass


class TestModelIntegrity:
    """Test model integrity verification."""

    def test_calculate_checksum(self):
        """Test calculating model file checksum."""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("test model content")
            temp_path = f.name

        try:
            checksum1 = calculate_model_checksum(temp_path)
            checksum2 = calculate_model_checksum(temp_path)

            # Same file should produce same checksum
            assert checksum1 == checksum2
            assert len(checksum1) == 64  # SHA-256 is 64 hex chars
        finally:
            Path(temp_path).unlink()

    def test_verify_model_integrity_valid(self):
        """Test integrity verification with valid file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("test model content")
            temp_path = f.name

        try:
            checksum = calculate_model_checksum(temp_path)
            is_valid = verify_model_integrity(temp_path, checksum)

            assert is_valid is True
        finally:
            Path(temp_path).unlink()

    def test_verify_model_integrity_tampered(self):
        """Test integrity verification detects tampering."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("original content")
            temp_path = f.name

        try:
            # Calculate original checksum
            original_checksum = calculate_model_checksum(temp_path)

            # Tamper with file
            with open(temp_path, 'w') as f:
                f.write("tampered content")

            # Verification should fail
            is_valid = verify_model_integrity(temp_path, original_checksum)
            assert is_valid is False
        finally:
            Path(temp_path).unlink()
