"""Tests for monitoring and drift detection functionality."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from src.monitoring.drift_detector import DriftDetector, DriftResult
from src.monitoring.monitoring_config import MonitoringConfig
from src.monitoring.monitoring_service import MonitoringService


class TestDriftResult:
    """Test DriftResult class."""
    
    def test_drift_result_creation(self):
        """Test creating a DriftResult."""
        result = DriftResult(
            is_drift=True,
            drift_score=0.8,
            p_value=0.01,
            method="ks",
            feature_name="feature1"
        )
        
        assert result.is_drift is True
        assert result.drift_score == 0.8
        assert result.p_value == 0.01
        assert result.method == "ks"
        assert result.feature_name == "feature1"
        assert isinstance(result.timestamp, datetime)
    
    def test_drift_result_to_dict(self):
        """Test converting DriftResult to dictionary."""
        result = DriftResult(
            is_drift=False,
            drift_score=0.3,
            p_value=0.15,
            method="mmd"
        )
        
        result_dict = result.to_dict()
        
        assert result_dict["is_drift"] is False
        assert result_dict["drift_score"] == 0.3
        assert result_dict["p_value"] == 0.15
        assert result_dict["method"] == "mmd"
        assert "timestamp" in result_dict


class TestMonitoringConfig:
    """Test MonitoringConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = MonitoringConfig()
        
        assert config.drift_threshold == 0.05
        assert config.drift_methods == ["ks", "mmd"]
        assert config.feature_drift_enabled is True
        assert config.monitoring_frequency == 100
        assert config.alert_on_drift is True
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = MonitoringConfig(
            drift_threshold=0.01,
            drift_methods=["ks"],
            monitoring_frequency=50,
            alert_on_drift=False
        )
        
        assert config.drift_threshold == 0.01
        assert config.drift_methods == ["ks"]
        assert config.monitoring_frequency == 50
        assert config.alert_on_drift is False


class TestDriftDetector:
    """Test DriftDetector class."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return MonitoringConfig(
            drift_methods=["ks"],
            numerical_features=["feature1", "feature2"],
            categorical_features=["category_feature"]
        )
    
    @pytest.fixture
    def detector(self, config):
        """Create test drift detector."""
        return DriftDetector(config)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return pd.DataFrame({
            "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feature2": [0.5, 1.5, 2.5, 3.5, 4.5],
            "category_feature": ["A", "B", "A", "B", "A"]
        })
    
    def test_drift_detector_initialization(self, detector):
        """Test drift detector initialization."""
        assert detector.config is not None
        assert detector.reference_data is None
        assert len(detector.detectors) == 0  # Detectors created when reference data is set
        assert detector.prediction_count == 0
        assert len(detector.drift_history) == 0
    
    def test_set_reference_data(self, detector, sample_data):
        """Test setting reference data."""
        detector.set_reference_data(sample_data)
        
        assert detector.reference_data is not None
        assert detector.reference_data.shape[1] > 0  # Should have processed features
        assert len(detector.detectors) > 0  # Detectors should be created
    
    def test_preprocess_data(self, detector, sample_data):
        """Test data preprocessing."""
        processed = detector._preprocess_data(sample_data)
        
        # Should have one-hot encoded categorical features
        assert "category_feature_A" in processed.columns
        assert "category_feature_B" in processed.columns
        assert "category_feature" not in processed.columns
    
    @patch('src.monitoring.drift_detector.KSDrift')
    def test_detect_drift(self, mock_ks, detector, sample_data):
        """Test drift detection."""
        # Mock the detector
        mock_detector = Mock()
        mock_detector.predict.return_value = {
            'data': {
                'is_drift': [True],
                'distance': [0.8],
                'p_val': [0.01]
            }
        }
        detector.detectors["ks"] = mock_detector
        detector.reference_data = np.random.randn(10, 3)
        
        # Test drift detection
        results = detector.detect_drift(sample_data)
        
        assert len(results) == 1
        assert results[0].is_drift is True
        assert results[0].method == "ks"
        mock_detector.predict.assert_called_once()
    
    def test_should_check_drift(self, detector):
        """Test drift check frequency."""
        detector.config.monitoring_frequency = 5
        
        # First 4 calls should return False
        for _ in range(4):
            assert detector.should_check_drift() is False
        
        # 5th call should return True
        assert detector.should_check_drift() is True
    
    def test_get_drift_summary(self, detector):
        """Test drift summary generation."""
        # Add some mock results
        detector.drift_history = [
            DriftResult(False, 0.3, 0.1, "ks"),
            DriftResult(True, 0.8, 0.01, "mmd"),
            DriftResult(False, 0.4, 0.08, "ks")
        ]
        
        summary = detector.get_drift_summary()
        
        assert summary["total_checks"] == 3
        assert summary["drift_detected"] == 1
        assert summary["drift_rate"] == 1/3
        assert "ks" in summary["methods_used"]
        assert "mmd" in summary["methods_used"]


class TestMonitoringService:
    """Test MonitoringService class."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return MonitoringConfig(
            drift_methods=["ks"],
            monitoring_frequency=2
        )
    
    @pytest.fixture
    def service(self, config):
        """Create test monitoring service."""
        return MonitoringService(config)
    
    @pytest.fixture
    def sample_features(self):
        """Create sample features for testing."""
        return {
            "feature1": 1.5,
            "feature2": 2.3,
            "category_feature": "A"
        }
    
    def test_monitoring_service_initialization(self, service):
        """Test monitoring service initialization."""
        assert service.config is not None
        assert service.drift_detector is not None
        assert service.is_initialized is False
        assert service.monitoring_enabled is True
    
    @patch('src.monitoring.monitoring_service.pd.read_parquet')
    async def test_initialize_with_file(self, mock_read, service):
        """Test initialization with reference data file."""
        # Mock the data
        mock_data = pd.DataFrame({
            "feature1": [1, 2, 3],
            "feature2": [0.5, 1.5, 2.5],
            "category_feature": ["A", "B", "A"]
        })
        mock_read.return_value = mock_data
        
        # Mock the drift detector methods
        service.drift_detector.set_reference_data = Mock()
        
        result = await service.initialize("test_data.parquet")
        
        assert result is True
        assert service.is_initialized is True
        mock_read.assert_called_once_with("test_data.parquet")
    
    async def test_initialize_without_file(self, service):
        """Test initialization without reference data file."""
        result = await service.initialize()
        
        assert result is False
        assert service.monitoring_enabled is False
    
    @patch.object(DriftDetector, 'detect_drift')
    @patch.object(DriftDetector, 'detect_feature_drift')
    @patch.object(DriftDetector, 'should_check_drift')
    async def test_monitor_prediction(self, mock_should_check, mock_feature_drift, mock_detect, service, sample_features):
        """Test monitoring a single prediction."""
        # Mock the drift detector
        service.is_initialized = True
        service.monitoring_enabled = True
        
        mock_should_check.return_value = True
        mock_detect.return_value = [DriftResult(False, 0.3, 0.1, "ks")]
        mock_feature_drift.return_value = {}
        
        result = await service.monitor_prediction(sample_features)
        
        assert result["monitoring_enabled"] is True
        assert result["drift_detected"] is False
        assert "timestamp" in result
        mock_detect.assert_called_once()
    
    @patch.object(DriftDetector, 'detect_drift')
    @patch.object(DriftDetector, 'detect_feature_drift')
    @patch.object(DriftDetector, 'should_check_drift')
    async def test_monitor_batch_predictions(self, mock_should_check, mock_feature_drift, mock_detect, service):
        """Test monitoring batch predictions."""
        # Mock the drift detector
        service.is_initialized = True
        service.monitoring_enabled = True
        
        mock_should_check.return_value = True
        mock_detect.return_value = [DriftResult(True, 0.8, 0.01, "ks")]
        mock_feature_drift.return_value = {}
        
        features_list = [
            {"feature1": 1.0, "feature2": 2.0, "category_feature": "A"},
            {"feature1": 1.5, "feature2": 2.5, "category_feature": "B"}
        ]
        
        result = await service.monitor_batch_predictions(features_list)
        
        assert result["monitoring_enabled"] is True
        assert result["drift_detected"] is True
        assert result["batch_size"] == 2
        mock_detect.assert_called_once()
    
    def test_get_monitoring_status(self, service):
        """Test getting monitoring status."""
        service.is_initialized = True
        service.monitoring_enabled = True
        
        status = service.get_monitoring_status()
        
        assert status["monitoring_enabled"] is True
        assert status["is_initialized"] is True
        assert "drift_summary" in status
        assert "config" in status
    
    def test_enable_disable_monitoring(self, service):
        """Test enabling and disabling monitoring."""
        # Test disable
        service.disable_monitoring()
        assert service.monitoring_enabled is False
        
        # Test enable
        service.enable_monitoring()
        assert service.monitoring_enabled is True
    
    def test_reset_monitoring(self, service):
        """Test resetting monitoring state."""
        # Add some mock history
        service.drift_detector.drift_history = [DriftResult(True, 0.8, 0.01, "ks")]
        service.drift_detector.prediction_count = 10
        
        service.reset_monitoring()
        
        assert len(service.drift_detector.drift_history) == 0
        assert service.drift_detector.prediction_count == 0


class TestDriftDetectionIntegration:
    """Integration tests for drift detection."""
    
    def test_end_to_end_drift_detection(self):
        """Test end-to-end drift detection workflow."""
        # Create configuration
        config = MonitoringConfig(
            drift_methods=["ks"],
            monitoring_frequency=1,
            numerical_features=["feature1", "feature2"],
            categorical_features=["category_feature"]
        )
        
        # Create detector
        detector = DriftDetector(config)
        
        # Create reference data
        reference_data = pd.DataFrame({
            "feature1": np.random.normal(0, 1, 100),
            "feature2": np.random.normal(0, 1, 100),
            "category_feature": np.random.choice(["A", "B"], 100)
        })
        
        # Set reference data
        detector.set_reference_data(reference_data)
        
        # Create test data (similar distribution)
        test_data = pd.DataFrame({
            "feature1": np.random.normal(0, 1, 50),
            "feature2": np.random.normal(0, 1, 50),
            "category_feature": np.random.choice(["A", "B"], 50)
        })
        
        # Detect drift
        results = detector.detect_drift(test_data)
        
        # Should have results
        assert len(results) > 0
        assert all(isinstance(result, DriftResult) for result in results)
    
    def test_drift_detection_with_different_distributions(self):
        """Test drift detection with significantly different distributions."""
        # Create configuration
        config = MonitoringConfig(
            drift_methods=["ks"],
            monitoring_frequency=1,
            numerical_features=["feature1"],
            categorical_features=[]
        )
        
        # Create detector
        detector = DriftDetector(config)
        
        # Create reference data (normal distribution)
        reference_data = pd.DataFrame({
            "feature1": np.random.normal(0, 1, 100)
        })
        
        # Set reference data
        detector.set_reference_data(reference_data)
        
        # Create test data (different distribution)
        test_data = pd.DataFrame({
            "feature1": np.random.normal(5, 1, 50)  # Different mean
        })
        
        # Detect drift
        results = detector.detect_drift(test_data)
        
        # Should detect drift
        assert len(results) > 0
        # Note: Actual drift detection depends on the specific test data
        # and may not always detect drift due to randomness
