"""Tests for feature engineering module."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.pricing_rf.features import (
    create_feature_pipeline,
    add_time_features,
    add_lag_features,
    add_rolling_features,
    add_price_change_features,
    create_all_features
)


class TestFeaturePipeline:
    """Test feature preprocessing pipeline."""
    
    def test_create_feature_pipeline(self):
        """Test feature pipeline creation."""
        categorical_features = ['category']
        numerical_features = ['feature1', 'feature2']
        
        pipeline = create_feature_pipeline(categorical_features, numerical_features)
        
        assert pipeline is not None
        assert len(pipeline.transformers) == 2
    
    def test_feature_pipeline_fit_transform(self):
        """Test pipeline fit and transform."""
        # Create sample data
        data = {
            'feature1': [1.0, 2.0, 3.0],
            'feature2': [4.0, 5.0, 6.0],
            'category': ['A', 'B', 'A']
        }
        df = pd.DataFrame(data)
        
        pipeline = create_feature_pipeline(['category'], ['feature1', 'feature2'])
        X_transformed = pipeline.fit_transform(df)
        
        assert X_transformed.shape[0] == 3
        assert X_transformed.shape[1] > 0


class TestTimeFeatures:
    """Test time-based feature creation."""
    
    def test_add_time_features(self):
        """Test adding time features."""
        # Create sample data with timestamp
        timestamps = pd.date_range('2024-01-01', periods=10, freq='D')
        df = pd.DataFrame({
            'timestamp': timestamps,
            'price': np.random.randn(10) * 100 + 1000
        })
        
        df_with_time = add_time_features(df, 'timestamp')
        
        # Check that time features were added
        expected_features = ['year', 'month', 'day', 'dayofweek', 'dayofyear', 'week', 'quarter']
        for feature in expected_features:
            assert feature in df_with_time.columns
        
        # Check cyclical encoding
        assert 'month_sin' in df_with_time.columns
        assert 'month_cos' in df_with_time.columns
        assert 'dayofweek_sin' in df_with_time.columns
        assert 'dayofweek_cos' in df_with_time.columns


class TestLagFeatures:
    """Test lag feature creation."""
    
    def test_add_lag_features(self):
        """Test adding lag features."""
        df = pd.DataFrame({
            'price': [100, 105, 98, 110, 95, 102, 108, 99, 103, 107]
        })
        
        df_with_lags = add_lag_features(df, 'price', [1, 2])
        
        # Check lag features were added
        assert 'price_lag_1' in df_with_lags.columns
        assert 'price_lag_2' in df_with_lags.columns
        
        # Check lag values
        assert pd.isna(df_with_lags['price_lag_1'].iloc[0])
        assert df_with_lags['price_lag_1'].iloc[1] == 100
        assert df_with_lags['price_lag_2'].iloc[2] == 100


class TestRollingFeatures:
    """Test rolling window features."""
    
    def test_add_rolling_features(self):
        """Test adding rolling features."""
        df = pd.DataFrame({
            'price': [100, 105, 98, 110, 95, 102, 108, 99, 103, 107]
        })
        
        df_with_rolling = add_rolling_features(df, 'price', [3, 5])
        
        # Check rolling features were added
        expected_features = [
            'price_rolling_mean_3', 'price_rolling_std_3',
            'price_rolling_min_3', 'price_rolling_max_3',
            'price_rolling_mean_5', 'price_rolling_std_5',
            'price_rolling_min_5', 'price_rolling_max_5'
        ]
        
        for feature in expected_features:
            assert feature in df_with_rolling.columns
        
        # Check that rolling mean is calculated correctly
        assert df_with_rolling['price_rolling_mean_3'].iloc[2] == (100 + 105 + 98) / 3


class TestPriceChangeFeatures:
    """Test price change features."""
    
    def test_add_price_change_features(self):
        """Test adding price change features."""
        df = pd.DataFrame({
            'price': [100, 105, 98, 110, 95, 102, 108, 99, 103, 107]
        })
        
        df_with_changes = add_price_change_features(df, 'price', [1, 2])
        
        # Check change features were added
        assert 'price_pct_change_1' in df_with_changes.columns
        assert 'price_pct_change_2' in df_with_changes.columns
        assert 'price_diff_1' in df_with_changes.columns
        assert 'price_diff_2' in df_with_changes.columns
        
        # Check percentage change calculation
        expected_pct_change = (105 - 100) / 100
        assert abs(df_with_changes['price_pct_change_1'].iloc[1] - expected_pct_change) < 1e-10


class TestAllFeatures:
    """Test comprehensive feature creation."""
    
    def test_create_all_features(self):
        """Test creating all features."""
        # Create sample data
        timestamps = pd.date_range('2024-01-01', periods=10, freq='D')
        df = pd.DataFrame({
            'timestamp': timestamps,
            'price': np.random.randn(10) * 100 + 1000,
            'feature1': np.random.randn(10),
            'feature2': np.random.randn(10),
            'category': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B']
        })
        
        df_with_features = create_all_features(
            df, 'timestamp', 'price', ['category'], ['feature1', 'feature2']
        )
        
        # Check that the dataframe has more columns than the original
        assert len(df_with_features.columns) > len(df.columns)
        
        # Check that time features were added
        assert 'year' in df_with_features.columns
        assert 'month' in df_with_features.columns
        
        # Check that lag features were added
        assert 'price_lag_1' in df_with_features.columns
        
        # Check that rolling features were added
        assert 'price_rolling_mean_7' in df_with_features.columns

