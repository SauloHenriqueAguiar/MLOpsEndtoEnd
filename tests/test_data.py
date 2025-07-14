"""
Tests for data processing components.
"""

import pytest
import pandas as pd
import numpy as np
from src.data.data_loader import DataLoader
from src.data.data_preprocessing import DataPreprocessor
import tempfile
import os


class TestDataLoader:
    """Test DataLoader functionality."""
    
    def test_generate_synthetic_data(self):
        """Test synthetic data generation."""
        loader = DataLoader()
        df = loader.generate_synthetic_data(100)
        
        assert len(df) == 100
        assert 'area' in df.columns
        assert 'preco' in df.columns
        assert df['area'].min() >= 30
        assert df['preco'].min() >= 50000
    
    def test_save_and_load_data(self):
        """Test data saving and loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            loader = DataLoader(temp_dir)
            
            # Generate and save data
            df_original = loader.generate_synthetic_data(50)
            loader.save_data(df_original, "test_data.csv")
            
            # Load data
            df_loaded = loader.load_raw_data("test_data.csv")
            
            assert len(df_loaded) == 50
            pd.testing.assert_frame_equal(df_original, df_loaded)


class TestDataPreprocessor:
    """Test DataPreprocessor functionality."""
    
    def test_clean_data(self):
        """Test data cleaning."""
        # Create test data with outliers
        data = {
            'area': [100, 200, 10000, 150],  # 10000 is outlier
            'preco': [200000, 400000, 50000, 300000]
        }
        df = pd.DataFrame(data)
        
        preprocessor = DataPreprocessor()
        df_clean = preprocessor.clean_data(df)
        
        # Should remove outlier
        assert len(df_clean) < len(df)
        assert df_clean['area'].max() < 10000
    
    def test_split_data(self):
        """Test data splitting."""
        loader = DataLoader()
        df = loader.generate_synthetic_data(100)
        
        preprocessor = DataPreprocessor()
        X_train, X_test, y_train, y_test = preprocessor.split_data(df)
        
        assert len(X_train) == 80  # 80% for training
        assert len(X_test) == 20   # 20% for testing
        assert len(y_train) == 80
        assert len(y_test) == 20