"""
Data preprocessing utilities for house price prediction.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Handle data preprocessing tasks."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.imputer = SimpleImputer(strategy='median')
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare data."""
        df = df.copy()
        
        # Remove outliers
        for col in ['area', 'preco']:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            df = df[(df[col] >= Q1 - 1.5*IQR) & (df[col] <= Q3 + 1.5*IQR)]
        
        # Fill missing values
        numeric_cols = ['area', 'quartos', 'banheiros', 'idade', 'garagem']
        df[numeric_cols] = self.imputer.fit_transform(df[numeric_cols])
        
        return df.reset_index(drop=True)
    
    def encode_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features."""
        df = df.copy()
        df['bairro_encoded'] = self.label_encoder.fit_transform(df['bairro'])
        return df.drop('bairro', axis=1)
    
    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple:
        """Scale numerical features."""
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return (pd.DataFrame(X_train_scaled, columns=X_train.columns),
                pd.DataFrame(X_test_scaled, columns=X_test.columns))
    
    def split_data(self, df: pd.DataFrame, target_col: str = 'preco', 
                   test_size: float = 0.2) -> tuple:
        """Split data into train/test sets."""
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        
        return train_test_split(X, y, test_size=test_size, random_state=42)
    
    def process_data(self, df: pd.DataFrame, save_path: str = "data/processed") -> tuple:
        """Complete preprocessing pipeline."""
        # Clean data
        df_clean = self.clean_data(df)
        
        # Encode features
        df_encoded = self.encode_features(df_clean)
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(df_encoded)
        
        # Scale features
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        
        # Save processed data
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        X_train_scaled.to_csv(save_dir / "X_train.csv", index=False)
        X_test_scaled.to_csv(save_dir / "X_test.csv", index=False)
        y_train.to_csv(save_dir / "y_train.csv", index=False)
        y_test.to_csv(save_dir / "y_test.csv", index=False)
        
        logger.info(f"Processed data saved to {save_dir}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test


def main():
    """Test preprocessing functionality."""
    from data_loader import DataLoader
    
    loader = DataLoader()
    preprocessor = DataPreprocessor()
    
    # Load data
    try:
        df = loader.load_raw_data()
    except FileNotFoundError:
        df = loader.generate_synthetic_data()
    
    # Process data
    X_train, X_test, y_train, y_test = preprocessor.process_data(df)
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")


if __name__ == "__main__":
    main()