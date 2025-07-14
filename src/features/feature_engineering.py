import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

class FeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def preprocess_data(self, df, target_column):
        """Basic preprocessing: handle missing values and encode categoricals"""
        df = df.fillna(df.mean(numeric_only=True))
        
        for col in df.select_dtypes(include=['object']).columns:
            if col != target_column:
                df[col] = self.label_encoder.fit_transform(df[col].astype(str))
                
        return df
    
    def scale_features(self, X_train, X_test):
        """Scale numerical features"""
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled
    
    def split_data(self, df, target_column, test_size=0.2):
        """Split data into train/test sets"""
        X = df.drop(columns=[target_column])
        y = df[target_column]
        return train_test_split(X, y, test_size=test_size, random_state=42)