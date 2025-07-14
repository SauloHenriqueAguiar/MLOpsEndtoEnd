"""
Data loading utilities for house price prediction.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class DataLoader:
    """Handle data loading operations."""
    
    def __init__(self, data_path: str = "data/raw"):
        self.data_path = Path(data_path)
        
    def load_raw_data(self, filename: str = "house_prices.csv") -> pd.DataFrame:
        """Load raw data from CSV file."""
        file_path = self.data_path / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} records from {file_path}")
        return df
    
    def generate_synthetic_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """Generate synthetic house price data for testing."""
        np.random.seed(42)
        
        # Generate features
        areas = np.random.normal(120, 40, n_samples)
        areas = np.clip(areas, 30, 500)
        
        quartos = np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.1, 0.2, 0.4, 0.2, 0.1])
        banheiros = np.random.choice([1, 2, 3, 4], n_samples, p=[0.2, 0.5, 0.2, 0.1])
        idades = np.random.exponential(10, n_samples)
        idades = np.clip(idades, 0, 50)
        garagem = np.random.choice([0, 1], n_samples, p=[0.3, 0.7])
        bairros = np.random.choice(
            ['Centro', 'Zona Sul', 'Zona Norte', 'Zona Oeste', 'Zona Leste'],
            n_samples, p=[0.15, 0.25, 0.2, 0.2, 0.2]
        )
        
        # Generate prices with realistic relationships
        base_price = 2000 * areas + 50000 * quartos + 30000 * banheiros
        age_discount = 5000 * idades
        garage_bonus = 25000 * garagem
        
        # Neighborhood multipliers
        neighborhood_multipliers = {
            'Centro': 1.2, 'Zona Sul': 1.5, 'Zona Norte': 0.9,
            'Zona Oeste': 1.0, 'Zona Leste': 0.8
        }
        
        precos = []
        for i in range(n_samples):
            multiplier = neighborhood_multipliers[bairros[i]]
            price = (base_price[i] - age_discount[i] + garage_bonus[i]) * multiplier
            price += np.random.normal(0, 20000)  # Add noise
            precos.append(max(price, 50000))  # Minimum price
        
        df = pd.DataFrame({
            'area': areas,
            'quartos': quartos,
            'banheiros': banheiros,
            'idade': idades,
            'garagem': garagem,
            'bairro': bairros,
            'preco': precos
        })
        
        logger.info(f"Generated {len(df)} synthetic records")
        return df
    
    def save_data(self, df: pd.DataFrame, filename: str, path: Optional[str] = None) -> None:
        """Save DataFrame to CSV file."""
        if path is None:
            path = self.data_path
        else:
            path = Path(path)
        
        path.mkdir(parents=True, exist_ok=True)
        file_path = path / filename
        df.to_csv(file_path, index=False)
        logger.info(f"Data saved to {file_path}")


def main():
    """Generate and save synthetic data."""
    loader = DataLoader()
    df = loader.generate_synthetic_data(1000)
    loader.save_data(df, "house_prices.csv")
    print(f"Generated data shape: {df.shape}")
    print(df.head())


if __name__ == "__main__":
    main()