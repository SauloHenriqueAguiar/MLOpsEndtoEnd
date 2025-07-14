#!/usr/bin/env python3
"""
Script to generate synthetic data for testing.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_loader import DataLoader
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic house price data")
    parser.add_argument("--samples", type=int, default=1000, help="Number of samples to generate")
    parser.add_argument("--output", default="house_prices.csv", help="Output filename")
    
    args = parser.parse_args()
    
    # Initialize data loader
    loader = DataLoader()
    
    # Generate synthetic data
    logger.info(f"Generating {args.samples} synthetic samples...")
    df = loader.generate_synthetic_data(args.samples)
    
    # Save data
    loader.save_data(df, args.output)
    
    logger.info(f"Data saved to data/raw/{args.output}")
    logger.info(f"Data shape: {df.shape}")
    logger.info(f"Price range: ${df['preco'].min():,.2f} - ${df['preco'].max():,.2f}")


if __name__ == "__main__":
    main()