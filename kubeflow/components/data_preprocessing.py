"""
Kubeflow component for data preprocessing in house price prediction pipeline.
"""

from kfp import dsl
from kfp.dsl import component, Input, Output, Dataset, Metrics
from typing import NamedTuple
import yaml
import os


def load_pipeline_config():
    """Load pipeline configuration"""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'pipeline_config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# Load configuration
config = load_pipeline_config()
base_image = config['global']['base_image']
default_resources = config['global']['default_resources']


@component(
    base_image=base_image,
    packages_to_install=[
        "pandas==2.0.3",
        "numpy==1.24.3",
        "scikit-learn==1.3.0",
        "joblib==1.3.2",
        "pyyaml==6.0.1"
    ]
)
def data_preprocessing_component(
    # Input parameters
    n_samples: int = 1000,
    test_size: float = 0.2,
    validation_size: float = 0.2,
    random_state: int = 42,
    data_quality_threshold: float = 0.95,
    
    # Outputs
    processed_data: Output[Dataset],
    train_data: Output[Dataset],
    test_data: Output[Dataset],
    validation_data: Output[Dataset],
    preprocessing_metrics: Output[Metrics],
    data_schema: Output[Dataset]
    
) -> NamedTuple('PreprocessingOutput', [
    ('num_samples', int), 
    ('num_features', int),
    ('data_quality_score', float),
    ('processing_time', float)
]):
    """
    Comprehensive data preprocessing component for Kubeflow pipeline.
    
    Args:
        n_samples: Number of samples to generate (for synthetic data)
        test_size: Proportion of data for testing
        validation_size: Proportion of training data for validation
        random_state: Random seed for reproducibility
        data_quality_threshold: Minimum data quality score required
        
    Returns:
        NamedTuple with processing statistics
    """
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    import os
    import json
    import time
    from typing import NamedTuple
    import logging
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    start_time = time.time()
    
    logger.info("🔧 Starting data preprocessing component...")
    logger.info(f"Parameters: n_samples={n_samples}, test_size={test_size}")
    
    def generate_synthetic_data(n_samples: int, random_state: int) -> pd.DataFrame:
        """Generate synthetic house price data"""
        np.random.seed(random_state)
        
        logger.info(f"🏠 Generating {n_samples} synthetic house records...")
        
        # Generate house features
        area = np.random.normal(120, 40, n_samples)
        quartos = np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.1, 0.3, 0.35, 0.2, 0.05])
        banheiros = np.random.choice([1, 2, 3, 4], n_samples, p=[0.2, 0.5, 0.25, 0.05])
        idade = np.random.exponential(10, n_samples)
        garagem = np.random.choice([0, 1], n_samples, p=[0.3, 0.7])
        
        # Neighborhoods with different price multipliers
        bairros = ['Centro', 'Zona Sul', 'Zona Norte', 'Zona Oeste', 'Zona Leste']
        bairro_multiplier = {
            'Centro': 1.3, 'Zona Sul': 1.5, 'Zona Norte': 0.9, 
            'Zona Oeste': 1.1, 'Zona Leste': 0.8
        }
        bairro = np.random.choice(bairros, n_samples, p=[0.15, 0.25, 0.2, 0.2, 0.2])
        
        # Calculate base price
        preco_base = (
            area * 3000 +
            quartos * 15000 +
            banheiros * 10000 +
            garagem * 25000 -
            idade * 1000
        )
        
        # Apply neighborhood multiplier
        preco = np.array([preco_base[i] * bairro_multiplier[bairro[i]] for i in range(n_samples)])
        
        # Add noise
        preco += np.random.normal(0, 20000, n_samples)
        preco = np.abs(preco)
        
        # Create DataFrame
        df = pd.DataFrame({
            'area': area,
            'quartos': quartos,
            'banheiros': banheiros,
            'idade': idade,
            'garagem': garagem,
            'bairro': bairro,
            'preco': preco
        })
        
        return df
    
    def clean_data(df: pd.DataFrame) -> pd.DataFrame:
        """Clean and filter data"""
        logger.info("🧹 Cleaning data...")
        
        initial_count = len(df)
        
        # Remove outliers and invalid values
        df = df[df['area'] > 30]
        df = df[df['area'] < 500]
        df = df[df['preco'] > 50000]
        df = df[df['preco'] < 2000000]
        df = df[df['idade'] < 50]
        df = df[df['quartos'] >= 1]
        df = df[df['banheiros'] >= 1]
        
        # Reset index
        df = df.reset_index(drop=True)
        
        final_count = len(df)
        removed_count = initial_count - final_count
        
        logger.info(f"📊 Data cleaning results:")
        logger.info(f"  - Initial samples: {initial_count}")
        logger.info(f"  - Final samples: {final_count}")
        logger.info(f"  - Removed samples: {removed_count} ({removed_count/initial_count*100:.1f}%)")
        
        return df
    
    def calculate_data_quality(df: pd.DataFrame) -> float:
        """Calculate data quality score"""
        logger.info("📈 Calculating data quality score...")
        
        quality_checks = {
            'completeness': 1.0 - df.isnull().sum().sum() / (len(df) * len(df.columns)),
            'validity': len(df) / n_samples,  # Proportion of valid records after cleaning
            'consistency': 1.0,  # Assume consistent for synthetic data
            'accuracy': 1.0  # Assume accurate for synthetic data
        }
        
        # Weighted average
        weights = {'completeness': 0.3, 'validity': 0.4, 'consistency': 0.15, 'accuracy': 0.15}
        quality_score = sum(quality_checks[metric] * weights[metric] for metric in quality_checks)
        
        logger.info(f"📊 Data quality metrics:")
        for metric, score in quality_checks.items():
            logger.info(f"  - {metric.capitalize()}: {score:.3f}")
        logger.info(f"  - Overall quality score: {quality_score:.3f}")
        
        return quality_score
    
    def engineer_features(df: pd.DataFrame) -> tuple:
        """Apply feature engineering and encoding"""
        logger.info("🔧 Applying feature engineering...")
        
        data = df.copy()
        
        # Label encoding for categorical variables
        label_encoder = LabelEncoder()
        data['bairro_encoded'] = label_encoder.fit_transform(data['bairro'])
        
        # Feature engineering
        data['area_per_room'] = data['area'] / (data['quartos'] + 1)
        data['bathroom_ratio'] = data['banheiros'] / data['quartos']
        data['age_squared'] = data['idade'] ** 2
        data['total_rooms'] = data['quartos'] + data['banheiros']
        data['is_new'] = (data['idade'] < 5).astype(int)
        
        # Select features for modeling
        feature_columns = [
            'area', 'quartos', 'banheiros', 'idade', 'garagem', 'bairro_encoded',
            'area_per_room', 'bathroom_ratio', 'age_squared', 'total_rooms', 'is_new'
        ]
        
        X = data[feature_columns]
        y = data['preco']
        
        logger.info(f"✅ Feature engineering completed:")
        logger.info(f"  - Original features: {len(df.columns) - 1}")  # -1 for target
        logger.info(f"  - Engineered features: {len(feature_columns)}")
        logger.info(f"  - Feature names: {feature_columns}")
        
        return X, y, label_encoder, feature_columns
    
    def save_data_schema(feature_columns: list, label_encoder: LabelEncoder) -> dict:
        """Save data schema for validation"""
        logger.info("💾 Saving data schema...")
        
        schema = {
            'version': '1.0.0',
            'features': {
                'numeric_features': [
                    'area', 'quartos', 'banheiros', 'idade', 'garagem',
                    'area_per_room', 'bathroom_ratio', 'age_squared', 'total_rooms', 'is_new'
                ],
                'categorical_features': ['bairro_encoded'],
                'target': 'preco'
            },
            'feature_names': feature_columns,
            'encoders': {
                'bairro': {
                    'type': 'LabelEncoder',
                    'classes': label_encoder.classes_.tolist()
                }
            },
            'validation_rules': {
                'area': {'min': 30, 'max': 500},
                'quartos': {'min': 1, 'max': 10},
                'banheiros': {'min': 1, 'max': 6},
                'idade': {'min': 0, 'max': 50},
                'garagem': {'values': [0, 1]},
                'preco': {'min': 50000, 'max': 2000000}
            }
        }
        
        return schema
    
    # ====================================
    # Main Processing Logic
    # ====================================
    
    try:
        # Step 1: Generate or load data
        logger.info("📊 Step 1: Data Generation")
        df_raw = generate_synthetic_data(n_samples, random_state)
        
        # Step 2: Clean data
        logger.info("🧹 Step 2: Data Cleaning")
        df_clean = clean_data(df_raw)
        
        # Step 3: Calculate data quality
        logger.info("📈 Step 3: Data Quality Assessment")
        quality_score = calculate_data_quality(df_clean)
        
        if quality_score < data_quality_threshold:
            raise ValueError(f"Data quality score {quality_score:.3f} below threshold {data_quality_threshold}")
        
        # Step 4: Feature engineering
        logger.info("🔧 Step 4: Feature Engineering")
        X, y, label_encoder, feature_columns = engineer_features(df_clean)
        
        # Step 5: Split data
        logger.info("✂️ Step 5: Data Splitting")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        X_train_final, X_val, y_train_final, y_val = train_test_split(
            X_train, y_train, test_size=validation_size, random_state=random_state
        )
        
        logger.info(f"📊 Data split results:")
        logger.info(f"  - Training: {len(X_train_final)} samples")
        logger.info(f"  - Validation: {len(X_val)} samples")
        logger.info(f"  - Test: {len(X_test)} samples")
        
        # Step 6: Save datasets
        logger.info("💾 Step 6: Saving Datasets")
        
        # Save processed data
        processed_df = pd.concat([X, y], axis=1)
        processed_df.to_csv(processed_data.path, index=False)
        
        # Save train data
        train_df = pd.concat([X_train_final, y_train_final], axis=1)
        train_df.to_csv(train_data.path, index=False)
        
        # Save validation data
        val_df = pd.concat([X_val, y_val], axis=1)
        val_df.to_csv(validation_data.path, index=False)
        
        # Save test data
        test_df = pd.concat([X_test, y_test], axis=1)
        test_df.to_csv(test_data.path, index=False)
        
        # Save data schema
        schema = save_data_schema(feature_columns, label_encoder)
        with open(data_schema.path, 'w') as f:
            json.dump(schema, f, indent=2)
        
        # Step 7: Save metrics
        logger.info("📊 Step 7: Saving Metrics")
        processing_time = time.time() - start_time
        
        metrics_dict = {
            'data_quality_score': quality_score,
            'processing_time_seconds': processing_time,
            'total_samples': len(df_clean),
            'num_features': len(feature_columns),
            'train_samples': len(X_train_final),
            'validation_samples': len(X_val),
            'test_samples': len(X_test),
            'target_statistics': {
                'mean': float(y.mean()),
                'std': float(y.std()),
                'min': float(y.min()),
                'max': float(y.max())
            }
        }
        
        # Save metrics
        with open(preprocessing_metrics.path, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        
        logger.info(f"✅ Data preprocessing completed successfully!")
        logger.info(f"⏱️ Total processing time: {processing_time:.2f} seconds")
        
        # Return output tuple
        PreprocessingOutput = NamedTuple('PreprocessingOutput', [
            ('num_samples', int), 
            ('num_features', int),
            ('data_quality_score', float),
            ('processing_time', float)
        ])
        
        return PreprocessingOutput(
            num_samples=len(df_clean),
            num_features=len(feature_columns),
            data_quality_score=quality_score,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"❌ Error in data preprocessing: {str(e)}")
        raise


# Additional helper component for data validation
@component(
    base_image=base_image,
    packages_to_install=["pandas==2.0.3", "numpy==1.24.3", "jsonschema==4.19.1"]
)
def data_validation_component(
    dataset: Input[Dataset],
    schema: Input[Dataset],
    validation_report: Output[Metrics]
) -> NamedTuple('ValidationOutput', [('is_valid', bool), ('validation_score', float)]):
    """
    Validate dataset against schema.
    
    Args:
        dataset: Input dataset to validate
        schema: Data schema definition
        validation_report: Output validation metrics
        
    Returns:
        NamedTuple with validation results
    """
    import pandas as pd
    import json
    import numpy as np
    from typing import NamedTuple
    import logging
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("🔍 Starting data validation...")
    
    try:
        # Load data and schema
        df = pd.read_csv(dataset.path)
        with open(schema.path, 'r') as f:
            schema_def = json.load(f)
        
        validation_results = {
            'schema_compliance': True,
            'feature_validation': {},
            'overall_score': 0.0
        }
        
        # Validate features exist
        expected_features = schema_def['feature_names']
        missing_features = [f for f in expected_features if f not in df.columns]
        
        if missing_features:
            logger.error(f"Missing features: {missing_features}")
            validation_results['schema_compliance'] = False
        
        # Validate feature ranges
        validation_rules = schema_def.get('validation_rules', {})
        feature_scores = []
        
        for feature, rules in validation_rules.items():
            if feature in df.columns:
                if 'min' in rules and 'max' in rules:
                    valid_count = ((df[feature] >= rules['min']) & (df[feature] <= rules['max'])).sum()
                    score = valid_count / len(df)
                    validation_results['feature_validation'][feature] = {
                        'score': score,
                        'valid_samples': int(valid_count),
                        'total_samples': len(df)
                    }
                    feature_scores.append(score)
                    logger.info(f"Feature {feature}: {score:.3f} valid")
        
        # Calculate overall score
        overall_score = np.mean(feature_scores) if feature_scores else 0.0
        validation_results['overall_score'] = overall_score
        
        is_valid = validation_results['schema_compliance'] and overall_score >= 0.95
        
        # Save validation report
        with open(validation_report.path, 'w') as f:
            json.dump(validation_results, f, indent=2)
        
        logger.info(f"✅ Validation completed. Score: {overall_score:.3f}, Valid: {is_valid}")
        
        ValidationOutput = NamedTuple('ValidationOutput', [('is_valid', bool), ('validation_score', float)])
        return ValidationOutput(is_valid=is_valid, validation_score=overall_score)
        
    except Exception as e:
        logger.error(f"❌ Validation error: {str(e)}")
        raise