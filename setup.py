from setuptools import setup, find_packages

setup(
    name="mlops-house-price",
    version="1.0.0",
    description="MLOps End-to-End House Price Prediction",
    author="MLOps Team",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "scikit-learn==1.3.0",
        "pandas==2.0.3",
        "numpy==1.24.3",
        "fastapi==0.103.1",
        "uvicorn==0.23.2",
        "mlflow==2.5.0",
        "pydantic==2.3.0",
        "joblib==1.3.2",
        "python-dotenv==1.0.0",
        "pyyaml==6.0.1",
    ],
    extras_require={
        "dev": ["pytest", "black", "flake8"],
        "monitoring": ["prometheus-client", "grafana-api"],
    },
)