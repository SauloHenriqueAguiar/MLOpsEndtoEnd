#!/bin/bash
set -e

# MLOps API Entrypoint Script

echo "Starting MLOps House Price Prediction API..."

# Wait for dependencies (if needed)
if [ -n "$WAIT_FOR_SERVICES" ]; then
    echo "Waiting for services to be ready..."
    sleep 10
fi

# Check if model exists
if [ ! -f "${MODEL_PATH}/random_forest_model.pkl" ]; then
    echo "WARNING: Model file not found at ${MODEL_PATH}/random_forest_model.pkl"
    echo "Make sure to train and save the model first"
fi

# Create necessary directories
mkdir -p /app/logs
mkdir -p /app/data/models

# Set proper permissions
chown -R appuser:appuser /app/logs

# Start the application
exec "$@"