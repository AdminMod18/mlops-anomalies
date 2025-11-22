#!/bin/bash

# MLOps Environment Setup Script

set -e

echo "🚀 Setting up MLOps Anomaly Detection environment..."

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo "❌ Error: Run this script from the project root."
    exit 1
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

# Install dependencies
echo "📦 Installing dependencies..."
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# Create directories
echo "📁 Creating directories..."
mkdir -p data/raw data/processed models mlruns plots logs config

# Generate data
echo "🔄 Generating synthetic data..."
python -c "from src.utils import generate_synthetic_data; generate_synthetic_data('data/raw/anomaly_data.csv', n_samples=10000)"

# Preprocess
echo "🔄 Preprocessing data..."
python src/preprocess.py

# Train
echo "🤖 Training model..."
python src/train.py

echo ""
echo "✅ Setup complete!"
echo ""
echo "To start the API: make api"
echo "To run tests: make test"
