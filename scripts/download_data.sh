#!/bin/bash
# Script to download TrackML dataset
# Note: You need to have Kaggle API credentials set up

echo "Downloading TrackML dataset..."

# Create data directory
mkdir -p data/raw

# Navigate to data directory
cd data/raw

# Download using Kaggle API
# First, make sure you have kaggle installed: pip install kaggle
# And have your API credentials in ~/.kaggle/kaggle.json

echo "Downloading training data..."
kaggle competitions download -c trackml-particle-identification -f train_1.zip
kaggle competitions download -c trackml-particle-identification -f train_2.zip
kaggle competitions download -c trackml-particle-identification -f train_3.zip
kaggle competitions download -c trackml-particle-identification -f train_4.zip
kaggle competitions download -c trackml-particle-identification -f train_5.zip

echo "Downloading detector geometry..."
kaggle competitions download -c trackml-particle-identification -f detectors.zip

echo "Extracting files..."
unzip -q train_1.zip
unzip -q train_2.zip
unzip -q train_3.zip
unzip -q train_4.zip
unzip -q train_5.zip
unzip -q detectors.zip

echo "Cleaning up zip files..."
rm *.zip

echo "Download complete!"
echo "Now run: python scripts/preprocess_data.py --data_dir data/raw --output_dir data/processed"
