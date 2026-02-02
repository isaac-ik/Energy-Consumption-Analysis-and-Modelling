#!/bin/bash

# Create directory structure
mkdir -p data/raw
mkdir -p data/processed
mkdir -p notebooks
mkdir -p src
mkdir -p models

# Create files
touch notebooks/energy_consumption.ipynb
touch src/preprocess.py
touch src/features.py
touch src/train_model.py
touch src/evaluate.py
touch requirements.txt
touch README.md

echo "Project structure created successfully."
