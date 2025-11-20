#!/bin/bash
# Setup script for repository structure

echo "================================"
echo "Setting up repository structure"
echo "================================"

# Create __init__.py files
echo "Creating package __init__.py files..."
touch src/__init__.py
touch src/models/__init__.py
touch src/data/__init__.py
touch src/training/__init__.py
touch src/analysis/__init__.py
touch src/utils/__init__.py
touch tests/__init__.py

# Create directories
echo "Creating directories..."
mkdir -p data
mkdir -p outputs/figures
mkdir -p outputs/checkpoints
mkdir -p outputs/logs
mkdir -p notebooks
mkdir -p docs

# Create placeholder READMEs
echo "# Data Directory" > data/README.md
echo "Place your genomic FASTA files here." >> data/README.md

echo "# Notebooks" > notebooks/README.md
echo "Jupyter notebooks for interactive analysis." >> notebooks/README.md

echo "✓ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. pip install -r requirements.txt"
echo "  2. pip install -e ."
echo "  3. python scripts/train.py --help"