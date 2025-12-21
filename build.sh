#!/bin/bash
echo "Building AI Trading Bot..."

# Create directories
mkdir -p app/data

# Install dependencies
pip install -r requirements.txt

# Set permissions
chmod +x *.py

echo "Build complete!"
echo "To start: python main.py"
