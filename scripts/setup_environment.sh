#!/bin/bash
echo "Setting up Multi-Agent Data Processor Environment"
echo "================================================"

# Create virtual environment
echo "Creating virtual environment..."
python -m venv venv

# Activate virtual environment
source venv/bin/activate
echo "✓ Virtual environment activated"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
if [ -f "requirements_minimal.txt" ]; then
    echo "Installing minimal requirements first..."
    pip install -r requirements_minimal.txt
else
    echo "Installing from requirements.txt..."
    pip install -r requirements.txt
fi

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file template..."
    cp .env.template .env
    echo "Please edit .env file and add your OpenAI API key"
fi

echo "Setup completed!"
echo "Next steps:"
echo "1. Edit .env file and add your OpenAI API key"
echo "2. Run: python minimal_test.py"
echo "3. Run: python test_simple.py"
