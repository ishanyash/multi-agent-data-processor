#!/bin/bash
echo "Running Multi-Agent Data Processor Tests"
echo "========================================"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✓ Virtual environment activated"
fi

# Create test data
echo "Creating test data..."
python minimal_test.py

# Run simple test with OpenAI
echo "Running simple test with OpenAI..."
python test_simple.py

# Run unit tests
echo "Running unit tests..."
python -m pytest tests/ -v

echo "All tests completed!"
