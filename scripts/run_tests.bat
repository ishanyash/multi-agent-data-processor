@echo off
echo Running Multi-Agent Data Processor Tests
echo ========================================

REM Activate virtual environment if it exists
if exist "venv" (
    call venv\Scripts\activate
    echo ✓ Virtual environment activated
)

REM Create test data
echo Creating test data...
python minimal_test.py

REM Run simple test with OpenAI
echo Running simple test with OpenAI...
python test_simple.py

REM Run unit tests
echo Running unit tests...
python -m pytest tests/ -v

echo All tests completed!
pause
