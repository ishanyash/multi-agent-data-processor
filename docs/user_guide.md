# User Guide

## Quick Start

1. **Setup Environment**
   ```bash
   source scripts/setup_environment.sh
   ```

2. **Configure API Key**
   ```bash
   echo "OPENAI_API_KEY=your_key_here" > .env
   ```

3. **Run Test**
   ```bash
   python minimal_test.py
   ```

4. **Process Your Data**
   ```bash
   python main_simple.py --dataset your_data.csv
   ```

## Features

- Intelligent data profiling
- Smart missing value handling
- Outlier detection and treatment
- Duplicate removal
- Data type optimization
- Quality assessment and reporting

## Usage Examples

### Basic Processing
```python
from main_simple import SimpleDataProcessingPipeline

pipeline = SimpleDataProcessingPipeline()
results = pipeline.process_dataset("data/my_data.csv")
print(f"Quality improved by {results['quality_improvement']['total_improvement']:.1f} points")
```

### Command Line Usage
```bash
# Basic processing
python main_simple.py --dataset data/sales_data.csv

# With custom job ID
python main_simple.py --dataset data/sales_data.csv --job-id sales_cleanup_2025

# Custom output directory
python main_simple.py --dataset data/sales_data.csv --output-dir results/
```

## Output

- Cleaned dataset
- Quality report
- Processing log
- Recommendations

## Configuration

Edit `.env` file:
```
OPENAI_API_KEY=your_key_here
OPENAI_MODEL_PRIMARY=gpt-4
OPENAI_MODEL_SECONDARY=gpt-3.5-turbo
```

## Troubleshooting

### Common Issues
- **API Key Error**: Check `.env` file configuration
- **Import Errors**: Install dependencies with `pip install -r requirements_minimal.txt`
- **Permission Errors**: Ensure write access to `output/` directory

### Getting Help
- Run `python minimal_test.py` first to test basic functionality
- Check logs for detailed error messages
- Review output files for processing details
