# API Reference

## Core Classes

### BaseAgent
Base class for all agents in the system.

```python
class BaseAgent(ABC):
    def __init__(self, name: str, model: str = None)
    def call_llm(self, messages: List[Dict], functions: Optional[List] = None)
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]  # Abstract
    def create_system_prompt(self) -> str
```

### DataProfilerAgent
Analyzes dataset characteristics and quality.

```python
class DataProfilerAgent(BaseAgent):
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]
    def _generate_statistical_profile(self, df: pd.DataFrame) -> Dict
    def _assess_data_quality(self, df: pd.DataFrame) -> Dict
    def _detect_outliers(self, df: pd.DataFrame) -> Dict
```

**Input**:
```python
{
    "dataset_path": "path/to/dataset.csv",
    "job_id": "optional_job_id"
}
```

**Output**:
```python
{
    "status": "profiling_complete",
    "dataset_shape": (rows, columns),
    "statistical_profile": {...},
    "quality_assessment": {...},
    "recommendations": [...]
}
```

### DataCleaningAgent  
Performs intelligent data cleaning operations.

```python
class DataCleaningAgent(BaseAgent):
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]
    def _create_cleaning_plan(self, df: pd.DataFrame, strategy: Dict) -> Dict
    def _apply_cleaning(self, df: pd.DataFrame, plan: Dict) -> tuple
    def _handle_missing_values(self, df: pd.DataFrame, step: Dict) -> tuple
```

**Input**:
```python
{
    "dataset_path": "path/to/dataset.csv",
    "cleaning_strategy": {
        "missing_values": "smart_imputation",
        "outliers": "cap",
        "duplicates": "remove"
    },
    "job_id": "optional_job_id"
}
```

**Output**:
```python
{
    "status": "cleaning_complete",
    "original_shape": (rows, columns),
    "cleaned_shape": (rows, columns),
    "output_path": "path/to/cleaned_dataset.csv",
    "cleaning_log": [...],
    "data_quality_improvement": {...}
}
```

## Pipeline Classes

### SimpleDataProcessingPipeline
Synchronous pipeline for basic processing.

```python
class SimpleDataProcessingPipeline:
    def __init__(self)
    def process_dataset(self, dataset_path: str, job_id: str = None) -> Dict
```

**Usage**:
```python
pipeline = SimpleDataProcessingPipeline()
results = pipeline.process_dataset("data/my_data.csv", "my_job")
```

## Configuration

### Settings
Application configuration management.

```python
class Settings:
    openai_api_key: str
    openai_model_primary: str = "gpt-4"
    openai_model_secondary: str = "gpt-3.5-turbo"
    max_retries: int = 3
    processing_timeout: int = 300
    log_level: str = "INFO"
```

### Environment Variables
- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `OPENAI_MODEL_PRIMARY`: Primary model (default: gpt-4)
- `OPENAI_MODEL_SECONDARY`: Secondary model (default: gpt-3.5-turbo)
- `MAX_RETRIES`: Maximum retry attempts (default: 3)
- `PROCESSING_TIMEOUT`: Processing timeout in seconds (default: 300)
- `LOG_LEVEL`: Logging level (default: INFO)

## Error Handling

All agents implement proper error handling:

```python
try:
    response = self.call_llm(messages)
    return response
except Exception as e:
    self.logger.error(f"LLM call failed: {str(e)}")
    raise
```

## Data Quality Metrics

### Completeness Score
```python
completeness = (1 - missing_cells / total_cells) * 100
```

### Uniqueness Score  
```python
uniqueness = (1 - duplicate_rows / total_rows) * 100
```

### Overall Quality Score
```python
overall_score = weighted_average([
    completeness * 0.3,
    consistency * 0.2, 
    validity * 0.2,
    uniqueness * 0.15,
    integrity * 0.15
])
```
