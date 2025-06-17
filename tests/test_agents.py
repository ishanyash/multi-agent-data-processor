import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from agents.data_profiler_simple import DataProfilerAgent
from agents.data_cleaning_simple import DataCleaningAgent

class TestAgents:
    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset for testing"""
        np.random.seed(42)
        data = {
            'id': range(1, 101),
            'name': [f'Person_{i}' if i % 10 != 0 else None for i in range(1, 101)],
            'age': [np.random.randint(18, 80) if i % 15 != 0 else None for i in range(1, 101)],
            'salary': [np.random.normal(50000, 15000) if i % 20 != 0 else None for i in range(1, 101)],
            'department': np.random.choice(['IT', 'HR', 'Finance', 'Marketing'], 100)
        }
        
        # Add some duplicates
        df = pd.DataFrame(data)
        df = pd.concat([df, df.iloc[:5]], ignore_index=True)
        
        # Add some outliers
        df.loc[0, 'salary'] = 500000  # Outlier
        df.loc[1, 'age'] = 150  # Outlier
        
        return df
    
    def test_data_profiler_basic(self, sample_dataset):
        """Test data profiler agent without OpenAI"""
        # Save sample dataset
        Path('data').mkdir(exist_ok=True)
        sample_dataset.to_csv('data/test_dataset.csv', index=False)
        
        profiler = DataProfilerAgent()
        
        # Test the statistical profiling without LLM
        stats_profile = profiler._generate_statistical_profile(sample_dataset)
        quality_assessment = profiler._assess_data_quality(sample_dataset)
        
        assert "shape" in stats_profile
        assert "columns" in stats_profile
        assert "completeness" in quality_assessment
        assert "consistency" in quality_assessment
    
    def test_data_cleaner_basic(self, sample_dataset):
        """Test data cleaning agent without OpenAI"""
        # Save sample dataset
        Path('data').mkdir(exist_ok=True)
        sample_dataset.to_csv('data/test_dataset.csv', index=False)
        
        cleaner = DataCleaningAgent()
        
        # Test basic cleaning functionality
        issues = cleaner._identify_issues(sample_dataset)
        
        assert "missing_values" in issues
        assert "duplicates" in issues
        assert "outliers" in issues
        
        # Test duplicate removal
        cleaned_df, log_entry = cleaner._remove_duplicates(sample_dataset.copy(), {})
        assert len(cleaned_df) <= len(sample_dataset)
        assert "remove_duplicates" in log_entry["step"]

if __name__ == "__main__":
    pytest.main([__file__])
