"""
Minimal test script that works without OpenAI API
Tests basic data processing functionality
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json

class MinimalDataProcessor:
    def __init__(self):
        self.name = "MinimalProcessor"
    
    def profile_data(self, df):
        """Basic data profiling without LLM"""
        profile = {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "missing_percentage": (df.isnull().sum() / len(df) * 100).to_dict(),
            "duplicates": df.duplicated().sum(),
            "duplicate_percentage": (df.duplicated().sum() / len(df)) * 100
        }
        
        # Basic recommendations
        recommendations = []
        if profile["duplicate_percentage"] > 5:
            recommendations.append("Remove duplicate rows")
        if max(profile["missing_percentage"].values()) > 20:
            recommendations.append("Handle missing values")
        if any(df.select_dtypes(include=[np.number]).columns):
            recommendations.append("Check for outliers in numerical columns")
        
        return {
            "status": "profiling_complete",
            "profile": profile,
            "recommendations": recommendations
        }
    
    def clean_data(self, df):
        """Basic data cleaning without LLM"""
        # Create a copy to avoid SettingWithCopyWarning
        df = df.copy()
        original_shape = df.shape
        cleaning_log = []
        
        # Remove duplicates
        initial_count = len(df)
        df = df.drop_duplicates()
        duplicates_removed = initial_count - len(df)
        if duplicates_removed > 0:
            cleaning_log.append(f"Removed {duplicates_removed} duplicate rows")
        
        # Handle missing values
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                if df[col].dtype in ['int64', 'float64']:
                    # Fill numeric with mean
                    mean_val = df[col].mean()
                    df[col].fillna(mean_val, inplace=True)
                    cleaning_log.append(f"Filled {missing_count} missing values in {col} with mean: {mean_val:.2f}")
                else:
                    # Fill categorical with mode
                    mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else "Unknown"
                    df[col].fillna(mode_val, inplace=True)
                    cleaning_log.append(f"Filled {missing_count} missing values in {col} with mode: {mode_val}")
        
        # Basic outlier handling for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            outlier_count = outlier_mask.sum()
            
            if outlier_count > 0:
                # Cap outliers
                df.loc[df[col] < lower_bound, col] = lower_bound
                df.loc[df[col] > upper_bound, col] = upper_bound
                cleaning_log.append(f"Capped {outlier_count} outliers in {col}")
        
        return {
            "cleaned_df": df,
            "original_shape": original_shape,
            "cleaned_shape": df.shape,
            "cleaning_log": cleaning_log
        }
    
    def process_dataset(self, dataset_path, job_id="minimal_test"):
        """Complete processing pipeline"""
        print(f"Starting minimal processing for: {dataset_path}")
        
        # Load data
        df = pd.read_csv(dataset_path)
        print(f"Loaded dataset with shape: {df.shape}")
        
        # Profile data
        profiling_result = self.profile_data(df)
        print("Data profiling completed")
        
        # Clean data
        cleaning_result = self.clean_data(df)
        print("Data cleaning completed")
        
        # Save cleaned data
        output_path = dataset_path.replace('.csv', '_minimal_cleaned.csv')
        cleaning_result["cleaned_df"].to_csv(output_path, index=False)
        
        # Calculate improvements
        original_missing = df.isnull().sum().sum()
        cleaned_missing = cleaning_result["cleaned_df"].isnull().sum().sum()
        original_duplicates = df.duplicated().sum()
        cleaned_duplicates = cleaning_result["cleaned_df"].duplicated().sum()
        
        results = {
            "job_id": job_id,
            "status": "completed",
            "input_dataset": dataset_path,
            "output_dataset": output_path,
            "profiling": profiling_result,
            "cleaning": cleaning_result,
            "improvements": {
                "missing_values_reduced": int(original_missing - cleaned_missing),
                "duplicates_removed": int(original_duplicates - cleaned_duplicates),
                "data_completeness": (1 - cleaned_missing / (cleaning_result["cleaned_df"].shape[0] * cleaning_result["cleaned_df"].shape[1])) * 100,
                "rows_retained": (cleaning_result["cleaned_df"].shape[0] / df.shape[0]) * 100
            }
        }
        
        # Save results
        results_path = f"output/minimal_results_{job_id}.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        return results

def create_test_data():
    """Create test dataset"""
    np.random.seed(42)
    data = {
        'id': range(1, 101),
        'name': [f'Person_{i}' if i % 10 != 0 else None for i in range(1, 101)],
        'age': [np.random.randint(18, 80) if i % 15 != 0 else None for i in range(1, 101)],
        'salary': [np.random.normal(50000, 15000) if i % 20 != 0 else None for i in range(1, 101)],
        'department': np.random.choice(['IT', 'HR', 'Finance'], 100)
    }
    
    df = pd.DataFrame(data)
    # Add duplicates
    df = pd.concat([df, df.iloc[:5]], ignore_index=True)
    # Add outliers
    df.loc[0, 'salary'] = 500000
    df.loc[1, 'age'] = 150
    
    Path('data').mkdir(exist_ok=True)
    df.to_csv('data/minimal_test.csv', index=False)
    return 'data/minimal_test.csv'

def main():
    """Run minimal test"""
    print("=" * 50)
    print("MINIMAL DATA PROCESSOR TEST")
    print("=" * 50)
    
    # Create directories
    Path('data').mkdir(exist_ok=True)
    Path('output').mkdir(exist_ok=True)
    
    # Create test data
    dataset_path = create_test_data()
    print(f"Created test dataset: {dataset_path}")
    
    # Process data
    processor = MinimalDataProcessor()
    results = processor.process_dataset(dataset_path)
    
    # Display results
    print("\n" + "=" * 30)
    print("PROCESSING RESULTS")
    print("=" * 30)
    print(f"Status: {results['status']}")
    print(f"Input: {results['input_dataset']}")
    print(f"Output: {results['output_dataset']}")
    
    improvements = results['improvements']
    print(f"\nImprovements:")
    print(f"  Missing values reduced: {improvements['missing_values_reduced']}")
    print(f"  Duplicates removed: {improvements['duplicates_removed']}")
    print(f"  Data completeness: {improvements['data_completeness']:.1f}%")
    print(f"  Rows retained: {improvements['rows_retained']:.1f}%")
    
    print(f"\nCleaning actions:")
    for action in results['cleaning']['cleaning_log']:
        print(f"  • {action}")
    
    print(f"\nRecommendations:")
    for rec in results['profiling']['recommendations']:
        print(f"  • {rec}")
    
    print(f"\n✅ Minimal test completed successfully!")
    print(f"Results saved to: output/minimal_results_{results['job_id']}.json")

if __name__ == "__main__":
    main()
