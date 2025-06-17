import pandas as pd
import numpy as np
from pathlib import Path
import json
from main_simple import SimpleDataProcessingPipeline

def create_test_dataset():
    """Create a simple test dataset with common data issues"""
    np.random.seed(42)
    n_rows = 200
    
    # Create sample data with issues
    data = {
        'customer_id': range(1, n_rows + 1),
        'name': [f'Customer_{i}' if i % 20 != 0 else None for i in range(1, n_rows + 1)],
        'age': [np.random.randint(18, 80) if i % 15 != 0 else None for i in range(1, n_rows + 1)],
        'income': [np.random.normal(50000, 15000) if i % 25 != 0 else None for i in range(1, n_rows + 1)],
        'city': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston'], n_rows),
        'category': np.random.choice(['Premium', 'Standard', 'Basic'], n_rows)
    }
    
    df = pd.DataFrame(data)
    
    # Add duplicates
    df = pd.concat([df, df.iloc[:10]], ignore_index=True)
    
    # Add outliers
    df.loc[0, 'income'] = 500000  # Outlier
    df.loc[1, 'age'] = 150  # Impossible age
    
    # Ensure data directory exists
    Path('data').mkdir(exist_ok=True)
    df.to_csv('data/test_dataset.csv', index=False)
    print("Created test dataset: data/test_dataset.csv")
    return 'data/test_dataset.csv'

def run_simple_test():
    """Run a simple test of the pipeline"""
    print("=" * 60)
    print("SIMPLE MULTI-AGENT DATA PROCESSOR TEST")
    print("=" * 60)
    
    # Create test dataset
    dataset_path = create_test_dataset()
    
    # Initialize pipeline
    pipeline = SimpleDataProcessingPipeline()
    
    try:
        # Run processing
        print("\nRunning data processing pipeline...")
        results = pipeline.process_dataset(dataset_path, job_id="simple_test")
        
        # Display results
        print("\n" + "=" * 40)
        print("RESULTS SUMMARY")
        print("=" * 40)
        print(f"✅ Processing Status: {results['status']}")
        print(f"📁 Input Dataset: {results['input_dataset']}")
        print(f"📁 Output Dataset: {results['output_dataset']}")
        print(f"📊 Quality Improvement: {results['quality_improvement']}")
        
        # Show some quality metrics
        quality_improvement = results['quality_improvement']
        print(f"\n📈 Quality Improvements:")
        print(f"   • Missing values reduced: {quality_improvement.get('missing_values_reduced', 0)}")
        print(f"   • Duplicates removed: {quality_improvement.get('duplicates_removed', 0)}")
        print(f"   • Data completeness: {quality_improvement.get('data_completeness', 0):.1f}%")
        print(f"   • Rows retained: {quality_improvement.get('rows_retained', 0):.1f}%")
        
        # Show recommendations
        recommendations = results.get('recommendations', [])
        if recommendations:
            print(f"\n💡 Recommendations:")
            for i, rec in enumerate(recommendations[:3], 1):
                print(f"   {i}. {rec}")
        
        print(f"\n📋 Detailed results saved to: output/processing_results_{results['job_id']}.json")
        
        # Load and verify the cleaned dataset
        cleaned_df = pd.read_csv(results['output_dataset'])
        original_df = pd.read_csv(dataset_path)
        
        print(f"\n🔍 Dataset Comparison:")
        print(f"   • Original shape: {original_df.shape}")
        print(f"   • Cleaned shape: {cleaned_df.shape}")
        print(f"   • Original missing values: {original_df.isnull().sum().sum()}")
        print(f"   • Cleaned missing values: {cleaned_df.isnull().sum().sum()}")
        print(f"   • Original duplicates: {original_df.duplicated().sum()}")
        print(f"   • Cleaned duplicates: {cleaned_df.duplicated().sum()}")
        
        print("\n🎉 Test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        print("This might be due to:")
        print("1. Missing OpenAI API key in .env file")
        print("2. Network connectivity issues")
        print("3. OpenAI API rate limits")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Ensure output directory exists
    Path('output').mkdir(exist_ok=True)
    
    success = run_simple_test()
    if success:
        print("\n✅ All tests passed! Your setup is working correctly.")
    else:
        print("\n❌ Tests failed. Please check the error messages above.")
