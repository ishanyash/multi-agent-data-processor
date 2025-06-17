#!/usr/bin/env python3
"""
Enhanced Test Data Generator for Multi-Agent Data Processor
Creates test datasets in multiple formats with various data quality issues
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import random
import os
from pathlib import Path

def create_comprehensive_test_dataset():
    """Create a comprehensive test dataset with various data quality issues"""
    
    np.random.seed(42)
    random.seed(42)
    
    # Base data size
    n_rows = 1000
    
    # Generate base data
    data = {
        'customer_id': range(1, n_rows + 1),
        'name': [f"Customer_{i}" for i in range(1, n_rows + 1)],
        'email': [f"customer{i}@email.com" for i in range(1, n_rows + 1)],
        'age': np.random.randint(18, 80, n_rows),
        'income': np.random.normal(50000, 20000, n_rows),
        'purchase_amount': np.random.exponential(100, n_rows),
        'registration_date': [
            (datetime.now() - timedelta(days=random.randint(1, 1000))).strftime('%Y-%m-%d')
            for _ in range(n_rows)
        ],
        'category': np.random.choice(['Premium', 'Standard', 'Basic'], n_rows, p=[0.2, 0.5, 0.3]),
        'is_active': np.random.choice([True, False], n_rows, p=[0.8, 0.2]),
        'score': np.random.uniform(0, 100, n_rows)
    }
    
    df = pd.DataFrame(data)
    
    # Introduce data quality issues
    
    # 1. Missing values (various patterns)
    # Random missing in age (5%)
    missing_age_idx = np.random.choice(df.index, size=int(0.05 * n_rows), replace=False)
    df.loc[missing_age_idx, 'age'] = np.nan
    
    # Missing income for younger customers (pattern-based missing)
    young_customers = df[df['age'] < 25].index
    missing_income_idx = np.random.choice(young_customers, size=int(0.3 * len(young_customers)), replace=False)
    df.loc[missing_income_idx, 'income'] = np.nan
    
    # Missing email (10%)
    missing_email_idx = np.random.choice(df.index, size=int(0.1 * n_rows), replace=False)
    df.loc[missing_email_idx, 'email'] = np.nan
    
    # Missing purchase_amount (15%)
    missing_purchase_idx = np.random.choice(df.index, size=int(0.15 * n_rows), replace=False)
    df.loc[missing_purchase_idx, 'purchase_amount'] = np.nan
    
    # 2. Duplicate rows (5%)
    duplicate_rows = df.sample(int(0.05 * n_rows))
    df = pd.concat([df, duplicate_rows], ignore_index=True)
    
    # 3. Outliers
    # Income outliers (very high values)
    outlier_idx = np.random.choice(df.index, size=20, replace=False)
    df.loc[outlier_idx, 'income'] = np.random.uniform(200000, 500000, 20)
    
    # Purchase amount outliers (very high values)
    outlier_purchase_idx = np.random.choice(df.index, size=15, replace=False)
    df.loc[outlier_purchase_idx, 'purchase_amount'] = np.random.uniform(5000, 20000, 15)
    
    # Age outliers (unrealistic ages)
    outlier_age_idx = np.random.choice(df.index, size=5, replace=False)
    df.loc[outlier_age_idx, 'age'] = np.random.uniform(150, 200, 5)
    
    # 4. Data type issues
    # Mix numeric and string in income column
    string_income_idx = np.random.choice(df.dropna(subset=['income']).index, size=10, replace=False)
    df.loc[string_income_idx, 'income'] = df.loc[string_income_idx, 'income'].astype(str) + 'k'
    
    # 5. Consistency issues
    # Inconsistent category naming
    inconsistent_idx = np.random.choice(df.index, size=30, replace=False)
    df.loc[inconsistent_idx, 'category'] = df.loc[inconsistent_idx, 'category'].str.lower()
    
    # Add some with extra spaces
    space_idx = np.random.choice(df.index, size=20, replace=False)
    df.loc[space_idx, 'category'] = ' ' + df.loc[space_idx, 'category'] + ' '
    
    # 6. Add some text data with quality issues
    df['description'] = [
        f"Customer description {i}" if i % 3 != 0 else f"  CUSTOMER DESCRIPTION {i}  "
        for i in range(len(df))
    ]
    
    # Add some completely different descriptions
    weird_idx = np.random.choice(df.index, size=50, replace=False)
    df.loc[weird_idx, 'description'] = [
        random.choice(['N/A', 'NULL', 'TBD', '---', 'UNKNOWN', ''])
        for _ in range(50)
    ]
    
    return df

def save_in_multiple_formats(df, base_name="comprehensive_test_data"):
    """Save dataset in multiple formats"""
    
    # Ensure data directory exists
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    formats_saved = []
    
    # 1. CSV
    csv_path = data_dir / f"{base_name}.csv"
    df.to_csv(csv_path, index=False)
    formats_saved.append(("CSV", str(csv_path)))
    
    # 2. JSON
    json_path = data_dir / f"{base_name}.json"
    df.to_json(json_path, orient='records', indent=2)
    formats_saved.append(("JSON", str(json_path)))
    
    # 3. Excel
    excel_path = data_dir / f"{base_name}.xlsx"
    df.to_excel(excel_path, index=False, engine='openpyxl')
    formats_saved.append(("Excel", str(excel_path)))
    
    # 4. TSV
    tsv_path = data_dir / f"{base_name}.tsv"
    df.to_csv(tsv_path, sep='\t', index=False)
    formats_saved.append(("TSV", str(tsv_path)))
    
    # 5. Parquet (if pyarrow is available)
    try:
        # Clean data for Parquet (it doesn't handle mixed types well)
        df_parquet = df.copy()
        for col in df_parquet.columns:
            if df_parquet[col].dtype == 'object':
                # Convert all to string to avoid mixed type issues
                df_parquet[col] = df_parquet[col].astype(str)
        
        parquet_path = data_dir / f"{base_name}.parquet"
        df_parquet.to_parquet(parquet_path, index=False)
        formats_saved.append(("Parquet", str(parquet_path)))
    except Exception as e:
        print(f"Parquet format not available: {str(e)}")
    
    # 6. Pickle
    pickle_path = data_dir / f"{base_name}.pkl"
    df.to_pickle(pickle_path)
    formats_saved.append(("Pickle", str(pickle_path)))
    
    return formats_saved

def create_specialized_test_datasets():
    """Create specialized test datasets for different scenarios"""
    
    datasets = {}
    
    # 1. High missing values dataset
    high_missing_df = create_comprehensive_test_dataset()
    # Introduce more missing values
    for col in ['age', 'income', 'purchase_amount', 'email']:
        missing_idx = np.random.choice(high_missing_df.index, size=int(0.4 * len(high_missing_df)), replace=False)
        high_missing_df.loc[missing_idx, col] = np.nan
    
    datasets['high_missing'] = high_missing_df
    
    # 2. Many duplicates dataset
    many_duplicates_df = create_comprehensive_test_dataset()
    # Add more duplicates
    duplicate_rows = many_duplicates_df.sample(int(0.3 * len(many_duplicates_df)))
    many_duplicates_df = pd.concat([many_duplicates_df, duplicate_rows], ignore_index=True)
    
    datasets['many_duplicates'] = many_duplicates_df
    
    # 3. Outlier heavy dataset
    outlier_heavy_df = create_comprehensive_test_dataset()
    # Add more outliers
    for col in ['age', 'income', 'purchase_amount', 'score']:
        if col in outlier_heavy_df.columns:
            outlier_idx = np.random.choice(outlier_heavy_df.index, size=int(0.1 * len(outlier_heavy_df)), replace=False)
            if col == 'age':
                outlier_heavy_df.loc[outlier_idx, col] = np.random.uniform(100, 200, len(outlier_idx))
            elif col == 'income':
                outlier_heavy_df.loc[outlier_idx, col] = np.random.uniform(300000, 1000000, len(outlier_idx))
            elif col == 'purchase_amount':
                outlier_heavy_df.loc[outlier_idx, col] = np.random.uniform(10000, 50000, len(outlier_idx))
            elif col == 'score':
                outlier_heavy_df.loc[outlier_idx, col] = np.random.uniform(200, 500, len(outlier_idx))
    
    datasets['outlier_heavy'] = outlier_heavy_df
    
    # 4. Clean dataset (for comparison)
    clean_df = pd.DataFrame({
        'id': range(1, 501),
        'name': [f"Item_{i}" for i in range(1, 501)],
        'value': np.random.normal(100, 20, 500),
        'category': np.random.choice(['A', 'B', 'C'], 500),
        'date': [
            (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
            for i in range(500)
        ]
    })
    
    datasets['clean'] = clean_df
    
    return datasets

def main():
    """Main function to create all test datasets"""
    
    print("🔄 Creating comprehensive test datasets...")
    
    # Create main comprehensive dataset
    df = create_comprehensive_test_dataset()
    
    print(f"✅ Created main dataset with {len(df)} rows and {len(df.columns)} columns")
    print(f"   - Missing values: {df.isnull().sum().sum()}")
    print(f"   - Duplicate rows: {df.duplicated().sum()}")
    
    # Save in multiple formats
    formats_saved = save_in_multiple_formats(df, "comprehensive_test_data")
    
    print("\n📁 Saved in formats:")
    for format_name, file_path in formats_saved:
        file_size = os.path.getsize(file_path) / 1024  # KB
        print(f"   - {format_name}: {file_path} ({file_size:.1f} KB)")
    
    # Create specialized datasets
    print("\n🔄 Creating specialized test datasets...")
    specialized_datasets = create_specialized_test_datasets()
    
    for dataset_name, dataset_df in specialized_datasets.items():
        print(f"\n📊 {dataset_name.replace('_', ' ').title()} Dataset:")
        print(f"   - Shape: {dataset_df.shape}")
        print(f"   - Missing values: {dataset_df.isnull().sum().sum()}")
        print(f"   - Duplicates: {dataset_df.duplicated().sum()}")
        
        # Save each specialized dataset
        formats_saved = save_in_multiple_formats(dataset_df, f"{dataset_name}_test_data")
        
        print(f"   - Saved in {len(formats_saved)} formats")
    
    print("\n✅ All test datasets created successfully!")
    print("📂 Check the 'data/' directory for all generated files")
    
    # Create a summary report
    summary_report = {
        'created_at': datetime.now().isoformat(),
        'datasets': {
            'comprehensive': {
                'description': 'Main dataset with various data quality issues',
                'rows': len(df),
                'columns': len(df.columns),
                'missing_values': int(df.isnull().sum().sum()),
                'duplicates': int(df.duplicated().sum())
            }
        }
    }
    
    for name, dataset_df in specialized_datasets.items():
        summary_report['datasets'][name] = {
            'description': f'Specialized dataset: {name.replace("_", " ")}',
            'rows': len(dataset_df),
            'columns': len(dataset_df.columns),
            'missing_values': int(dataset_df.isnull().sum().sum()),
            'duplicates': int(dataset_df.duplicated().sum())
        }
    
    # Save summary report
    with open('data/test_datasets_summary.json', 'w') as f:
        json.dump(summary_report, f, indent=2)
    
    print("📋 Summary report saved to: data/test_datasets_summary.json")

if __name__ == "__main__":
    main()
