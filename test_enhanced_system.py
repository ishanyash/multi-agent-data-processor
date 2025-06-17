#!/usr/bin/env python3
"""
Test Script for Enhanced Multi-Agent Data Processor
Tests multi-format support, interactive decision-making, and download options
"""

import pandas as pd
import sys
from pathlib import Path
import json

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.file_handler import EnhancedFileHandler
from agents.interactive_agent import InteractiveDataAgent

def test_multi_format_loading():
    """Test loading different file formats"""
    
    print("🔍 Testing Multi-Format File Loading...")
    print("=" * 50)
    
    file_handler = EnhancedFileHandler()
    
    # Test different formats
    test_files = [
        "data/comprehensive_test_data.csv",
        "data/comprehensive_test_data.json", 
        "data/comprehensive_test_data.xlsx",
        "data/comprehensive_test_data.tsv",
        "data/comprehensive_test_data.parquet",
        "data/comprehensive_test_data.pkl"
    ]
    
    results = {}
    
    for file_path in test_files:
        if Path(file_path).exists():
            try:
                print(f"\n📁 Loading {file_path}...")
                df, metadata = file_handler.load_file(file_path)
                
                results[file_path] = {
                    'success': True,
                    'shape': df.shape,
                    'format': metadata['original_format'],
                    'size_mb': metadata['file_size_mb'],
                    'conversion_log': metadata['conversion_log']
                }
                
                print(f"   ✅ Success: {df.shape[0]} rows × {df.shape[1]} columns")
                print(f"   📊 Size: {metadata['file_size_mb']:.2f} MB")
                print(f"   🔄 Conversion: {metadata['conversion_log']}")
                
            except Exception as e:
                results[file_path] = {
                    'success': False,
                    'error': str(e)
                }
                print(f"   ❌ Failed: {str(e)}")
        else:
            print(f"   ⚠️  File not found: {file_path}")
    
    return results

def test_interactive_decision_making():
    """Test AI-powered interactive decision making"""
    
    print("\n\n🤖 Testing Interactive Decision Making...")
    print("=" * 50)
    
    # Load test data
    file_handler = EnhancedFileHandler()
    
    try:
        df, metadata = file_handler.load_file("data/comprehensive_test_data.csv")
        print(f"📊 Loaded test dataset: {df.shape[0]} rows × {df.shape[1]} columns")
        
        # Initialize interactive agent
        agent = InteractiveDataAgent()
        
        # Analyze data issues
        print("\n🔍 Analyzing data quality issues...")
        issues = agent.analyze_data_issues(df)
        
        print(f"   📈 AI Quality Score: {issues['ai_insights']['quality_score']}/100")
        print(f"   🎯 Business Impact: {issues['ai_insights']['business_impact'].upper()}")
        print(f"   📋 Strategy: {issues['ai_insights']['recommended_strategy']}")
        
        # Display issues summary
        print(f"\n📊 Issues Found:")
        print(f"   • Missing Values: {len(issues['missing_values'])} columns")
        print(f"   • Duplicates: {issues['duplicates']['count']} rows")
        print(f"   • Outliers: {len(issues['outliers'])} columns")
        print(f"   • Data Type Issues: {len(issues['data_types'])} columns")
        
        # Generate decision interface
        print("\n🤖 Generating decision interface...")
        decision_interface = agent.generate_decision_interface(issues)
        
        print(f"   🎯 Total Decisions: {decision_interface['summary']['total_decisions']}")
        print(f"   👤 User Input Required: {decision_interface['summary']['user_input_required']}")
        print(f"   🤖 Auto Decisions: {decision_interface['summary']['auto_decisions']}")
        
        # Show sample decisions
        print(f"\n📋 Sample Decision Points:")
        for i, decision in enumerate(decision_interface['decision_points'][:3]):
            print(f"   {i+1}. {decision['title']}")
            print(f"      📝 {decision['description']}")
            print(f"      🔧 Options: {len(decision['options'])} available")
        
        # Test automatic decision application (simulate user choices)
        print(f"\n🚀 Testing Decision Application...")
        
        # Create sample user decisions (choose first option for each)
        user_decisions = []
        for decision in decision_interface['decision_points'][:3]:  # Test first 3
            user_decisions.append({
                'id': decision['id'],
                'type': decision['type'],
                'column': decision.get('column'),
                'method': decision['options'][0]['method']  # Choose first option
            })
        
        # Apply decisions
        processed_df, processing_log = agent.apply_decisions(df, user_decisions)
        
        print(f"   ✅ Applied {len(user_decisions)} decisions")
        print(f"   📊 Result: {processed_df.shape[0]} rows × {processed_df.shape[1]} columns")
        print(f"   📋 Processing Log:")
        for log_entry in processing_log[:5]:  # Show first 5 entries
            print(f"      • {log_entry}")
        
        return {
            'original_shape': df.shape,
            'processed_shape': processed_df.shape,
            'issues_found': len(decision_interface['decision_points']),
            'decisions_applied': len(user_decisions),
            'quality_score': issues['ai_insights']['quality_score']
        }
        
    except Exception as e:
        print(f"❌ Error in interactive testing: {str(e)}")
        return {'error': str(e)}

def test_multiple_download_formats():
    """Test saving in multiple formats"""
    
    print("\n\n💾 Testing Multiple Download Formats...")
    print("=" * 50)
    
    # Create sample processed data
    data = {
        'id': range(1, 101),
        'name': [f'Item_{i}' for i in range(1, 101)],
        'value': [i * 1.5 for i in range(1, 101)],
        'category': ['A', 'B', 'C'] * 33 + ['A']
    }
    
    df = pd.DataFrame(data)
    print(f"📊 Created sample dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    
    # Test saving in different formats
    file_handler = EnhancedFileHandler()
    
    formats_to_test = ['csv', 'json', 'xlsx', 'parquet', 'pickle']
    results = {}
    
    for format_type in formats_to_test:
        try:
            output_path = f"output/test_output.{format_type}"
            Path("output").mkdir(exist_ok=True)
            
            success = file_handler.save_file(df, output_path, format_type)
            
            if success and Path(output_path).exists():
                file_size = Path(output_path).stat().st_size / 1024  # KB
                results[format_type] = {
                    'success': True,
                    'path': output_path,
                    'size_kb': file_size
                }
                print(f"   ✅ {format_type.upper()}: {output_path} ({file_size:.1f} KB)")
            else:
                results[format_type] = {'success': False, 'error': 'Save failed'}
                print(f"   ❌ {format_type.upper()}: Save failed")
                
        except Exception as e:
            results[format_type] = {'success': False, 'error': str(e)}
            print(f"   ❌ {format_type.upper()}: {str(e)}")
    
    return results

def test_specialized_datasets():
    """Test with specialized datasets"""
    
    print("\n\n🎯 Testing Specialized Datasets...")
    print("=" * 50)
    
    specialized_files = [
        ("data/high_missing_test_data.csv", "High Missing Values"),
        ("data/many_duplicates_test_data.csv", "Many Duplicates"),
        ("data/outlier_heavy_test_data.csv", "Outlier Heavy"),
        ("data/clean_test_data.csv", "Clean Dataset")
    ]
    
    file_handler = EnhancedFileHandler()
    agent = InteractiveDataAgent()
    
    results = {}
    
    for file_path, description in specialized_files:
        if Path(file_path).exists():
            print(f"\n📊 Testing: {description}")
            
            try:
                # Load data
                df, metadata = file_handler.load_file(file_path)
                
                # Quick analysis
                missing_values = df.isnull().sum().sum()
                duplicates = df.duplicated().sum()
                
                # AI analysis
                issues = agent.analyze_data_issues(df)
                quality_score = issues['ai_insights']['quality_score']
                
                results[description] = {
                    'shape': df.shape,
                    'missing_values': missing_values,
                    'duplicates': duplicates,
                    'quality_score': quality_score,
                    'decision_points': len(agent.generate_decision_interface(issues)['decision_points'])
                }
                
                print(f"   📊 Shape: {df.shape[0]} rows × {df.shape[1]} columns")
                print(f"   🕳️  Missing Values: {missing_values}")
                print(f"   👥 Duplicates: {duplicates}")
                print(f"   📈 Quality Score: {quality_score}/100")
                print(f"   🤖 Decision Points: {results[description]['decision_points']}")
                
            except Exception as e:
                results[description] = {'error': str(e)}
                print(f"   ❌ Error: {str(e)}")
        else:
            print(f"   ⚠️  File not found: {file_path}")
    
    return results

def main():
    """Run all tests"""
    
    print("🚀 Enhanced Multi-Agent Data Processor - Comprehensive Test")
    print("=" * 60)
    
    # Test results storage
    test_results = {}
    
    # Run tests
    test_results['multi_format_loading'] = test_multi_format_loading()
    test_results['interactive_decisions'] = test_interactive_decision_making()
    test_results['download_formats'] = test_multiple_download_formats()
    test_results['specialized_datasets'] = test_specialized_datasets()
    
    # Summary
    print("\n\n📋 TEST SUMMARY")
    print("=" * 60)
    
    # Multi-format loading summary
    loading_results = test_results['multi_format_loading']
    successful_formats = sum(1 for r in loading_results.values() if r.get('success', False))
    print(f"📁 Multi-Format Loading: {successful_formats}/{len(loading_results)} formats successful")
    
    # Interactive decisions summary
    if 'error' not in test_results['interactive_decisions']:
        decisions_result = test_results['interactive_decisions']
        print(f"🤖 Interactive Decisions: {decisions_result['decisions_applied']} decisions applied")
        print(f"   Quality Score: {decisions_result['quality_score']}/100")
        print(f"   Shape Change: {decisions_result['original_shape']} → {decisions_result['processed_shape']}")
    
    # Download formats summary
    download_results = test_results['download_formats']
    successful_downloads = sum(1 for r in download_results.values() if r.get('success', False))
    print(f"💾 Download Formats: {successful_downloads}/{len(download_results)} formats successful")
    
    # Specialized datasets summary
    specialized_results = test_results['specialized_datasets']
    successful_specialized = sum(1 for r in specialized_results.values() if 'error' not in r)
    print(f"🎯 Specialized Datasets: {successful_specialized}/{len(specialized_results)} datasets processed")
    
    print(f"\n✅ Enhanced system testing completed!")
    print(f"🌐 Web interface running at: http://localhost:8501 or http://localhost:8503")
    
    # Save test results
    with open('test_results.json', 'w') as f:
        json.dump(test_results, f, indent=2, default=str)
    
    print(f"📋 Detailed results saved to: test_results.json")

if __name__ == "__main__":
    main() 