#!/usr/bin/env python3
"""
Run test with forced environment variable loading to bypass cache issues
"""
import os
import sys
from pathlib import Path

def setup_environment():
    """Setup environment variables directly"""
    print("🔧 Setting up environment variables...")
    
    # Read .env file directly and set environment variables
    env_file = Path('.env')
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value
                    if key == 'OPENAI_API_KEY':
                        print(f"✅ Set {key}: {value[:20]}...")
    
    # Verify the environment variable is set
    api_key = os.environ.get('OPENAI_API_KEY')
    if api_key and api_key.startswith('sk-'):
        print(f"✅ Environment setup complete: {api_key[:20]}...")
        return True
    else:
        print(f"❌ Environment setup failed: {api_key}")
        return False

def run_test():
    """Run the actual test"""
    print("\n🚀 Running test with fixed environment...")
    
    # Import after setting environment
    from main_simple import SimpleDataProcessingPipeline
    
    # Create test dataset
    from test_simple import create_test_dataset
    dataset_path = create_test_dataset()
    
    # Run pipeline
    pipeline = SimpleDataProcessingPipeline()
    try:
        results = pipeline.process_dataset(dataset_path, job_id="fixed_test")
        
        print("\n🎉 SUCCESS! Test completed without errors!")
        print(f"   Status: {results['status']}")
        print(f"   Output: {results['output_dataset']}")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        return False

def main():
    """Main function"""
    print("=" * 60)
    print("CACHE-FIX TEST RUNNER")
    print("=" * 60)
    
    # Step 1: Setup environment
    if not setup_environment():
        print("❌ Environment setup failed. Cannot proceed.")
        return False
    
    # Step 2: Run test
    success = run_test()
    
    print("\n" + "=" * 60)
    print(f"RESULT: {'✅ SUCCESS' if success else '❌ FAILED'}")
    print("=" * 60)
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 