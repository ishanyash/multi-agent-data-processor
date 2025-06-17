#!/usr/bin/env python3
"""
Quick start script for Multi-Agent Data Processor
Runs a complete test to verify everything is working
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            return True
        else:
            print(f"❌ {description} failed:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ {description} failed with error: {e}")
        return False

def check_api_key():
    """Check if OpenAI API key is configured"""
    env_file = Path(".env")
    if not env_file.exists():
        return False
    
    content = env_file.read_text()
    return "OPENAI_API_KEY=your_openai_api_key_here" not in content

def main():
    print("🚀 Multi-Agent Data Processor - Quick Start")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("main_simple.py").exists():
        print("❌ Please run this script from the multi-agent-data-processor directory")
        sys.exit(1)
    
    # Run setup
    success = run_command("python setup_project.py", "Setting up project structure")
    if not success:
        sys.exit(1)
    
    # Check API key
    if check_api_key():
        print("✅ OpenAI API key is configured")
    else:
        print("⚠️  OpenAI API key not configured - only minimal test will work")
        print("   Edit .env file and add your API key to run full tests")
    
    # Test minimal functionality (no OpenAI required)
    success = run_command("python minimal_test.py", "Running minimal test (no OpenAI)")
    if not success:
        print("❌ Basic setup has issues. Please check your Python environment.")
        sys.exit(1)
    
    # Test with OpenAI if API key is configured
    if check_api_key():
        success = run_command("python test_simple.py", "Running OpenAI integration test")
        if success:
            print("\n🎉 All tests passed! Your multi-agent data processor is ready!")
        else:
            print("\n⚠️  OpenAI test failed. Check your API key and try again.")
    else:
        print("\n✅ Basic functionality verified!")
        print("   Add your OpenAI API key to .env to test full functionality")
    
    print("\n📋 Next steps:")
    print("1. Process your own data: python main_simple.py --dataset your_data.csv")
    print("2. View results in the output/ directory")
    print("3. Check docs/ for detailed documentation")
    print("4. Run scripts/run_tests.sh for comprehensive testing")

if __name__ == "__main__":
    main()
