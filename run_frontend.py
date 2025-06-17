#!/usr/bin/env python3
"""
Launch script for the Multi-Agent Data Processor Frontend
"""
import subprocess
import sys
import os
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = ['streamlit', 'pandas', 'matplotlib', 'plotly']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n💡 Install missing packages with:")
        print("   pip install -r requirements_frontend.txt")
        return False
    
    return True

def setup_environment():
    """Setup the environment for the frontend"""
    print("🔧 Setting up environment...")
    
    # Create necessary directories
    directories = ['uploads', 'output', 'data']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"   ✅ Created directory: {directory}")
    
    # Check for .env file
    if not Path('.env').exists():
        print("⚠️  .env file not found. Creating basic configuration...")
        try:
            # Run the create_env_file script
            subprocess.run([sys.executable, 'create_env_file.py'], check=True)
            print("   ✅ .env file created")
        except subprocess.CalledProcessError:
            print("   ❌ Failed to create .env file")
    else:
        print("   ✅ .env file exists")
    
    return True

def launch_frontend():
    """Launch the Streamlit frontend"""
    print("🚀 Launching Multi-Agent Data Processor Frontend...")
    print("   📊 The web interface will open in your browser")
    print("   🛑 Press Ctrl+C to stop the server")
    print("   🌐 Default URL: http://localhost:8501")
    print("-" * 50)
    
    try:
        # Launch Streamlit
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 'app.py',
            '--server.headless', 'false',
            '--server.port', '8501',
            '--server.address', 'localhost'
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to launch frontend: {e}")
        return False
    except KeyboardInterrupt:
        print("\n🛑 Frontend stopped by user")
        return True
    
    return True

def main():
    """Main function"""
    print("=" * 60)
    print("🤖 MULTI-AGENT DATA PROCESSOR FRONTEND")
    print("=" * 60)
    
    # Step 1: Check dependencies
    if not check_dependencies():
        print("\n❌ Dependency check failed. Please install required packages.")
        return False
    
    print("✅ All dependencies are installed")
    
    # Step 2: Setup environment
    if not setup_environment():
        print("\n❌ Environment setup failed")
        return False
    
    print("✅ Environment setup complete")
    
    # Step 3: Launch frontend
    print("\n" + "=" * 60)
    success = launch_frontend()
    
    if success:
        print("\n✅ Frontend session completed")
    else:
        print("\n❌ Frontend launch failed")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 