#!/usr/bin/env python3
"""
Setup script for Multi-Agent Data Processor
Creates the complete directory structure and essential files
"""

import os
from pathlib import Path

def create_directory_structure():
    """Create the complete directory structure"""
    
    directories = [
        "agents",
        "config", 
        "utils",
        "tests",
        "data",
        "output",
        "logs",
        "scripts",
        "docs"
    ]
    
    print("Creating directory structure...")
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✓ Created {directory}/")
        
        # Create __init__.py files for Python packages
        if directory in ["agents", "config", "utils", "tests"]:
            init_file = Path(directory) / "__init__.py"
            if not init_file.exists():
                init_file.touch()
                print(f"✓ Created {directory}/__init__.py")

def make_scripts_executable():
    """Make shell scripts executable"""
    scripts = [
        "scripts/run_tests.sh",
        "scripts/setup_environment.sh"
    ]
    
    for script in scripts:
        script_path = Path(script)
        if script_path.exists():
            os.chmod(script_path, 0o755)
            print(f"✓ Made {script} executable")

def main():
    """Main setup function"""
    print("🚀 Finalizing Multi-Agent Data Processor Setup")
    print("=" * 50)
    
    # Create any missing directories
    create_directory_structure()
    print()
    
    # Make scripts executable
    make_scripts_executable()
    print()
    
    print("✅ Repository setup completed!")
    print()
    print("Your repository is ready! Next steps:")
    print("1. Edit .env file and add your OpenAI API key")
    print("2. Install dependencies: pip install -r requirements_minimal.txt")
    print("3. Run initial test: python minimal_test.py")
    print("4. Run OpenAI test: python test_simple.py")
    print()
    print("Repository structure:")
    print("- agents/          (Agent implementations)")
    print("- config/          (Configuration files)")
    print("- utils/           (Utility modules)")
    print("- tests/           (Test files)")
    print("- data/            (Input datasets)")
    print("- output/          (Processing results)")
    print("- logs/            (Log files)")
    print("- scripts/         (Utility scripts)")
    print("- docs/            (Documentation)")

if __name__ == "__main__":
    main()
