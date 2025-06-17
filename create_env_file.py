#!/usr/bin/env python3
"""
Create .env file with proper API key configuration
"""

import os
import getpass

def create_env_file():
    """Create the .env file with user-provided API key"""
    
    print("🔧 Setting up .env file for OpenAI API configuration")
    print("=" * 50)
    
    # Check if .env already exists
    if os.path.exists('.env'):
        response = input("📁 .env file already exists. Overwrite? (y/N): ").lower()
        if response != 'y':
            print("❌ Cancelled. Existing .env file preserved.")
            return False
    
    # Get API key from user
    print("\n🔑 Please provide your OpenAI API key:")
    print("   You can get one from: https://platform.openai.com/account/api-keys")
    print("   Format: sk-proj-...")
    
    api_key = getpass.getpass("Enter your OpenAI API key: ").strip()
    
    if not api_key:
        print("❌ No API key provided. Setup cancelled.")
        return False
    
    if not api_key.startswith('sk-'):
        print("⚠️  Warning: API key doesn't start with 'sk-'. Please verify it's correct.")
        response = input("Continue anyway? (y/N): ").lower()
        if response != 'y':
            print("❌ Setup cancelled.")
            return False
    
    # Create env content with user's API key
    env_content = f"""OPENAI_API_KEY={api_key}
OPENAI_MODEL_PRIMARY=gpt-4
OPENAI_MODEL_SECONDARY=gpt-3.5-turbo
MAX_RETRIES=3
PROCESSING_TIMEOUT=300
LOG_LEVEL=INFO"""

    try:
        with open('.env', 'w') as f:
            f.write(env_content)
        
        print("\n✅ .env file created successfully!")
        print("✅ The file contains your OpenAI API key and configuration.")
        
        # Verify the file was created (without exposing the key)
        with open('.env', 'r') as f:
            content = f.read()
            if 'OPENAI_API_KEY=' in content and len(api_key) > 10:
                masked_key = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
                print(f"✅ API key verified in .env file: {masked_key}")
                return True
                
    except Exception as e:
        print(f"❌ Failed to create .env file: {e}")
        return False

def create_env_template():
    """Create a template .env file for users to fill in"""
    
    template_content = """# OpenAI API Configuration
# Get your API key from: https://platform.openai.com/account/api-keys
OPENAI_API_KEY=your_openai_api_key_here

# Model Configuration
OPENAI_MODEL_PRIMARY=gpt-4
OPENAI_MODEL_SECONDARY=gpt-3.5-turbo

# Processing Configuration
MAX_RETRIES=3
PROCESSING_TIMEOUT=300
LOG_LEVEL=INFO"""

    try:
        with open('.env.template', 'w') as f:
            f.write(template_content)
        
        print("✅ .env.template file created!")
        print("📝 Copy this to .env and add your API key")
        return True
        
    except Exception as e:
        print(f"❌ Failed to create .env.template: {e}")
        return False

if __name__ == "__main__":
    print("🤖 Multi-Agent Data Processor - Environment Setup")
    print("=" * 50)
    
    choice = input("\nChoose setup method:\n1. Interactive setup (recommended)\n2. Create template file\nEnter choice (1/2): ").strip()
    
    if choice == '2':
        success = create_env_template()
        if success:
            print("\n🎉 Template created! Next steps:")
            print("1. Copy .env.template to .env")
            print("2. Edit .env and add your OpenAI API key")
            print("3. Run: python test_simple.py")
    else:
        success = create_env_file()
        if success:
            print("\n🎉 .env file setup complete!")
            print("Now you can run: python test_simple.py")
        else:
            print("\n❌ .env file setup failed.") 