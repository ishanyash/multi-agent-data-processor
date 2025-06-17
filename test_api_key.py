#!/usr/bin/env python3
"""
Test script to verify API key loading and OpenAI connection
"""
import sys
import os

def test_env_loading():
    """Test environment variable loading"""
    print("🔍 Testing environment variable loading...")
    
    # Test direct .env file reading
    from pathlib import Path
    env_file = Path('.env')
    
    if env_file.exists():
        print("✅ .env file exists")
        with open(env_file, 'r') as f:
            content = f.read()
            if 'OPENAI_API_KEY=sk-' in content:
                print("✅ .env file contains API key")
            else:
                print("❌ .env file missing API key")
    else:
        print("❌ .env file not found")
    
    # Test dotenv loading
    from dotenv import load_dotenv
    load_dotenv(override=True)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key and api_key.startswith('sk-'):
        print(f"✅ Environment variable loaded: {api_key[:20]}...")
    else:
        print(f"❌ Environment variable issue: {api_key}")
    
    return api_key

def test_settings_loading():
    """Test settings module loading"""
    print("\n🔍 Testing settings module loading...")
    
    # Clear cache first
    modules_to_clear = ['config', 'config.settings']
    for module_name in modules_to_clear:
        if module_name in sys.modules:
            print(f"   Clearing {module_name} from cache")
            del sys.modules[module_name]
    
    # Import settings
    from config.settings import settings
    
    if settings.openai_api_key and settings.openai_api_key.startswith('sk-'):
        print(f"✅ Settings API key loaded: {settings.openai_api_key[:20]}...")
        return True
    else:
        print(f"❌ Settings API key issue: {settings.openai_api_key}")
        return False

def test_openai_connection():
    """Test OpenAI connection"""
    print("\n🔍 Testing OpenAI connection...")
    
    try:
        from openai import OpenAI
        from config.settings import settings
        
        client = OpenAI(api_key=settings.openai_api_key)
        
        # Make a simple test call
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Say 'test successful'"}],
            max_tokens=10
        )
        
        print("✅ OpenAI connection successful!")
        print(f"   Response: {response.choices[0].message.content}")
        return True
        
    except Exception as e:
        print(f"❌ OpenAI connection failed: {str(e)}")
        return False

def main():
    """Main test function"""
    print("=" * 60)
    print("API KEY AND CONNECTION TEST")
    print("=" * 60)
    
    # Test 1: Environment loading
    env_key = test_env_loading()
    
    # Test 2: Settings loading
    settings_ok = test_settings_loading()
    
    # Test 3: OpenAI connection (only if settings are OK)
    if settings_ok:
        connection_ok = test_openai_connection()
    else:
        connection_ok = False
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Environment loading: {'✅ PASS' if env_key and env_key.startswith('sk-') else '❌ FAIL'}")
    print(f"Settings loading: {'✅ PASS' if settings_ok else '❌ FAIL'}")
    print(f"OpenAI connection: {'✅ PASS' if connection_ok else '❌ FAIL'}")
    
    if all([env_key and env_key.startswith('sk-'), settings_ok, connection_ok]):
        print("\n🎉 All tests passed! The cache issue is resolved.")
        return True
    else:
        print("\n⚠️  Some tests failed. Cache issue may persist.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 