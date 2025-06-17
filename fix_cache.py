#!/usr/bin/env python3
"""
Fix environment variable caching issue by clearing Python module cache
"""
import sys
import importlib

def clear_cache_and_reload():
    """Clear module cache and reload settings"""
    print("🔄 Clearing Python module cache...")
    
    # Clear specific modules from cache
    modules_to_clear = [
        'config',
        'config.settings', 
        'agents.base_agent_simple',
        'agents.data_profiler_simple',
        'agents.data_cleaning_simple'
    ]
    
    for module_name in modules_to_clear:
        if module_name in sys.modules:
            print(f"   Clearing {module_name}")
            del sys.modules[module_name]
    
    print("✅ Cache cleared!")
    
    # Force reload settings
    print("🔄 Reloading settings...")
    from config.settings import settings
    
    print(f"🔧 Settings loaded with API key: {settings.openai_api_key[:20] if settings.openai_api_key else 'None'}...")
    
    return settings

if __name__ == "__main__":
    settings = clear_cache_and_reload()
    print("✅ Cache fix completed!") 