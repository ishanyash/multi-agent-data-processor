import os
from dotenv import load_dotenv
from pathlib import Path

class Settings:
    def __init__(self):
        # Force reload environment variables with override=True to bypass caching
        env_path = Path(__file__).parent.parent / '.env'
        load_dotenv(dotenv_path=env_path, override=True)
        
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_model_primary = os.getenv("OPENAI_MODEL_PRIMARY", "gpt-4")
        self.openai_model_secondary = os.getenv("OPENAI_MODEL_SECONDARY", "gpt-3.5-turbo")
        self.max_retries = int(os.getenv("MAX_RETRIES", "3"))
        self.processing_timeout = int(os.getenv("PROCESSING_TIMEOUT", "300"))
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        
        # Debug info (can be removed later)
        if self.openai_api_key:
            print(f"✅ API Key loaded: {self.openai_api_key[:20]}...")
        else:
            print("❌ No API Key found!")

# Create settings instance
settings = Settings()
