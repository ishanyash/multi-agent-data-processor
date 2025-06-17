import logging
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from openai import OpenAI  # Use synchronous client for simplicity

class BaseAgent(ABC):
    def __init__(self, name: str, model: str = None):
        # Force reload settings to avoid caching issues
        import importlib
        if 'config.settings' in __import__('sys').modules:
            importlib.reload(__import__('sys').modules['config.settings'])
        
        from config.settings import settings
        
        self.name = name
        self.model = model or settings.openai_model_primary
        self.client = OpenAI(
            api_key=settings.openai_api_key,
            timeout=30.0
        )
        self.logger = logging.getLogger(f"Agent.{name}")
        
        # Debug logging
        self.logger.info(f"Agent {name} initialized with API key: {settings.openai_api_key[:20] if settings.openai_api_key else 'None'}...")
        
    def call_llm(self, messages: List[Dict], functions: Optional[List] = None):
        """Make OpenAI API call with error handling (synchronous)"""
        try:
            if functions:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    functions=functions,
                    function_call="auto"
                )
            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages
                )
            return response
        except Exception as e:
            self.logger.error(f"LLM call failed: {str(e)}")
            raise
    
    @abstractmethod
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Main processing method - must be implemented by each agent"""
        pass
    
    def create_system_prompt(self) -> str:
        """Create agent-specific system prompt"""
        return f"You are a {self.name} specialized in data processing tasks."
