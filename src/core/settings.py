import os
from functools import lru_cache

class Settings:
    default_language : str = "English"
    default_max_tokens : int = 1000
    default_open_ai_model : str = "gpt-4o-mini"
    openai_api_key : str = os.getenv("OPENAI_API_KEY")
    default_temperature : float = 0.5
    tavily_api_key : str = os.getenv("TAVILY_API_KEY")

@lru_cache
def get_config():
    return Settings()

settings = get_config()
        

