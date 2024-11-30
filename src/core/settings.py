import os

class Settings:
    def __init__(self):
        self.default_language = "English"
        self.default_max_tokens = os.getenv("DEFAULT_MAX_TOKENS")
        self.default_open_ai_model = os.getenv("DEFAULT_OPEN_AI_MODEL")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.default_temperature = os.getenv("DEFAULT_TEMPERATURE")
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")

settings = Settings()
        

