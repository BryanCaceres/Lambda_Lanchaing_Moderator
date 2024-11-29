from abc import ABC, abstractmethod
from core.settings import settings
from langchain_openai import ChatOpenAI

class AIAgentInterface(ABC):
    def __init__(self):
        self.llm = ChatOpenAI(
            model_name=settings.default_open_ai_model,
            temperature=settings.default_temperature,
            openai_api_key=settings.openai_api_key
        )

    @abstractmethod
    def moderate(self, context: str) -> str:
        pass
    