from abc import ABC, abstractmethod
from core.settings import settings
from langchain_core.prompts import PromptTemplate

class PromptsInterface(ABC):
    def __init__(self):
        self.language = settings.default_language
        self.reward = None
        self.security_instructions = None

    @abstractmethod
    def get_promt(self) -> PromptTemplate:
        pass


