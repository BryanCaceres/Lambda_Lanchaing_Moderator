from langchain.chains import LLMChain
from promts import GeneralModerator
from .agent_interface import AIAgentInterface

class GeneralModeration(AIAgentInterface):
    def __init__(self):
        super().__init__()
        self.general_moderator = GeneralModerator()

    def moderate(self, context: str) -> str:

        prompt_template = self.general_moderator.get_promt()
        chain = prompt_template | self.llm

        result = chain.invoke({"input_text": context})
        return result.content