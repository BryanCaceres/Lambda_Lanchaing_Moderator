from promts import GeneralModerator
from .agent_interface import AIAgentInterface
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser

class GeneralModeration(AIAgentInterface):
    def __init__(self):
        super().__init__()
        self.general_moderator = GeneralModerator()

    def moderate(self, comment: str) -> str:
        json_parser = JsonOutputParser()
        prompt_template = self.general_moderator.get_promt()
        chain = prompt_template | self.llm | json_parser

        result = chain.invoke({"comment_body": comment})
        return result
