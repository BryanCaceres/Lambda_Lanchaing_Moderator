from langchain_core.prompts import PromptTemplate
from promts.promts_interface import PromptsInterface

class GeneralModerator(PromptsInterface):
    def get_promt(self) -> PromptTemplate:
        return PromptTemplate(
            input_variables=["input_text"],
            template="Tell me a funny fact about {input_text}?"
        )
