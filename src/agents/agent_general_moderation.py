from promts import GeneralModeratorPromt
from .agent_interface import AIAgentInterface
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import tavily_search
import logging

agent_promt_template = GeneralModeratorPromt()

class GeneralModerationAgent(AIAgentInterface):
    def __init__(self):
        super().__init__()
        self.tools = [tavily_search.tool]
        self._set_agent_config(run_name="general_moderation_agent")
        self.general_moderator_promt = agent_promt_template.get_promt()

    def moderate(self, comment: str) -> dict:
        """
        Execute the general moderation agent
        :param comment: str with the comment body
        :return: dict with the first moderation result with the following keys:
            - short_name: str with the short name of the moderation
            - hate_speech: bool with the hate speech rule result
            - harassment_bullying: bool with the harassment bullying rule result
            - spam_self_promotion: bool with the spam self promotion rule result
            - personal_information: bool with the personal information rule result
            - other_reason: bool with the other reason result
            - breaked_rules: str with the breaked rules
            - reason: str with the reason of the infraction
            - evidence: str with the evidence of the infraction
        """

        formatted_tools = self._get_agent_tools_string()

        general_moderator_agent = create_tool_calling_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.general_moderator_promt
        )
        
        moderation_agent_executor = AgentExecutor(
            agent=general_moderator_agent, 
            tools=self.tools, 
            verbose=True,
            return_intermediate_steps=True,
            max_iterations=5,
            early_stopping_method="force"
        )

        result = moderation_agent_executor.invoke(
            {
                "comment_body": comment, 
                "tools": formatted_tools
            }, 
            config=self.agent_config
        )

        logging.debug(f'Result: {result}')
        
        parsed_result = self.json_parser.parse(result.get("output", "{}"))
        parsed_result["general_result"] = self._general_moderation_result(parsed_result)
        
        enriched_response = {
            "moderation_result": parsed_result,
            "tool_usage": self._extract_tool_usage(result)
        }

        return enriched_response
    
    def _general_moderation_result(self, result: dict) -> bool:
        """
        Check all the rules of the general moderation and return the result
        :param result: dict with the moderation result
        :return: bool with the result
        """
        rules = [
            result.get("hate_speech", False),
            result.get("harassment_bullying", False),
            result.get("spam_self_promotion", False),
            result.get("personal_information", False),
            result.get("other_reason", False)
        ]
        return any(rules)
        
        