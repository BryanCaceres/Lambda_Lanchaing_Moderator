from promts import HatespeechExpertPromt
from .agent_interface import AIAgentInterface
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import tavily_search
import logging

agent_promt_template = HatespeechExpertPromt()

class HatespeechExpertAgent(AIAgentInterface):
    def __init__(self):
        super().__init__()
        self._set_agent_config(run_name="hatespeech_expert_agent")
        self.hatespeech_expert_promt = agent_promt_template.get_promt()
        self.tools = [tavily_search.tool]

    def moderate(self, comment_body: str, moderation_name: str, moderation_reason: str, infraction_evidence: str) -> dict:
        """
        Execute the hatespeech expert agent
        :param comment_body: str with the comment body
        :param moderation_name: str with the moderation name
        :param moderation_reason: str with the moderation reason
        :param infraction_evidence: str with the infraction evidence
        :return: dict with the moderation result
        """

        formatted_tools = self._get_agent_tools_string()

        hatespeech_expert_agent = create_tool_calling_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.hatespeech_expert_promt
        )

        hatespeech_expert_executor = AgentExecutor(
            agent=hatespeech_expert_agent, 
            tools=self.tools, 
            verbose=True,
            return_intermediate_steps=True,
            max_iterations=5,
            early_stopping_method="force"
        )

        result = hatespeech_expert_executor.invoke(
            {
                "comment_body": comment_body,
                "moderation_name": moderation_name, 
                "moderation_reason": moderation_reason, 
                "infraction_evidence": infraction_evidence, 
                "tools": formatted_tools
            }, 
            config=self.agent_config
        )

        logging.debug(f'Result: {result}')

        parsed_result = self.json_parser.parse(result.get("output", "{}"))

        enriched_response = {
            "moderation_result": parsed_result,
            "tool_usage": self._extract_tool_usage(result)
        }

        return enriched_response
