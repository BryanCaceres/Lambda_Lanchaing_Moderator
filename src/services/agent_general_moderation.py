from promts import GeneralModeratorPromt
from .agent_interface import AIAgentInterface
from langchain.agents import create_tool_calling_agent, AgentExecutor
from .tool_tavily_search import tavily_search
import logging

class GeneralModeration(AIAgentInterface):
    def __init__(self):
        super().__init__()
        self.general_moderator_promt = GeneralModeratorPromt().get_promt()
        self.tools = [tavily_search.tool]
        self._set_agent_config(run_name="general_moderation_agent")

    def moderate(self, comment: str) -> dict:

        formatted_tools = self._get_agent_tools()
        general_moderator_agent = create_tool_calling_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.general_moderator_promt.partial(tools=formatted_tools)
        )
        
        moderation_agent_executor = AgentExecutor(
            agent=general_moderator_agent, 
            tools=self.tools, 
            verbose=True,
            return_intermediate_steps=True,
            max_iterations=5,
            early_stopping_method="force"
        )
        
        result = moderation_agent_executor.invoke({"comment_body": comment}, config=self.agent_config)

        logging.info(f'########### result: {result}')
        
        parsed_result = self.json_parser.parse(result.get("output", "{}"))

        enriched_response = {
            "moderation_result": parsed_result,
            "tool_usage": self._extract_tool_usage(result)
        }
        
        return enriched_response
