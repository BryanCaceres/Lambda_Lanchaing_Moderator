from abc import ABC, abstractmethod
from core.settings import settings
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import Tool


class AIAgentInterface(ABC):
    def __init__(self):
        self.tools = []
        self.json_parser = JsonOutputParser()
        self.llm = ChatOpenAI(
            model_name=settings.default_open_ai_model,
            temperature=settings.default_temperature,
            openai_api_key=settings.openai_api_key
        )
    
    def _get_agent_tools(self) -> str:
        """
        Get the agent tools in a formatted string
        """
        formatted_tools = "\n".join([f"{tool.__class__.__name__}: {tool.description}" for tool in self.tools])
        return formatted_tools

    def _set_agent_config(self, run_name: str):
        """
        Set the agent config for the agent
        """
        self.agent_config = RunnableConfig(
            max_concurrency=5, 
            tags=["moderation", "v1"],
            metadata={
                "project": "community_moderation",
                "version": "1.0.0"
            },
            run_name=run_name
        )

    def _extract_tool_usage(self, result: dict) -> list:
        """
        Extrae información sobre el uso de herramientas de los pasos intermedios
        """
        tool_usage = []
        if "intermediate_steps" in result:
            for step in result["intermediate_steps"]:
                tool_action = step[0]  # ToolAgentAction
                tool_output = step[1]  # Resultado de la herramienta
                
                tool_usage.append({
                    "tool_name": tool_action.tool,  # Cambiado de tool.name a tool
                    "tool_input": tool_action.tool_input,
                    "tool_output": tool_output
                })
        return tool_usage
        
    @abstractmethod
    def moderate(self, context: str) -> str:
        pass
    