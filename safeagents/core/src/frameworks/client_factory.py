"""
Client factory for creating framework-specific clients.
This module standardizes client creation across different frameworks.
"""

from typing import Any, Dict
from .framework_types import Framework


class ClientFactory:
    """
    Factory class for creating framework-specific clients.
    Standardizes client creation and configuration across frameworks.
    """

    @staticmethod
    def create_client(framework: Framework, llm_config: Dict[str, Any]) -> Any:
        """
        Create a framework-specific client.

        Args:
            framework: The framework type
            llm_config: LLM configuration dictionary

        Returns:
            Framework-specific client instance

        Raises:
            ValueError: If framework is not supported
        """
        if framework == Framework.AUTOGEN:
            return ClientFactory._create_autogen_client(llm_config)
        elif framework == Framework.LANGGRAPH:
            return ClientFactory._create_langgraph_client(llm_config)
        elif framework == Framework.OPENAI_AGENTS:
            return ClientFactory._create_openai_agents_client(llm_config)
        else:
            raise ValueError(f"Unsupported framework: {framework}")

    @staticmethod
    def _create_autogen_client(llm_config: Dict[str, Any]) -> Any:
        """
        Create Autogen framework client.

        Args:
            llm_config: LLM configuration dictionary

        Returns:
            AzureOpenAIChatCompletionClient instance
        """
        from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
        return AzureOpenAIChatCompletionClient(**llm_config)

    @staticmethod
    def _create_langgraph_client(llm_config: Dict[str, Any]) -> Any:
        """
        Create LangGraph framework client.

        Args:
            llm_config: LLM configuration dictionary

        Returns:
            AzureChatOpenAI instance
        """
        from langchain_openai.chat_models import AzureChatOpenAI

        # Transform config keys to LangGraph format
        langgraph_config = {
            "name": llm_config["model"],
            "temperature": llm_config["temperature"],
            "azure_deployment": llm_config["azure_deployment"],
            "azure_endpoint": llm_config["azure_endpoint"],
            "api_version": llm_config["api_version"],
            "azure_ad_token_provider": llm_config["azure_ad_token_provider"],
        }

        return AzureChatOpenAI(**langgraph_config)

    @staticmethod
    def _create_openai_agents_client(llm_config: Dict[str, Any]) -> Any:
        """
        Create OpenAI Agents framework client.
        Also sets up global OpenAI Agents configuration.

        Args:
            llm_config: LLM configuration dictionary

        Returns:
            AsyncAzureOpenAI instance
        """
        import os
        from openai import AsyncAzureOpenAI
        from agents import set_default_openai_api, set_default_openai_client, set_tracing_disabled

        # Create client configuration
        openai_agents_config = {
            "azure_endpoint": llm_config["azure_endpoint"],
            "azure_ad_token_provider": llm_config["azure_ad_token_provider"],
            "api_version": llm_config["api_version"],
        }

        client = AsyncAzureOpenAI(**openai_agents_config)

        # Set global defaults for OpenAI Agents framework
        set_default_openai_api("chat_completions")
        set_default_openai_client(client, False)
        set_tracing_disabled(disabled=True)

        # Set environment variables
        os.environ["OPENAI_API_TYPE"] = "azure"
        os.environ["OPENAI_API_BASE"] = llm_config["azure_endpoint"]
        os.environ["OPENAI_API_VERSION"] = llm_config["api_version"]

        return client

    @staticmethod
    def bind_tools_for_framework(client: Any, framework: Framework, tools: list,
                                   parallel_tool_calls: bool = True) -> Any:
        """
        Bind tools to a client in a framework-specific way.

        Args:
            client: The client instance
            framework: The framework type
            tools: List of tools to bind
            parallel_tool_calls: Whether to allow parallel tool calls

        Returns:
            Client with tools bound (framework-dependent)
        """
        if framework == Framework.LANGGRAPH:
            # LangGraph uses bind_tools method
            return client.bind_tools(tools, parallel_tool_calls=parallel_tool_calls)
        elif framework == Framework.AUTOGEN:
            # Autogen handles tools differently (passed to agents, not bound to client)
            return client
        elif framework == Framework.OPENAI_AGENTS:
            # OpenAI Agents handles tools at agent level
            return client
        else:
            return client
