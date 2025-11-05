"""
Registry and factory for LangGraph special agents.

This module provides a registry pattern for managing LangGraph special agent types
and a factory for creating special agent instances with proper state management.
"""

import logging
from typing import Dict, Type, Any, Optional, Tuple
from .base import LangGraphSpecialAgentBase

logger = logging.getLogger(__name__)


class LangGraphSpecialAgentRegistry:
    """
    Registry for managing LangGraph special agent types.
    Allows dynamic registration of new special agent types without modifying core code.
    """

    _registry: Dict[str, Type[LangGraphSpecialAgentBase]] = {}

    @classmethod
    def register(cls, agent_type: str, agent_class: Type[LangGraphSpecialAgentBase]) -> None:
        """
        Register a LangGraph special agent type.

        Args:
            agent_type: Type identifier (e.g., "web_surfer", "file_surfer")
            agent_class: The special agent class
        """
        agent_type_key = agent_type.lower()
        cls._registry[agent_type_key] = agent_class
        logger.debug(f"Registered LangGraph special agent type '{agent_type_key}' with class {agent_class.__name__}")

    @classmethod
    def get_agent_class(cls, agent_type: str) -> Type[LangGraphSpecialAgentBase]:
        """
        Get the special agent class for a specific type.

        Args:
            agent_type: Type identifier

        Returns:
            The special agent class

        Raises:
            ValueError: If agent type is not registered
        """
        agent_type_key = agent_type.lower()
        if agent_type_key not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(
                f"Unknown LangGraph special agent type: {agent_type}. "
                f"Available types: {available}"
            )
        return cls._registry[agent_type_key]

    @classmethod
    def list_agent_types(cls) -> list[str]:
        """List all registered LangGraph special agent types."""
        return list(cls._registry.keys())

    @classmethod
    def is_registered(cls, agent_type: str) -> bool:
        """Check if a LangGraph special agent type is registered."""
        return agent_type.lower() in cls._registry


def register_langgraph_special_agent(agent_type: str):
    """
    Decorator for automatically registering LangGraph special agent classes.

    Usage:
        @register_langgraph_special_agent("web_surfer")
        class LangGraphWebSurferAgent(LangGraphSpecialAgentBase):
            pass
    """
    def decorator(cls):
        LangGraphSpecialAgentRegistry.register(agent_type, cls)
        return cls
    return decorator


def get_langgraph_special_agent(
    agent_type: str,
    name: str,
    model: Any,
    state_config: Optional[Dict[str, Any]] = None
) -> Tuple[Any, type, Dict[str, Any]]:
    """
    Factory function to create a LangGraph special agent instance with state.

    Args:
        agent_type: Type of special agent (e.g., "web_surfer", "file_surfer")
        name: Name for the agent
        model: LangGraph model to use
        state_config: Optional configuration for the agent's state

    Returns:
        tuple: (agent_instance, state_class, initial_state) - The LangGraph agent, its state class, and initial state

    Raises:
        ValueError: If agent_type is not registered
    """
    state_config = state_config or {}

    # Get the special agent class
    agent_class = LangGraphSpecialAgentRegistry.get_agent_class(agent_type)

    # Create special agent instance
    special_agent = agent_class(name=name, model=model)

    # Create state class
    state_class = special_agent.create_state_class(**state_config)

    # Create tools with state
    tools = special_agent.create_tools(state_class)

    # Create the actual agent instance
    agent_instance = special_agent.create_agent(tools)

    # Initialize state
    initial_state = special_agent.initialize_state(**state_config)

    return agent_instance, state_class, initial_state
