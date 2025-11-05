"""
Registry and factory for special agents.

This module provides a registry pattern for managing special agent types
and a factory for creating special agent instances with proper context.
"""

import logging
from typing import Dict, Type, Any, Optional
from .base import SpecialAgentBase, SpecialAgentContext

logger = logging.getLogger(__name__)


class SpecialAgentRegistry:
    """
    Registry for managing special agent types.
    Allows dynamic registration of new special agent types without modifying core code.
    """

    _registry: Dict[str, Type[SpecialAgentBase]] = {}

    @classmethod
    def register(cls, agent_type: str, agent_class: Type[SpecialAgentBase]) -> None:
        """
        Register a special agent type.

        Args:
            agent_type: Type identifier (e.g., "web_surfer", "file_surfer")
            agent_class: The special agent class
        """
        agent_type_key = agent_type.lower()
        cls._registry[agent_type_key] = agent_class
        logger.debug(f"Registered special agent type '{agent_type_key}' with class {agent_class.__name__}")

    @classmethod
    def get_agent_class(cls, agent_type: str) -> Type[SpecialAgentBase]:
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
                f"Unknown special agent type: {agent_type}. "
                f"Available types: {available}"
            )
        return cls._registry[agent_type_key]

    @classmethod
    def list_agent_types(cls) -> list[str]:
        """List all registered special agent types."""
        return list(cls._registry.keys())

    @classmethod
    def is_registered(cls, agent_type: str) -> bool:
        """Check if a special agent type is registered."""
        return agent_type.lower() in cls._registry


def register_special_agent(agent_type: str):
    """
    Decorator for automatically registering special agent classes.

    Usage:
        @register_special_agent("web_surfer")
        class WebSurferAgent(SpecialAgentBase):
            pass
    """
    def decorator(cls):
        SpecialAgentRegistry.register(agent_type, cls)
        return cls
    return decorator


def get_special_agent(
    agent_type: str,
    name: str,
    model: str,
    model_settings: Optional[Any] = None,
    context_config: Optional[Dict[str, Any]] = None
) -> tuple[Any, SpecialAgentContext]:
    """
    Factory function to create a special agent instance with context.

    Args:
        agent_type: Type of special agent (e.g., "web_surfer", "file_surfer")
        name: Name for the agent
        model: Model to use
        model_settings: Optional model settings
        context_config: Optional configuration for the agent's context

    Returns:
        tuple: (agent_instance, context) - The OpenAI Agent and its context

    Raises:
        ValueError: If agent_type is not registered
    """
    context_config = context_config or {}

    # Get the special agent class
    agent_class = SpecialAgentRegistry.get_agent_class(agent_type)

    # Create special agent instance
    special_agent = agent_class(name=name, model=model, model_settings=model_settings)

    # Create context
    context = special_agent.create_context(**context_config)
    special_agent.set_context(context)

    # Create the actual agent instance
    agent_instance = special_agent.create_agent(context)

    return agent_instance, context
