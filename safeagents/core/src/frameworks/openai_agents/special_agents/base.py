"""
Base classes for special agents with context management.

This module provides the foundation for creating special agents that require
persistent context (like browser sessions, file browsers, etc.).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class SpecialAgentContext:
    """
    Base class for special agent contexts.

    Each special agent can extend this to add specific context attributes
    (e.g., browser session, file browser state, execution environment).
    """
    agent_name: str
    config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize context-specific resources."""
        pass

    async def cleanup(self):
        """Cleanup resources when agent is done."""
        pass


class SpecialAgentBase(ABC):
    """
    Abstract base class for special agents.

    All special agents should inherit from this class and implement
    the required methods for agent creation and context management.
    """

    def __init__(self, name: str, model: str, model_settings: Optional[Any] = None):
        """
        Initialize special agent base.

        Args:
            name: Agent name
            model: Model name to use
            model_settings: Optional model settings
        """
        self.name = name
        self.model = model
        self.model_settings = model_settings
        self._context: Optional[SpecialAgentContext] = None

    @abstractmethod
    def create_context(self, **kwargs) -> SpecialAgentContext:
        """
        Create and return the context for this special agent.

        Args:
            **kwargs: Context-specific configuration

        Returns:
            SpecialAgentContext: The created context
        """
        pass

    @abstractmethod
    def create_agent(self, context: SpecialAgentContext) -> Any:
        """
        Create and return the OpenAI Agents framework agent instance.

        Args:
            context: The context for this agent

        Returns:
            Agent: OpenAI Agents framework agent instance
        """
        pass

    def get_context(self) -> Optional[SpecialAgentContext]:
        """Get the current context."""
        return self._context

    def set_context(self, context: SpecialAgentContext):
        """Set the context."""
        self._context = context

    async def cleanup(self):
        """Cleanup agent resources."""
        if self._context:
            await self._context.cleanup()
