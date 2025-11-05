"""
Base classes for LangGraph special agents with state management.

LangGraph uses a state-based approach, so special agents need to manage state
that persists across tool calls within a graph execution.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, TypedDict
from dataclasses import dataclass, field


class LangGraphSpecialAgentState(TypedDict, total=False):
    """
    Base TypedDict for LangGraph special agent state.

    LangGraph requires state to be TypedDict for proper graph state management.
    Each special agent can extend this to add specific state fields.
    """
    agent_name: str
    config: Dict[str, Any]


@dataclass
class LangGraphSpecialAgentBase(ABC):
    """
    Abstract base class for LangGraph special agents.

    All LangGraph special agents should inherit from this class and implement
    the required methods for agent creation and state management.
    """

    name: str
    model: Any  # LangGraph ChatModel

    def __post_init__(self):
        """Initialize after dataclass creation."""
        self._state_instance: Optional[Any] = None

    @abstractmethod
    def create_state_class(self, **kwargs) -> type:
        """
        Create and return the state TypedDict class for this special agent.

        Args:
            **kwargs: State-specific configuration

        Returns:
            A TypedDict class for managing agent state
        """
        pass

    @abstractmethod
    def create_tools(self, state_class: type) -> list:
        """
        Create and return the tools for this special agent.

        Args:
            state_class: The state TypedDict class

        Returns:
            List of LangGraph tools
        """
        pass

    @abstractmethod
    def create_agent(self, tools: list) -> Any:
        """
        Create and return the LangGraph agent instance.

        Args:
            tools: List of tools for the agent

        Returns:
            LangGraph agent instance (e.g., from create_react_agent)
        """
        pass

    @abstractmethod
    def initialize_state(self, **kwargs) -> Dict[str, Any]:
        """
        Initialize the state instance for this agent.

        Args:
            **kwargs: Initialization parameters

        Returns:
            Initial state dictionary
        """
        pass

    async def cleanup_state(self, state: Dict[str, Any]):
        """
        Cleanup agent state resources.

        Args:
            state: The state dictionary to clean up
        """
        pass
