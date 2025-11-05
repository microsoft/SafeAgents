"""
Integration module for LangGraph special agents.

This module provides the interface between the LangGraph framework and special agents.
"""

import logging
from typing import Any, Optional, List
from ....models.agent import Agent
from .registry import get_langgraph_special_agent

logger = logging.getLogger(__name__)


def get_special_agent(
    agent: Agent,
    model: Any,
    handoffs: Optional[List[Any]] = None
) -> Any:
    """
    Create a LangGraph special agent from an Agent configuration.

    This function integrates special agents into the LangGraph framework by:
    1. Creating the special agent with proper state management
    2. Handling agent-specific tools and handoffs
    3. Returning a LangGraph-compatible agent instance

    Args:
        agent: The Agent configuration containing special_agent type and kwargs
        model: LangGraph model to use for the agent
        handoffs: Optional list of handoff tools for agent collaboration

    Returns:
        LangGraph agent instance configured with special agent capabilities

    Raises:
        ValueError: If special_agent type is not recognized or configuration is invalid
    """
    if not agent.special_agent:
        raise ValueError(f"Agent '{agent.name}' does not have a special_agent type specified")

    # Extract special agent configuration
    special_agent_type = agent.special_agent
    special_agent_kwargs = getattr(agent, 'special_agent_kwargs', {})

    logger.info(f"Creating LangGraph special agent '{agent.name}' of type '{special_agent_type}'")

    try:
        # Create the special agent using the registry factory
        # This returns (agent_instance, state_class, initial_state)
        agent_instance, state_class, initial_state = get_langgraph_special_agent(
            agent_type=special_agent_type,
            name=agent.name,
            model=model,
            state_config=special_agent_kwargs
        )

        # The agent instance is stored in the state
        langgraph_agent = initial_state.get("_agent")

        if langgraph_agent is None:
            raise ValueError(
                f"Special agent '{special_agent_type}' did not return a valid agent instance"
            )

        logger.info(f"Successfully created LangGraph special agent '{agent.name}'")
        return langgraph_agent

    except Exception as e:
        logger.error(f"Failed to create special agent '{agent.name}': {e}")
        raise ValueError(
            f"Failed to create special agent '{agent.name}' of type '{special_agent_type}': {e}"
        ) from e
