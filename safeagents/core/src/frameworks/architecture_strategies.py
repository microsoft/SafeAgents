"""
Architecture strategy classes for different orchestration patterns.
This module provides abstract strategies for centralized and decentralized architectures.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
from .framework_types import Framework
from .architectures import Architecture
from .constants import DELEGATOR_AGENT_NAME, PRESENTER_AGENT_NAME, DEFAULT_DECENTRALIZED_MAX_TURNS


class ArchitectureStrategy(ABC):
    """
    Abstract base class for architecture strategies.
    Encapsulates architecture-specific team building logic.
    """

    def __init__(self, framework: Framework):
        self.framework = framework

    @abstractmethod
    def build_team(self, agents: List[Any], client: Any, **kwargs) -> Any:
        """
        Build a team with the specific architecture pattern.

        Args:
            agents: List of instantiated framework-specific agents
            client: Framework-specific client
            **kwargs: Additional architecture-specific parameters

        Returns:
            Framework-specific team instance
        """
        pass


class CentralizedStrategy(ArchitectureStrategy):
    """
    Strategy for centralized (supervisor-based) architectures.
    A central coordinator/supervisor manages task delegation.
    """

    def build_team(self, agents: List[Any], client: Any, **kwargs) -> Any:
        """
        Build a centralized team with a supervisor.

        Args:
            agents: List of instantiated agents
            client: Framework-specific client
            **kwargs: Additional parameters (max_turns, prompt, etc.)

        Returns:
            Centralized team instance
        """
        if self.framework == Framework.AUTOGEN:
            return self._build_autogen_centralized(agents, client, **kwargs)
        elif self.framework == Framework.LANGGRAPH:
            return self._build_langgraph_centralized(agents, client, **kwargs)
        elif self.framework == Framework.OPENAI_AGENTS:
            return self._build_openai_agents_centralized(agents, client, **kwargs)
        else:
            raise NotImplementedError(f"Centralized architecture not implemented for {self.framework}")

    def _build_autogen_centralized(self, agents: List[Any], client: Any, **kwargs) -> Any:
        """Build Autogen MagenticOneGroupChat."""
        from autogen_agentchat.teams import MagenticOneGroupChat
        max_turns = kwargs.get('max_turns')
        return MagenticOneGroupChat(agents, model_client=client, max_turns=max_turns)

    def _build_langgraph_centralized(self, agents: List[Any], client: Any, **kwargs) -> Any:
        """Build LangGraph supervisor team."""
        from langgraph_supervisor import create_supervisor

        prompt = kwargs.get('prompt')
        if not prompt:
            from .prompts import LANGGRAPH_SUPERVISOR_PROMPT
            prompt = LANGGRAPH_SUPERVISOR_PROMPT

        return create_supervisor(
            agents=agents,
            model=client,
            prompt=prompt,
        ).compile()

    def _build_openai_agents_centralized(self, agents: List[Any], client: Any, **kwargs) -> Any:
        """
        Build OpenAI Agents planner-orchestrator team.
        Returns a dictionary representing the team structure.
        """
        planner = kwargs.get('planner')
        orchestrator = kwargs.get('orchestrator')

        if not planner or not orchestrator:
            raise ValueError("OpenAI Agents centralized architecture requires 'planner' and 'orchestrator'")

        return {
            "planner": planner,
            "orchestrator": orchestrator,
            "agents": agents
        }


class DecentralizedStrategy(ArchitectureStrategy):
    """
    Strategy for decentralized (swarm-based) architectures.
    Agents autonomously handoff tasks to each other.
    """

    def build_team(self, agents: List[Any], client: Any, **kwargs) -> Any:
        """
        Build a decentralized team with handoffs.

        Args:
            agents: List of instantiated agents
            client: Framework-specific client
            **kwargs: Additional parameters (delegator, presenter, max_turns, etc.)

        Returns:
            Decentralized team instance
        """
        if self.framework == Framework.AUTOGEN:
            return self._build_autogen_decentralized(agents, client, **kwargs)
        elif self.framework == Framework.LANGGRAPH:
            return self._build_langgraph_decentralized(agents, client, **kwargs)
        elif self.framework == Framework.OPENAI_AGENTS:
            return self._build_openai_agents_decentralized(agents, client, **kwargs)
        else:
            raise NotImplementedError(f"Decentralized architecture not implemented for {self.framework}")

    def _build_autogen_decentralized(self, agents: List[Any], client: Any, **kwargs) -> Any:
        """Build Autogen Swarm."""
        from autogen_agentchat.teams import Swarm
        from autogen_agentchat.conditions import HandoffTermination

        delegator = kwargs.get('delegator')
        presenter = kwargs.get('presenter')
        max_turns = kwargs.get('max_turns', DEFAULT_DECENTRALIZED_MAX_TURNS)

        if not delegator or not presenter:
            raise ValueError("Autogen decentralized architecture requires 'delegator' and 'presenter'")

        termination = HandoffTermination(target=PRESENTER_AGENT_NAME)
        return Swarm([delegator] + agents + [presenter], termination_condition=termination, max_turns=max_turns)

    def _build_langgraph_decentralized(self, agents: List[Any], client: Any, **kwargs) -> Any:
        """Build LangGraph Swarm."""
        from langgraph_swarm import create_swarm

        delegator = kwargs.get('delegator')
        presenter = kwargs.get('presenter')
        default_agent = kwargs.get('default_active_agent', DELEGATOR_AGENT_NAME)

        if not delegator or not presenter:
            raise ValueError("LangGraph decentralized architecture requires 'delegator' and 'presenter'")

        return create_swarm(
            agents=[delegator] + agents + [presenter],
            default_active_agent=default_agent,
        ).compile()

    def _build_openai_agents_decentralized(self, agents: List[Any], client: Any, **kwargs) -> Any:
        """OpenAI Agents decentralized not yet implemented."""
        raise NotImplementedError("Decentralized architecture is not yet supported in OpenAI Agents framework")


class ArchitectureStrategyFactory:
    """
    Factory for creating architecture strategies.
    """

    @staticmethod
    def create_strategy(architecture: Architecture, framework: Framework) -> ArchitectureStrategy:
        """
        Create an architecture strategy instance.

        Args:
            architecture: The architecture type (centralized/decentralized)
            framework: The framework type

        Returns:
            Architecture strategy instance

        Raises:
            ValueError: If architecture type is unknown
        """
        if architecture == Architecture.CENTRALIZED:
            return CentralizedStrategy(framework)
        elif architecture == Architecture.DECENTRALIZED:
            return DecentralizedStrategy(framework)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
