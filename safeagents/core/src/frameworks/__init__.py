"""
Framework implementations for SafeAgents multi-agent systems.

Example of how to add a new framework:

# Step 1: Add to Framework enum (in framework_types.py)
class Framework(Enum):
    # ... existing frameworks
    CREWAI = "crewai"

# Step 2: Create the Team subclass with decorator
from .base import Team, register_framework
from .architectures import Architecture
from .framework_types import Framework
from ..models.design_choices import DesignChoices
from ..models.agent import Agent
from ..models.tool import Tool

@register_framework(Framework.CREWAI)
class TeamCrewAI(Team):

    def __init__(self, agents: list[Agent], architecture: Architecture | str, design_choices: DesignChoices,
                 termination_condition=None, max_turns=None):
        super().__init__(agents, Framework.CREWAI, architecture, design_choices, termination_condition, max_turns)

    def get_client(self):
        # CrewAI-specific client setup
        pass

    def create_delegator_presenter_agents(self) -> None:
        # Setup delegator and presenter agents
        pass

    def instantiate_agents(self) -> None:
        # Convert generic Agents to CrewAI agents
        pass

    def process_tool(self, tool: Tool):
        # Convert generic Tool to CrewAI tool
        return tool.func

    def instantiate_team(self) -> None:
        # Build team depending on architecture
        pass

    async def run(self, task):
        # CrewAI-specific execution
        pass

    async def run_stream(self, task):
        # Optional streaming execution
        pass

# Step 3: Import in __init__.py
from .crewai.crewai import TeamCrewAI
"""

from .base import Team, TeamRegistry, register_framework
from .framework_types import Framework
from .architectures import Architecture
from .constants import (
    DELEGATOR_AGENT_NAME,
    PRESENTER_AGENT_NAME,
    DEFAULT_MAX_TURNS,
    DEFAULT_DECENTRALIZED_MAX_TURNS,
    DEFAULT_RECURSION_LIMIT,
)

# Import framework implementations to register them
from .autogen.autogen import TeamAutogen
from .langgraph.langgraph import TeamLanggraph
from .openai_agents.openai_agents import TeamOpenAIAgents

__all__ = [
    "Team",
    "TeamRegistry",
    "register_framework",
    "Framework",
    "Architecture",
    "TeamAutogen",
    "TeamLanggraph",
    "TeamOpenAIAgents",
    "DELEGATOR_AGENT_NAME",
    "PRESENTER_AGENT_NAME",
    "DEFAULT_MAX_TURNS",
    "DEFAULT_DECENTRALIZED_MAX_TURNS",
    "DEFAULT_RECURSION_LIMIT",
]
