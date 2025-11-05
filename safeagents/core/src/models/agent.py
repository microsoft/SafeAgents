from dataclasses import dataclass, field
from typing import List, Optional, Callable

from .tool import Tool

@dataclass
class AgentConfig:
    """
    The config storing info about the agent.
    Available special agents: "web_surfer", "file_surfer", "coder", "code_executor".
    name: name of the agent. Do NOT keep: `presenter_safeagents`, `delegator_safeagents`. Make sure it is a valid python identifier.
    handoffs: List of names of other agents to which this agent can handoff tasks.
    handoff_description: Description of the current agent; this will tell other agents when to handoff to this agent.
    """
    name: str
    tools: List[Tool] = field(default_factory=list)
    system_message: Optional[str] = None
    handoffs: List[str] = field(default_factory=list)
    handoff_description: Optional[str] = None
    description: Optional[str] = None
    special_agent: Optional[str] = None
    special_agent_kwargs: dict = field(default_factory=dict)

class Agent:
    """
    Defines the behavior and capabilities of a single agent.
    """

    def __init__(self, config: AgentConfig):
        self.name = config.name
        # self.model_client = config.model_client
        self.tools = config.tools or []
        self.system_message = config.system_message
        self.handoffs = config.handoffs or []
        self.handoff_description = config.handoff_description or None
        self.description = config.description
        self.special_agent = config.special_agent or None
        self.special_agent_kwargs = config.special_agent_kwargs or {}

        if self.special_agent and self.tools:
            raise ValueError("An agent cannot be both a special agent and have tools.")

    # NOTE: No need to have a seperate method for an agent to run, because you only run a team
    # def act(self, task):
    #     """
    #     Perform an action for the given task.
    #     This should be overridden by framework-specific agent implementations.
    #     """
    #     raise NotImplementedError("Subclasses should implement this method.")

    def add_tool(self, tool):
        """
        Add a tool to the agent's capabilities.
        """
        self.tools.append(tool)

    def add_handoff(self, handoff):
        """
        Add a handoff to the agent's capabilities.
        """
        self.handoffs.append(handoff)
