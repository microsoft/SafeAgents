from ..base import Team, register_framework, Architecture
from ..framework_types import Framework
from ..client_factory import ClientFactory
from ..constants import DELEGATOR_AGENT_NAME, PRESENTER_AGENT_NAME
from ..prompts import (
    AUTOGEN_DELEGATOR_PROMPT,
    AUTOGEN_PRESENTER_PROMPT,
    AUTOGEN_PRESENTER_DESCRIPTION,
    PRESENTER_HANDOFF_DESCRIPTION,
    get_default_agent_handoff_description
)
from ...models.agent import Agent, Tool

@register_framework(Framework.AUTOGEN)
class TeamAutogen(Team):
    """Team implementation for the Autogen framework."""

    def __init__(self, agents: list[Agent], architecture: Architecture | str,
                 termination_condition=None, max_turns=None):
        super().__init__(agents, Framework.AUTOGEN, architecture, termination_condition, max_turns)

    def get_client(self):
        """Returns the autogen client using ClientFactory."""
        llm_config = self.get_llm_config()
        return ClientFactory.create_client(Framework.AUTOGEN, llm_config)

    def create_delegator_presenter_agents(self) -> None:
        from autogen_agentchat.agents import AssistantAgent
        from autogen_agentchat.base import Handoff as HandoffBase

        # create handoffbase for each agent
        handoffs = {}
        all_agents = self.agents
        for agent in all_agents:
            handoffs[agent.name] = HandoffBase(
                target=agent.name,
                description=agent.handoff_description if agent.handoff_description else get_default_agent_handoff_description(agent.name),
            )
        # NOTE: no handoff ever required for delegator as it only delegates
        handoffs[PRESENTER_AGENT_NAME] = HandoffBase(
            target=PRESENTER_AGENT_NAME,
            description=PRESENTER_HANDOFF_DESCRIPTION,
        )
        self.handoff_bases = handoffs

        self.delegator = AssistantAgent(
            DELEGATOR_AGENT_NAME,
            model_client=self.client,
            system_message=AUTOGEN_DELEGATOR_PROMPT,
            handoffs=[handoffs[agent.name] for agent in self.agents]
        )
        self.presenter = AssistantAgent(
            name=PRESENTER_AGENT_NAME,
            model_client=self.client,
            system_message=AUTOGEN_PRESENTER_PROMPT,
            description=AUTOGEN_PRESENTER_DESCRIPTION,
        )
    
    def instantiate_agents(self) -> None:
        # Instantiating agents in Autogen framework
        from autogen_agentchat.agents import AssistantAgent
        self_instantiated_agents = []
        all_agent_names = [agent.name for agent in self.agents]
        for agent in self.agents:
            if isinstance(agent, Agent):
                # STEP 1: Wrap tools for tracking (SafeAgents level)
                if agent.tools:
                    tracked_tools = [
                        self._wrap_tool_for_tracking(tool, agent.name)
                        for tool in agent.tools
                    ]
                else:
                    tracked_tools = []

                # STEP 2: Process tools for Autogen format (framework level)
                processed_tools = [
                    self.process_tool(tool)
                    for tool in tracked_tools
                ] if tracked_tools else []

                handoffs = None
                # Only process handoffs in DECENTRALIZED architecture
                if self.architecture == Architecture.DECENTRALIZED:
                    # Validate handoffs
                    for handoff in agent.handoffs:
                        if handoff not in all_agent_names:
                            raise ValueError(f"Handoff '{handoff}' for agent '{agent.name}' is not a valid agent name in the team.")

                    # Create handoff list with presenter
                    handoff_names = agent.handoffs + [PRESENTER_AGENT_NAME]

                    # assign handoffs to all the agents for swarm architecture
                    autogen_handoffs = []
                    for handoff in handoff_names:
                        if handoff not in self.handoff_bases:
                            raise ValueError(f"Handoff '{handoff}' for agent '{agent.name}' is not a valid agent name in the team.")
                        autogen_handoffs.append(self.handoff_bases[handoff])
                    handoffs = autogen_handoffs
                elif agent.handoffs:
                    # Warn if handoffs are specified in centralized mode (they will be ignored)
                    import warnings
                    warnings.warn(f"Agent '{agent.name}' has handoffs defined, but handoffs are only used in DECENTRALIZED architecture. Handoffs will be ignored.")
                
                if agent.special_agent:
                    # This takes care of special agents like WebSurfer, FileSurfer, CoderAgent, etc.
                    from .special_agents.agents import get_special_agent
                    # Pass tracking callback to special agent for tool tracking
                    agent = get_special_agent(agent, self.client, handoffs, tool_call_tracker=self.track_tool_call)
                else:
                    # This creates a generic Agent
                    agent = AssistantAgent(
                        name=agent.name,
                        model_client=self.client,
                        tools=processed_tools,
                        system_message=agent.system_message,
                        handoffs=handoffs,
                        description=agent.description
                    )
            else:
                raise ValueError("Each agent must be an instance of Agent")
            
            self_instantiated_agents.append(agent)

        self.agents = self_instantiated_agents

    def process_tool(self, tool: Tool):
        """
        Process the generic SafeAgent's Tool of the agents to specific framework's tool.
        Must be implemented by each framework-specific subclass.
        """
        return tool.func

    def instantiate_team(self) -> None:
        from ..architecture_strategies import ArchitectureStrategyFactory

        strategy = ArchitectureStrategyFactory.create_strategy(self.architecture, self.framework)

        if self.architecture == Architecture.CENTRALIZED:
            self.team = strategy.build_team(
                agents=self.agents,
                client=self.client,
                max_turns=self.max_turns
            )
        elif self.architecture == Architecture.DECENTRALIZED:
            self.team = strategy.build_team(
                agents=self.agents,
                client=self.client,
                delegator=self.delegator,
                presenter=self.presenter,
                max_turns=self.max_turns
            )

    async def run(self, task, verbose: bool = False, assessment: list[str] = None, attack_detector=None):
        """
        Run the team on the given task.

        Args:
            task: The task to execute
            verbose: If True, print logs to console using Console UI. If False, collect logs silently.
            assessment: Optional list of assessment methods to run (e.g., ["aria", "dharma"])
            attack_detector: Optional function to detect attacks. Takes execution_trace dict,
                           returns True if attack detected. If attack detected and "aria" in
                           assessment, ARIA score is set to "4".

        Returns:
            dict: Execution result with logs, messages, and optional assessment scores
        """
        from io import StringIO
        import sys

        # Capture output
        logs = []
        messages = []

        if verbose:
            # Use Console for interactive output
            from autogen_agentchat.ui import Console
            await Console(self.team.run_stream(task=task))

        # Collect messages from team run
        # Note: team.run() returns a coroutine, which when awaited returns an async iterator
        result_stream = self.team.run_stream(task=task)
        async for message in result_stream:
            # Extract message content
            source = "unknown"
            if hasattr(message, "source"):
                src = getattr(message, "source")
                if isinstance(src, dict):
                    source = src.get("name", "unknown")
                elif isinstance(src, str):
                    source = src
                else:
                    source = str(src)

            msg_dict = {
                "source": source,
                "content": str(getattr(message, "content", ""))
            }
            messages.append(msg_dict)
            log_entry = f"[{msg_dict['source']}]: {msg_dict['content']}"
            logs.append(log_entry)

        result = {
            "stop_reason": "Task completed",
            "messages": messages,
            "logs": "\n".join(logs)
        }

        # Populate execution trace with task and timing
        self.execution_trace['task'] = task
        self.execution_trace['logs'] = result["logs"]
        self.execution_trace['messages'] = result.get("messages", [])

        # Run attack detection if provided
        result = self._run_attack_detection(result, attack_detector, assessment)

        # Run assessment if requested (and not already set by attack detection)
        if assessment:
            from ...evaluation.assessment import Assessment
            # Only run assessment for methods not already set
            existing_assessments = result.get('assessment', {})
            methods_to_run = [m for m in assessment if m not in existing_assessments]
            if methods_to_run:
                new_assessments = Assessment.evaluate_logs(
                    result["logs"],
                    methods=methods_to_run,
                    framework="autogen"
                )
                if 'assessment' not in result:
                    result['assessment'] = {}
                result['assessment'].update(new_assessments)

        # Always include execution trace in results
        result['execution_trace'] = self.execution_trace

        return result

    async def run_stream(self, task, verbose: bool = False, assessment: list[str] = None, attack_detector=None):
        """
        Run the team on the given task with streaming.

        Args:
            task: The task to execute
            verbose: If True, print logs to console using Console UI. If False, collect logs silently.
            assessment: Optional list of assessment methods to run (e.g., ["aria", "dharma"])
            attack_detector: Optional function to detect attacks

        Returns:
            dict: Execution result with logs, messages, and optional assessment scores
        """
        # For streaming, delegate to run() with same verbose setting
        return await self.run(task, verbose=verbose, assessment=assessment, attack_detector=attack_detector)

"""
@dataclass
class AgentConfig:
    # The config storing info about the agent.
    # Available special agents: "web_surfer", "file_surfer", "coder", "code_executor".
    name: str
    tools: List[Tool] = field(default_factory=list)
    system_message: Optional[str] = None
    handoffs: List[str] = field(default_factory=list)
    handoff_description: Optional[str] = None
    description: Optional[str] = None
    special_agent: Optional[str] = None
    special_agent_kwargs: dict = field(default_factory=dict)
"""





























# from safe_agents.team import Team
# from autogen_agentchat.teams import MagenticOneGroupChat
# from autogen_agentchat.ui import Console
# from autogen_ext.agents.web_surfer import MultimodalWebSurfer
# from autogen_ext.agents.file_surfer import FileSurfer
# from autogen_ext.agents.magentic_one import MagenticOneCoderAgent
# from autogen_agentchat.agents import CodeExecutorAgent, AssistantAgent
# from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor

# class MagenticTeam(Team):
#     async def run_stream(self, task):
#         return await Console(
#             self.agents.run_stream(
#                 task=f"Execute the following task using the ToolUser agent if it has a relevant tool. If not, try other agents. Task: {task.prompt}."
#             )
#         )

# def build_team(task, model, tools):
#     tool_agent = AssistantAgent(
#         name="ToolUser",
#         model_client=model.get_client(),
#         tools=[t.func for t in tools],
#         system_message="Use tools to solve tasks.",
#     )

#     surfer = MultimodalWebSurfer(
#         "WebSurfer",
#         model_client=model.get_client(),
#     )
#     file_surfer = FileSurfer("FileSurfer", model_client=model.get_client())
#     coder = MagenticOneCoderAgent("Coder", model_client=model.get_client())
#     terminal = CodeExecutorAgent(
#         "ComputerTerminal", code_executor=LocalCommandLineCodeExecutor()
#     )

#     team = MagenticOneGroupChat(
#         [surfer, coder, file_surfer, terminal, tool_agent], model_client=model.get_client()
#     )
#     return MagenticTeam(team)
