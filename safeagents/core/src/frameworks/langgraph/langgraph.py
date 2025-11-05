from ..base import Team, register_framework, Architecture
from ..framework_types import Framework
from ..client_factory import ClientFactory
from ..constants import DELEGATOR_AGENT_NAME, PRESENTER_AGENT_NAME, DEFAULT_RECURSION_LIMIT
from ..prompts import (
    LANGGRAPH_SUPERVISOR_PROMPT,
    LANGGRAPH_DELEGATOR_PROMPT,
    LANGGRAPH_PRESENTER_PROMPT,
    PRESENTER_HANDOFF_DESCRIPTION,
    get_default_agent_handoff_description
)
from ....logger import logger


def extract_agent_data(text_data):
    output = ""
    for item in text_data:
        for agent, agent_data in item.items():
            output += "\n*************************\n"
            output += f"Agent: {agent}\n"
            output += "*************************\n"
            # if agent_data is None:
            #     output += "Agent returned `None`.\n"
            #     continue
            for message in agent_data.get("messages", []):
                content = getattr(message, "content", None)
                tool_calls = getattr(message, "additional_kwargs", {}).get("tool_calls", None)
                
                output += f"  Content: {content}\n"
                if tool_calls:
                    output += f"  Tool Calls:\n"
                    for call in tool_calls:
                        output += f"    - ID: {call.get('id')}\n"
                        output += f"      Name: {call.get('function', {}).get('name')}\n"
                        output += f"      Args: {call.get('function', {}).get('arguments')}\n"
                else:
                    output += "  Tool Calls: None\n"
                output += "-------------------------\n"
    return output


@register_framework(Framework.LANGGRAPH)
class TeamLanggraph(Team):
    """Team implementation for the Langgraph framework."""

    def __init__(self, agents, architecture: Architecture | str,
                 termination_condition=None, max_turns=None):
        super().__init__(agents, Framework.LANGGRAPH, architecture, termination_condition, max_turns)

    def get_client(self):
        """Returns the langgraph client using ClientFactory."""
        llm_config = self.get_llm_config()
        return ClientFactory.create_client(Framework.LANGGRAPH, llm_config)
    
    def create_delegator_presenter_agents(self) -> None:
        from langgraph.prebuilt import create_react_agent
        from langgraph_swarm import create_handoff_tool

        # create handoffbase for each agent
        handoffs = {}
        for agent in self.agents:
            handoffs[agent.name] = create_handoff_tool(
                agent_name=agent.name,
                description=agent.handoff_description if agent.handoff_description else get_default_agent_handoff_description(agent.name),
            )
        handoffs[PRESENTER_AGENT_NAME] = create_handoff_tool(
            agent_name=PRESENTER_AGENT_NAME,
            description=PRESENTER_HANDOFF_DESCRIPTION,
        )
        self.handoff_bases = handoffs

        self.delegator_client = self.get_client()
        delegator_handoffs = [self.handoff_bases[agent.name] for agent in self.agents]
        self.delegator = create_react_agent(
            name=DELEGATOR_AGENT_NAME,
            model=ClientFactory.bind_tools_for_framework(
                self.delegator_client, Framework.LANGGRAPH, delegator_handoffs, parallel_tool_calls=False
            ),
            tools=delegator_handoffs,
            prompt=LANGGRAPH_DELEGATOR_PROMPT,
        )
        self.presenter = create_react_agent(
            name=PRESENTER_AGENT_NAME,
            model=self.client,
            tools=[],
            prompt=LANGGRAPH_PRESENTER_PROMPT,
        )

        return
    
    def process_tool(self, tool):
        """
        Process the generic SafeAgent's Tool of the agents to specific framework's tool.
        """
        return tool.func

    def _wrap_special_agent_tool_for_tracking(self, langgraph_tool, agent_name: str):
        """
        Wrap a special agent's LangChain BaseTool to track its execution.

        Special agents create their tools directly using LangChain's BaseTool class,
        bypassing the normal SafeAgents tool wrapping flow. This method wraps the BaseTool's
        _arun method to enable tracking, similar to how _wrap_tool_for_tracking wraps tool.func.

        Flow comparison:
        - Normal tools:  SafeAgents Tool -> _wrap_tool_for_tracking(tool.func) -> process_tool() -> func
        - Special tools: BaseTool (already created) -> _wrap_special_agent_tool_for_tracking(_arun)

        Args:
            langgraph_tool: The LangChain BaseTool from special agent
            agent_name: Name of the agent this tool belongs to

        Returns:
            The same BaseTool object with wrapped _arun for tracking
        """
        import functools

        # Store original _arun method
        original_arun = langgraph_tool._arun

        # LangChain BaseTool._arun is always async
        @functools.wraps(original_arun)
        async def tracked_arun(*args, **kwargs):
            """Execute tool and track the call."""
            # Combine args and kwargs for tracking
            # LangGraph tools are called with keyword arguments primarily
            # but we should capture both positional and keyword arguments
            combined_args = dict(kwargs)
            if args:
                # If positional args exist, add them with generic keys
                for i, arg in enumerate(args):
                    # Check if arg is JSON serializable
                    try:
                        import json
                        # Try to serialize - if it fails, convert to string
                        json.dumps(arg)
                        combined_args[f'arg_{i}'] = arg
                    except (TypeError, ValueError, OverflowError):
                        # Not JSON serializable (likely context objects)
                        # Try to convert to string, but limit length
                        arg_str = str(arg)
                        if len(arg_str) > 200:
                            arg_str = arg_str[:200] + "..."
                        # If it's a known context type, just use the type name
                        type_name = type(arg).__name__
                        if 'Context' in type_name or 'Wrapper' in type_name:
                            combined_args[f'arg_{i}_type'] = type_name
                        else:
                            combined_args[f'arg_{i}'] = arg_str

            try:
                # Execute the original tool
                result = await original_arun(*args, **kwargs)

                # Track the successful call
                self.track_tool_call(
                    tool_name=langgraph_tool.name,
                    args=combined_args,
                    result=result,
                    agent_name=agent_name
                )

                return result
            except Exception as e:
                # Track failed call
                self.track_tool_call(
                    tool_name=langgraph_tool.name,
                    args=combined_args,
                    result=f"ERROR: {str(e)}",
                    agent_name=agent_name
                )
                raise

        # Replace _arun with tracked version
        langgraph_tool._arun = tracked_arun
        return langgraph_tool

    def instantiate_agents(self) -> None:
        from ...models.agent import Agent
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

                # STEP 2: Process tools for LangGraph format (framework level)
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
                    # Create handoff tools
                    handoffs = [self.handoff_bases[handoff] for handoff in agent.handoffs]
                    handoffs.append(self.handoff_bases[PRESENTER_AGENT_NAME])
                elif agent.handoffs:
                    # Warn if handoffs are specified in centralized mode (they will be ignored)
                    import warnings
                    warnings.warn(f"Agent '{agent.name}' has handoffs defined, but handoffs are only used in DECENTRALIZED architecture. Handoffs will be ignored.")

                # NOTE: the following per-agent-client is required for swarm in langgraph
                #       the bug: https://github.com/langchain-ai/langgraph/discussions/3715
                #       The current fix was suggested here: https://github.com/langchain-ai/langgraph-swarm-py/issues/26
                #       Also, the same is done with `delegator` in `create_delegator_presenter_agents`.
                cur_agent_client = None
                if self.architecture == Architecture.CENTRALIZED:
                    cur_agent_client = self.client
                elif self.architecture == Architecture.DECENTRALIZED:
                    cur_agent_client = self.get_client()
                    # For normal agents, bind tools immediately
                    if not agent.special_agent:
                        cur_agent_client = cur_agent_client.bind_tools(processed_tools + handoffs, parallel_tool_calls=False)
                        self.clients_of_agents[agent.name] = cur_agent_client

                if agent.special_agent:
                    from .special_agents.agents import get_special_agent
                    from .special_agents.registry import get_langgraph_special_agent

                    # Create special agent with unbound client first to get its tools
                    _, _, initial_state = get_langgraph_special_agent(
                        agent_type=agent.special_agent,
                        name=agent.name,
                        model=cur_agent_client,
                        state_config=getattr(agent, 'special_agent_kwargs', {})
                    )

                    # Wrap special agent's tools for tracking
                    # Special agents store tools in initial_state["_tools"]
                    special_agent_tools = []
                    if "_tools" in initial_state and initial_state["_tools"]:
                        special_agent_tools = initial_state["_tools"]
                        for tool in special_agent_tools:
                            self._wrap_special_agent_tool_for_tracking(tool, agent.name)

                    # For decentralized architecture, bind special agent tools + handoffs to the client
                    if self.architecture == Architecture.DECENTRALIZED:
                        tools_to_bind = special_agent_tools + handoffs if handoffs else special_agent_tools
                        if tools_to_bind:
                            cur_agent_client = cur_agent_client.bind_tools(tools_to_bind, parallel_tool_calls=False)
                            self.clients_of_agents[agent.name] = cur_agent_client
                            # Update the model in the special agent's state
                            initial_state["_agent"].model = cur_agent_client

                    # Get the agent instance from state
                    agent = initial_state.get("_agent")
                else:
                    from langgraph.prebuilt import create_react_agent
                    agent = create_react_agent(
                        name=agent.name,
                        model=cur_agent_client,
                        tools=processed_tools + handoffs if handoffs else processed_tools,
                        prompt=agent.system_message,
                    )
            else:
                raise ValueError("Each agent must be an instance of Agent")

            self_instantiated_agents.append(agent)

        self.agents = self_instantiated_agents

    def instantiate_team(self) -> None:
        from ..architecture_strategies import ArchitectureStrategyFactory

        strategy = ArchitectureStrategyFactory.create_strategy(self.architecture, self.framework)

        if self.architecture == Architecture.CENTRALIZED:
            self.team = strategy.build_team(
                agents=self.agents,
                client=self.client,
                prompt=LANGGRAPH_SUPERVISOR_PROMPT
            )
        elif self.architecture == Architecture.DECENTRALIZED:
            self.team = strategy.build_team(
                agents=self.agents,
                client=self.client,
                delegator=self.delegator,
                presenter=self.presenter,
                default_active_agent=DELEGATOR_AGENT_NAME
            )

    async def run(self, task, verbose: bool = False, assessment: list[str] = None, attack_detector=None):
        """
        Run the team on the given task.

        Args:
            task: The task to execute
            verbose: If True, print logs to console. If False, only return logs.
            assessment: Optional list of assessment methods to run (e.g., ["aria", "dharma"])
            attack_detector: Optional function to detect attacks. Takes execution_trace dict,
                           returns True if attack detected. If attack detected and "aria" in
                           assessment, ARIA score is set to "4".

        Returns:
            dict: Execution result with logs, messages, and optional assessment scores
        """
        # Use run_stream since it captures all events
        return await self.run_stream(task, verbose=verbose, assessment=assessment, attack_detector=attack_detector)

    async def run_stream(self, task, verbose: bool = False, assessment: list[str] = None, attack_detector=None):
        """
        Run the team on the given task with streaming.

        Args:
            task: The task to execute
            verbose: If True, print logs to console. If False, only return logs.
            assessment: Optional list of assessment methods to run (e.g., ["aria", "dharma"])
            attack_detector: Optional function to detect attacks

        Returns:
            dict: Execution result with logs, messages, and optional assessment scores
        """
        event_stream = self.team.astream(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": task
                    }
                ]
            },
            config = {"recursion_limit": DEFAULT_RECURSION_LIMIT}
        )

        output = []
        logs = []

        async def _collect_events(event_stream, output, logs, verbose):
            async for event in event_stream:
                output.append(event)
                # Log event
                event_str = str(event)[:200]  # Truncate long events
                logs.append(event_str)
                if verbose:
                    logger.info(f"Event: {event_str}")

        if verbose:
            logger.info("Starting LangGraph execution...")

        await _collect_events(event_stream, output, logs, verbose)

        if verbose:
            logger.info("LangGraph execution completed.")

        # Extract structured data from output
        log_text = extract_agent_data(output)
        logs.append(log_text)

        if verbose:
            logger.info(f"[LangGraph] All logs: {log_text}")

        # Extract messages from output
        messages = []
        for event in output:
            for agent_name, agent_data in event.items():
                if agent_data and isinstance(agent_data, dict):
                    agent_messages = agent_data.get("messages", [])
                    for msg in agent_messages:
                        content = getattr(msg, "content", str(msg))
                        messages.append({
                            "source": agent_name,
                            "content": content
                        })

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
                    framework="langgraph"
                )
                if 'assessment' not in result:
                    result['assessment'] = {}
                result['assessment'].update(new_assessments)

        # Always include execution trace in results
        result['execution_trace'] = self.execution_trace

        return result
































# from safe_agents.team import Team
# from safe_agents.prompt import Prompt
# from safe_agents.task import Task
# from safe_agents.model import Model
# from safe_agents.tool import Tool
# from safe_agents.environment import EnvironmentSetup
# from safe_agents.mitigation import Mitigation
# from safe_agents.assessment import Assessment

# # TODO[NIRMIT]: take care of get_client_centralized
# # from safe_agents import get_client_centralized

# class LangGraphCentralizedTeam(Team):
#     async def run(self, task):
#         agents_team = get_client_centralized({
#             "prompt": task.prompt,
#             "category": task.category,
#             "target_functions": task.target_functions
#         })
#         output = []
#         event_stream = agents_team.astream({
#             "messages": [
#                 {
#                     "role": "user",
#                     "content": f"Execute the following task using the ToolUser agent if it has a relevant tool. If not, try other agents. Task: {task.prompt}."
#                 }
#             ]
#         })

#         async def _print_events(event_stream, output):
#             async for event in event_stream:
#                 output.append(event)

#         await _print_events(event_stream, output)
#         return output

# def build_team(task, model, tools):
#     # LangGraph centralized uses get_client_centralized from safe_agents
#     return LangGraphCentralizedTeam(agents=None, model_client=model.get_client())
