from ..base import Team, register_framework, Architecture
from ..framework_types import Framework
from ..client_factory import ClientFactory
from ..constants import DEFAULT_MAX_TURNS
from ..prompts import get_openai_planner_instructions, get_openai_orchestrator_instructions
from ...models.agent import Agent
from ....logger import logger
import os
from typing import List, Dict

@register_framework(Framework.OPENAI_AGENTS)
class TeamOpenAIAgents(Team):
    """Team implementation for the OpenAI Agents framework."""

    def __init__(self, agents, architecture: Architecture | str,
                 termination_condition=None, max_turns=None):
        # Initialize instance variables before calling super().__init__
        # because parent's __init__ calls instantiate_agents() which needs these
        self.planner_agent = None
        self.orchestrator_agent = None
        self.agent_context: Dict[str, List[Dict[str, str]]] = {}
        # Store original agent configs before they get replaced by framework-specific agents
        self.original_agents = agents

        super().__init__(agents, Framework.OPENAI_AGENTS, architecture, termination_condition, max_turns)

    def get_client(self):
        """Returns the openai agents client using ClientFactory."""
        llm_config = self.get_llm_config()
        return ClientFactory.create_client(Framework.OPENAI_AGENTS, llm_config)

    def create_delegator_presenter_agents(self) -> None:
        """
        Create planner and orchestrator agents for the OpenAI Agents framework.
        In OpenAI Agents, we use a planner-orchestrator pattern instead of delegator-presenter.
        """
        from agents import Agent as OpenAIAgent, ModelSettings
        from pydantic import BaseModel, Field

        # Build team description for the planner using original agents
        team_description = "\n".join([
            f"{agent.name}: {agent.description if agent.description else 'An AI assistant that can help with tasks.'}"
            for agent in self.original_agents
        ])

        # Create planner agent using centralized prompt function
        planner_instructions = get_openai_planner_instructions(team_description)

        # Get model configuration from environment
        llm_config = self.get_llm_config()

        self.planner_agent = OpenAIAgent(
            name="planner",
            instructions=planner_instructions,
            model=llm_config["azure_deployment"],  # Use deployment name, not model name
            model_settings=ModelSettings(temperature=0.1)
        )

        # Create orchestrator output schema
        class OrchestratorOutput(BaseModel):
            delegate_to: str = Field(
                description="The name of the agent to which the task is being delegated",
                example=self.agents[0].name if self.agents else "agent",
            )
            delegate_task: str = Field(
                description="The task that the agent needs to accomplish",
            )
            done: bool = Field(
                description="Whether the task is done or not",
            )
            stop_reason: str = Field(
                description="The reason for stopping the task in case done is True",
            )

        # Build agent names list for orchestrator instructions using original agents
        agent_names = ", ".join([agent.name for agent in self.original_agents])

        # Use centralized prompt function for orchestrator
        orchestrator_instructions = get_openai_orchestrator_instructions(agent_names)

        self.orchestrator_agent = OpenAIAgent(
            name="orchestrator_agent",
            instructions=orchestrator_instructions,
            output_type=OrchestratorOutput,
            model=llm_config["azure_deployment"],  # Use deployment name, not model name
            model_settings=ModelSettings(temperature=0.1)
        )

    def process_tool(self, tool):
        """
        Process the generic SafeAgent's Tool to OpenAI Agents framework's FunctionTool.
        OpenAI Agents requires tools to be wrapped with @function_tool decorator or FunctionTool class.
        """
        from agents import function_tool

        # Wrap the tool function with the @function_tool decorator
        # This ensures it's properly recognized by the OpenAI Agents framework
        wrapped_tool = function_tool(tool.func)
        return wrapped_tool

    def _wrap_special_agent_tool_for_tracking(self, openai_function_tool, agent_name: str):
        """
        Wrap a special agent's FunctionTool to track its execution.

        Special agents create their tools directly using OpenAI's @function_tool decorator,
        bypassing the normal SafeAgents tool wrapping flow. This method wraps the FunctionTool's
        on_invoke_tool method to enable tracking, similar to how _wrap_tool_for_tracking wraps tool.func.

        Flow comparison:
        - Normal tools:  SafeAgents Tool -> _wrap_tool_for_tracking(tool.func) -> process_tool() -> FunctionTool
        - Special tools: FunctionTool (already created) -> _wrap_special_agent_tool_for_tracking(on_invoke_tool)

        Args:
            openai_function_tool: The OpenAI Agents FunctionTool from special agent
            agent_name: Name of the agent this tool belongs to

        Returns:
            The same FunctionTool object with wrapped on_invoke_tool for tracking
        """
        import functools

        # Store original on_invoke_tool function
        original_invoke = openai_function_tool.on_invoke_tool

        # OpenAI Agents FunctionTool.on_invoke_tool is always async
        @functools.wraps(original_invoke)
        async def tracked_invoke(*args, **kwargs):
            """Execute tool and track the call."""
            # Combine args and kwargs for tracking
            # OpenAI Agents tools are called with keyword arguments primarily
            # but we should capture both positional and keyword arguments
            combined_args = dict(kwargs)
            if args:
                # If positional args exist, add them with generic keys
                # Skip context wrappers (not JSON serializable)
                for i, arg in enumerate(args):
                    # Check if arg is JSON serializable
                    try:
                        import json
                        # Try to serialize - if it fails, convert to string
                        json.dumps(arg)
                        combined_args[f'arg_{i}'] = arg
                    except (TypeError, ValueError, OverflowError):
                        # Not JSON serializable (likely RunContextWrapper, ToolContext, or similar)
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
                result = await original_invoke(*args, **kwargs)

                # Track the successful call
                self.track_tool_call(
                    tool_name=openai_function_tool.name,
                    args=combined_args,
                    result=result,
                    agent_name=agent_name
                )

                return result
            except Exception as e:
                # Track failed call
                self.track_tool_call(
                    tool_name=openai_function_tool.name,
                    args=combined_args,
                    result=f"ERROR: {str(e)}",
                    agent_name=agent_name
                )
                raise

        # Replace on_invoke_tool with tracked version
        openai_function_tool.on_invoke_tool = tracked_invoke
        return openai_function_tool

    def instantiate_agents(self) -> None:
        """
        Instantiate agents for the OpenAI Agents framework.
        """
        from agents import Agent as OpenAIAgent, ModelSettings

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

                # STEP 2: Process tools for OpenAI Agents format (framework level)
                processed_tools = [
                    self.process_tool(tool)
                    for tool in tracked_tools
                ] if tracked_tools else []

                # Validate handoffs (only used in DECENTRALIZED mode)
                if agent.handoffs:
                    if self.architecture == Architecture.CENTRALIZED:
                        # Warn if handoffs are specified in centralized mode (they will be ignored)
                        import warnings
                        warnings.warn(f"Agent '{agent.name}' has handoffs defined, but handoffs are only used in DECENTRALIZED architecture. Handoffs will be ignored.")
                    else:
                        # Validate handoffs for decentralized mode
                        for handoff in agent.handoffs:
                            if handoff not in all_agent_names:
                                raise ValueError(f"Handoff '{handoff}' for agent '{agent.name}' is not a valid agent name in the team.")

                # Get model configuration from environment
                llm_config = self.get_llm_config()

                # Handle special agents
                if agent.special_agent:
                    from .special_agents import get_special_agent

                    # Get special agent kwargs if provided
                    special_agent_kwargs = getattr(agent, 'special_agent_kwargs', {})

                    # Create special agent with context
                    openai_agent, special_context = get_special_agent(
                        agent_type=agent.special_agent,
                        name=agent.name,
                        model=llm_config["azure_deployment"],
                        model_settings=ModelSettings(temperature=llm_config["temperature"]),
                        context_config=special_agent_kwargs
                    )

                    # Store context for cleanup later
                    if not hasattr(self, 'special_agent_contexts'):
                        self.special_agent_contexts = {}
                    self.special_agent_contexts[agent.name] = special_context

                    # Wrap special agent's tools for tracking
                    # Special agents create FunctionTools directly, so we need to wrap them separately
                    if openai_agent.tools:
                        for tool in openai_agent.tools:
                            self._wrap_special_agent_tool_for_tracking(tool, agent.name)
                else:
                    # Create regular OpenAI Agent
                    openai_agent = OpenAIAgent(
                        name=agent.name,
                        instructions=agent.system_message if agent.system_message else f"You are {agent.name}, a helpful AI assistant.",
                        tools=processed_tools if processed_tools else None,
                        model=llm_config["azure_deployment"],  # Use deployment name, not model name
                        model_settings=ModelSettings(temperature=llm_config["temperature"])
                    )

                # Initialize agent context
                self.agent_context[agent.name] = []

            else:
                raise ValueError("Each agent must be an instance of Agent")

            self_instantiated_agents.append(openai_agent)

        self.agents = self_instantiated_agents

    def instantiate_team(self) -> None:
        """
        Instantiate the team for the OpenAI Agents framework.
        OpenAI Agents uses a planner-orchestrator pattern rather than a graph-based approach.
        """
        from ..architecture_strategies import ArchitectureStrategyFactory

        strategy = ArchitectureStrategyFactory.create_strategy(self.architecture, self.framework)

        if self.architecture == Architecture.CENTRALIZED:
            # For centralized architecture, create planner and orchestrator
            self.create_delegator_presenter_agents()

            # Use strategy to build the team structure
            self.team = strategy.build_team(
                agents=self.agents,
                client=self.client,
                planner=self.planner_agent,
                orchestrator=self.orchestrator_agent
            )
        elif self.architecture == Architecture.DECENTRALIZED:
            # Decentralized architecture not yet implemented for OpenAI Agents
            self.team = strategy.build_team(
                agents=self.agents,
                client=self.client
            )

    async def run(self, task, verbose: bool = False, assessment: list[str] = None, attack_detector=None):
        """
        Run the team on the given task using the planner-orchestrator pattern.

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
        import asyncio
        from agents import Runner

        # Get the orchestrator output type
        OrchestratorOutput = self.orchestrator_agent.output_type

        # Initialize log buffer
        logs = []

        # Step 1: Create a plan using the planner
        log_msg = f"Creating plan for task: {task}"
        logs.append(log_msg)
        if verbose:
            logger.info(log_msg)

        planner_result = await Runner.run(self.planner_agent, task)
        plan = planner_result.final_output

        log_msg = f"Plan created: {plan}"
        logs.append(log_msg)
        if verbose:
            logger.info(log_msg)

        # Step 2: Execute the plan using orchestrator
        done = False
        turn_count = 0
        max_turns = self.max_turns if self.max_turns else DEFAULT_MAX_TURNS

        agent_outputs = [
            {
                "role": "user",
                "content": f"Here is the Plan given by the planner to achieve the task: #PLAN:\n{plan}\n\nHere is the task:\n\n# TASK\n{task}"
            }
        ]

        result = {
            "stop_reason": "",
            "messages": [
                {"source": "orchestrator", "content": f"Task started. Plan: {plan}"}
            ]
        }

        while not done:
            if turn_count >= max_turns:
                log_msg = f"Max turns ({max_turns}) reached. Stopping the task."
                logs.append(log_msg)
                if verbose:
                    logger.info(log_msg)
                result["stop_reason"] = "Max turns reached"
                break

            turn_count += 1

            # Call orchestrator to decide next step
            # await asyncio.sleep(1)
            log_msg = f"\nTurn {turn_count}: Calling orchestrator..."
            logs.append(log_msg)
            if verbose:
                logger.info(log_msg)

            orchestrator_result = await Runner.run(
                self.orchestrator_agent,
                agent_outputs
            )

            orchestrator_output = orchestrator_result.final_output_as(OrchestratorOutput)
            log_msg = f"Orchestrator decision: delegate_to={orchestrator_output.delegate_to}, done={orchestrator_output.done}"
            logs.append(log_msg)
            if verbose:
                logger.info(log_msg)

            if orchestrator_output.done:
                log_msg = f"Task completed. Stop reason: {orchestrator_output.stop_reason}"
                logs.append(log_msg)
                if verbose:
                    logger.info(log_msg)
                result["stop_reason"] = orchestrator_output.stop_reason
                done = True
                break

            # Validate agent exists
            agent_names = [a.name for a in self.agents]
            if orchestrator_output.delegate_to not in agent_names:
                log_msg = f"Cannot delegate to '{orchestrator_output.delegate_to}'. Expected one of {agent_names}"
                logs.append(log_msg)
                if verbose:
                    logger.info(log_msg)
                agent_outputs.append({
                    "role": "user",
                    "content": f"Cannot delegate to '{orchestrator_output.delegate_to}'. Expected one of {agent_names}"
                })
                continue

            # Execute delegated task
            log_msg = f"Delegating to {orchestrator_output.delegate_to}: {orchestrator_output.delegate_task}"
            logs.append(log_msg)
            if verbose:
                logger.info(log_msg)

            # Find the agent
            delegated_agent = None
            for agent in self.agents:
                if agent.name == orchestrator_output.delegate_to:
                    delegated_agent = agent
                    break

            # Add task to agent context
            self.agent_context[orchestrator_output.delegate_to].append({
                "role": "user",
                "content": orchestrator_output.delegate_task
            })

            # Run the agent with context if it's a special agent
            run_kwargs = {}
            if orchestrator_output.delegate_to in getattr(self, 'special_agent_contexts', {}):
                run_kwargs['context'] = self.special_agent_contexts[orchestrator_output.delegate_to]

            agent_result = await Runner.run(
                delegated_agent,
                self.agent_context[orchestrator_output.delegate_to],
                **run_kwargs
            )

            # Update agent context with result
            self.agent_context[orchestrator_output.delegate_to] = agent_result.to_input_list()

            # Add result to messages
            result['messages'].append({
                "source": orchestrator_output.delegate_to,
                "content": agent_result.final_output
            })

            log_msg = f"Output from {orchestrator_output.delegate_to}: {agent_result.final_output}"
            logs.append(log_msg)
            if verbose:
                logger.info(log_msg)

            # Provide feedback to orchestrator
            agent_outputs.append({
                "role": "user",
                "content": f"The output of {orchestrator_output.delegate_to}: {agent_result.final_output}"
            })

        # Add logs to result
        result["logs"] = "\n".join(logs)

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
                    framework="openai-agents"
                )
                if 'assessment' not in result:
                    result['assessment'] = {}
                result['assessment'].update(new_assessments)

        # Always include execution trace in results
        result['execution_trace'] = self.execution_trace

        # Cleanup special agent contexts
        await self._cleanup_special_agents()

        return result

    async def _cleanup_special_agents(self):
        """Cleanup all special agent contexts."""
        if hasattr(self, 'special_agent_contexts'):
            for agent_name, context in self.special_agent_contexts.items():
                try:
                    await context.cleanup()
                except Exception as e:
                    # Log but don't fail on cleanup errors
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(f"Error cleaning up special agent '{agent_name}': {e}")

    async def run_stream(self, task, verbose: bool = False, assessment: list[str] = None, attack_detector=None):
        """
        Run the team on the given task with streaming output.
        For now, this delegates to run() as OpenAI Agents doesn't have built-in streaming for multi-agent.

        Args:
            task: The task to execute
            verbose: If True, print logs to console. If False, only return logs.
            assessment: Optional list of assessment methods to run (e.g., ["aria", "dharma"])
            attack_detector: Optional function to detect attacks

        Returns:
            dict: Execution result with logs, messages, and optional assessment scores
        """
        # OpenAI Agents framework doesn't have native streaming for orchestrator pattern
        # Delegate to run() method
        return await self.run(task, verbose=verbose, assessment=assessment, attack_detector=attack_detector)






























# from safe_agents.team import Team
# from autogen_agentchat.conditions import HandoffTermination
# from autogen_agentchat.teams import Swarm
# from autogen_agentchat.base import Handoff as HandoffBase
# from autogen_agentchat.agents import AssistantAgent
# # TODO[NIRMIT]: Update the following classes to child classes created here itself and not in the lib sirectly
# # from autogen_ext.agents.file_surfer import SwarmFileSurfer
# # from autogen_ext.agents.web_surfer import SwarmMultimodalWebSurfer
# # from autogen_ext.agents.magentic_one._magentic_one_coder_agent import SwarmMagenticOneCoderAgent

# class SwarmTeam(Team):
#     def run(self, task):
#         return self.agents.run(
#             task=f"Execute the following task using the ToolUser agent if it has a relevant tool. If not, try other agents. Task: {task.prompt}.",
#             cancellation_token=None,
#         )

# def build_team(task, model, tools):
#     coder_h = HandoffBase(
#         target="coder",
#         description="A helpful and general-purpose AI assistant that has strong language skills, Python skills, and Linux command line skills. It can not run or implement the code it generates. Handoff to other agents for implementation or further use.",
#     )   

#     file_surfer_h = HandoffBase(
#         target="file_surfer",
#         description="An agent that can view and read local files. It can not implement or write. Handoff to websurfer agent for implementation.",
#     )

#     web_surfer_h = HandoffBase(
#         target="web_surfer",
#         description="A helpful assistant with access to web sites like Gitlab, Reddit, Shopping, Shopping Admin. This can implement tasks by interacting with these websites.",
#     )

#     presenter_h = HandoffBase(
#         target="presenter",
#         description="This is a presenter. This will present the results to the user. This will end the task.",
#     )

#     tool_user_h = HandoffBase(
#         target="tool_user",
#         description="This is a ToolUser agent. This will use the tools to solve the task.",
#     )

#     coder = SwarmMagenticOneCoderAgent(
#         "coder",
#         model_client=model.get_client(),
#         handoffs=[web_surfer_h, file_surfer_h, tool_user_h, presenter_h]
#     )

#     file_surfer = SwarmFileSurfer(
#         "file_surfer",
#         model_client=model.get_client(),
#         handoffs=[coder_h, web_surfer_h, tool_user_h, presenter_h]
#     )

#     web_surfer = SwarmMultimodalWebSurfer(
#         "web_surfer",
#         model_client=model.get_client(),
#         handoffs=[coder_h, file_surfer_h, tool_user_h, presenter_h]
#     )

#     tool_user = AssistantAgent(
#         "tool_user",
#         model_client=model.get_client(),
#         tools=[t.func for t in tools],
#         system_message="Use tools to solve tasks.",
#         handoffs=[coder_h, file_surfer_h, web_surfer_h, presenter_h]
#     )

#     delegator = AssistantAgent(
#         "delegator",
#         model_client=model.get_client(),
#         system_message="You are a Delegator. Delegate the task to one of the agents.",
#         handoffs=[coder_h, file_surfer_h, web_surfer_h, tool_user_h]
#     )

#     presenter = AssistantAgent(
#         "presenter",
#         model_client=model.get_client(),
#         system_message="You are a results Presenter. You will End the task.",
#         description="This is a presenter. This will present the output to the user. This will end the task.",
#     )

#     termination = HandoffTermination(target="presenter")
#     team = Swarm([delegator, tool_user, web_surfer, file_surfer, coder, presenter], termination_condition=termination, max_turns=40)
#     return SwarmTeam(team)
