from abc import ABC, abstractmethod
from typing import Dict, Type, Any, Optional, Callable
import logging
from datetime import datetime

from ..models.agent import Agent
from ..models.tool import Tool
from .framework_types import Framework
from .architectures import Architecture

logger = logging.getLogger(__name__)


class TeamRegistry:
    """
    Registry for managing Team subclasses and their associated frameworks.
    This allows for dynamic registration of new frameworks without modifying core code.
    """
    _registry: Dict[str, Type['Team']] = {}
    
    @classmethod
    def register(cls, framework: Framework | str, team_class: Type['Team']) -> None:
        """Register a Team subclass for a specific framework."""
        framework_key = framework.value if isinstance(framework, Framework) else str(framework).lower()
        cls._registry[framework_key] = team_class
        logger.debug(f"Registered framework '{framework_key}' with class {team_class.__name__}")
    
    @classmethod
    def get_team_class(cls, framework: Framework | str) -> Type['Team']:
        """Get the Team subclass for a specific framework."""
        framework_key = framework.value if isinstance(framework, Framework) else str(framework).lower()
        if framework_key not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(f"Unknown framework: {framework}. Available frameworks: {available}")
        return cls._registry[framework_key]
    
    @classmethod
    def list_frameworks(cls) -> list[str]:
        """List all registered frameworks."""
        return list(cls._registry.keys())
    
    @classmethod
    def is_registered(cls, framework: Framework | str) -> bool:
        """Check if a framework is registered."""
        framework_key = framework.value if isinstance(framework, Framework) else str(framework).lower()
        return framework_key in cls._registry


def register_framework(framework: Framework | str):
    """
    Decorator for automatically registering Team subclasses.
    
    Usage:
    @register_framework(Framework.NEW_FRAMEWORK)
    class TeamNewFramework(Team):
        pass
    """
    def decorator(cls):
        TeamRegistry.register(framework, cls)
        return cls
    return decorator


class Team(ABC):
    """
    Abstract base class representing a multi-agent system with a specific architecture.
    All framework-specific implementations should inherit from this class.
    """

    def __init__(self, agents: Agent, framework: Framework | str, architecture: Architecture | str,
                 termination_condition=None, max_turns=None):
        self.team = None  # Will be set in instantiate_team
        self.agents = agents

        if isinstance(framework, str):
            try:
                framework = Framework(framework)
            except ValueError:
                raise ValueError(f"Invalid framework '{framework}'. Must be one of {[f.value for f in Framework]}")
        self.framework = framework

        if isinstance(architecture, str):
            try:
                architecture = Architecture(architecture)
            except ValueError:
                raise ValueError(f"Invalid architecture '{architecture}'. Must be one of {[a.value for a in Architecture]}")
        self.architecture = architecture

        self.termination_condition = termination_condition
        self.max_turns = max_turns

        # Execution trace for attack detection
        self.execution_trace = {
            'tool_calls': [],
            'bash_commands': [],
            'logs': '',
            'messages': [],
            'framework': framework.value if hasattr(framework, 'value') else str(framework),
            'task': None,
            'start_time': None,
            'end_time': None
        }

        self.client = self.get_client()
        if self.architecture == Architecture.DECENTRALIZED:
            self.clients_of_agents = {}
            self.create_delegator_presenter_agents()
        self.instantiate_agents()
        self.instantiate_team()
        assert self.team is not None, "Team instantiation failed."
    
    def __new__(cls, framework=None, *args, **kwargs):
        if cls is Team:  # Only redirect if directly instantiating Team
            team_class = TeamRegistry.get_team_class(framework)
            return team_class(*args, **kwargs)
        return super().__new__(cls)
    
    @classmethod
    def create(cls, framework: Framework | str, agents, architecture: Architecture | str, **kwargs) -> 'Team':
        """
        Factory method to create appropriate team instance.
        Alternative to using the Team() constructor directly.

        Args:
            framework: Framework to use (e.g., "autogen", "langgraph", "openai-agents")
            agents: List of agents or single agent
            architecture: Architecture type (e.g., "centralized", "decentralized")
            **kwargs: Additional arguments (termination_condition, max_turns, etc.)
        """
        team_class = TeamRegistry.get_team_class(framework)
        return team_class(agents, architecture, **kwargs)

    def get_llm_config(self) -> Dict[str, Any]:
        """
        Get LLM configuration from environment variables.
        Uses shared utility for consistency across frameworks and assessment.
        """
        from ..clients.azure_openai import get_azure_config
        return get_azure_config()

    @abstractmethod
    def get_client(self) -> Any:
        """
        Returns the appropriate client based on the framework.
        Must be implemented by each framework-specific subclass.
        """
        pass

    def track_tool_call(self, tool_name: str, args: dict, result: Any, agent_name: str = None):
        """
        Track a tool call for attack detection.

        Args:
            tool_name: Name of the tool that was called
            args: Arguments passed to the tool
            result: Result returned by the tool
            agent_name: Name of the agent that called the tool (optional)
        """
        self.execution_trace['tool_calls'].append({
            'name': tool_name,
            'args': args,
            'result': str(result)[:500] if result else None,  # Truncate long results
            'timestamp': datetime.now().isoformat(),
            'agent': agent_name
        })

    def track_bash_command(self, command: str, output: str, exit_code: int):
        """
        Track a bash command execution for attack detection.

        Args:
            command: The bash command that was executed
            output: Output from the command
            exit_code: Exit code of the command
        """
        self.execution_trace['bash_commands'].append({
            'command': command,
            'output': output[:500] if output else None,  # Truncate long output
            'exit_code': exit_code,
            'timestamp': datetime.now().isoformat()
        })

    def _wrap_tool_for_tracking(self, tool: Tool, agent_name: str = None) -> Tool:
        """
        Wrap a tool to track its execution.

        This is called during agent instantiation to automatically track
        all tool calls for attack detection.

        Args:
            tool: The tool to wrap
            agent_name: Name of the agent this tool belongs to

        Returns:
            Tool: The wrapped tool (same object, but with modified func)
        """
        import inspect
        import functools

        # Store original function
        original_func = tool.func

        # Check if original is async or sync
        if inspect.iscoroutinefunction(original_func):
            @functools.wraps(original_func)
            async def tracked_func(*args, **kwargs):
                """Execute async tool and track the call."""
                try:
                    # Execute the original tool
                    result = await original_func(*args, **kwargs)

                    # Track the successful call
                    self.track_tool_call(
                        tool_name=tool.name,
                        args=kwargs,
                        result=result,
                        agent_name=agent_name
                    )

                    return result
                except Exception as e:
                    # Track failed call
                    self.track_tool_call(
                        tool_name=tool.name,
                        args=kwargs,
                        result=f"ERROR: {str(e)}",
                        agent_name=agent_name
                    )
                    raise
        else:
            @functools.wraps(original_func)
            def tracked_func(*args, **kwargs):
                """Execute sync tool and track the call."""
                try:
                    # Execute the original tool
                    result = original_func(*args, **kwargs)

                    # Track the successful call
                    self.track_tool_call(
                        tool_name=tool.name,
                        args=kwargs,
                        result=result,
                        agent_name=agent_name
                    )

                    return result
                except Exception as e:
                    # Track failed call
                    self.track_tool_call(
                        tool_name=tool.name,
                        args=kwargs,
                        result=f"ERROR: {str(e)}",
                        agent_name=agent_name
                    )
                    raise

        # Replace func with tracked version
        tool.func = tracked_func
        return tool

    def create_delegator_presenter_agents(self) -> None:
        """
        Create delegator and presenter agents for decentralized architecture.
        Must be implemented by each framework-specific subclass.
        Should create agents with names 'delegator_safeagents' and 'presenter_safeagents'.
        These agents should be set as self.delegator and self.presenter
        Should populate self.handoff_bases with {agent_name: handoff_base} pairs.
        """
        raise NotImplementedError(f"create_delegator_presenter_agents not implemented for {self.framework.value} framework")
    
    @abstractmethod
    def instantiate_agents(self) -> None:
        """
        Instantiate agents based on the framework and architecture.
        Must be implemented by each framework-specific subclass.
        """
        pass

    @abstractmethod
    def process_tool(self, tool: Tool) -> Any:
        """
        Process the generic SafeAgent's Tool of the agents to specific framework's tool.
        Must be implemented by each framework-specific subclass.
        """
        pass

    @abstractmethod
    def instantiate_team(self) -> None:
        """
        Instantiate the team based on the framework and architecture.
        Must be implemented by each framework-specific subclass.
        """
        pass
    
    @abstractmethod
    async def run(self, task, verbose: bool = False, assessment: list[str] = None,
                  attack_detector: Optional[Callable[[Dict[str, Any]], bool]] = None) -> Any:
        """
        Execute the task using the team of agents.

        Args:
            task: The task to execute
            verbose: If True, print logs during execution. If False, silent execution.
            assessment: Optional list of assessment methods to run on logs (e.g., ["aria", "dharma"])
            attack_detector: Optional function to detect attacks. Takes execution_trace dict,
                           returns True if attack detected. If attack detected and "aria" in
                           assessment, ARIA score is set to "4".

        Returns:
            dict: Execution result with keys:
                - logs: Execution logs
                - messages: List of messages exchanged
                - stop_reason: Why execution stopped
                - execution_trace: Dict with tool_calls, bash_commands, etc.
                - attack_detected: bool (if attack_detector was provided)
                - assessment: Dict of assessment scores (if assessment requested)

        Must be implemented by each framework-specific subclass.

        Note:
            Framework implementations should:
            1. Execute the task and populate self.execution_trace
            2. Call _run_attack_detection() before assessment
            3. Return result dict with all required fields
        """
        pass

    def _run_attack_detection(self, result: Dict[str, Any],
                             attack_detector: Optional[Callable[[Dict[str, Any]], bool]],
                             assessment: Optional[list[str]]) -> Dict[str, Any]:
        """
        Run attack detection and handle ARIA scoring.

        This is a helper method that framework implementations should call
        after task execution but before assessment.

        Args:
            result: The result dict from task execution
            attack_detector: User-provided attack detector function
            assessment: List of assessment methods

        Returns:
            dict: Updated result with attack_detected field and potentially ARIA=4
        """
        if not attack_detector:
            result['attack_detected'] = False
            return result

        # Populate execution trace with final data
        self.execution_trace['logs'] = result.get('logs', '')
        self.execution_trace['messages'] = result.get('messages', [])
        self.execution_trace['task'] = self.execution_trace.get('task') or result.get('task', '')

        # Run attack detection
        try:
            is_attack = attack_detector(self.execution_trace)
            result['attack_detected'] = bool(is_attack)

            # If attack detected and ARIA assessment requested, set ARIA to 4
            if is_attack and assessment and 'aria' in assessment:
                if 'assessment' not in result:
                    result['assessment'] = {}
                result['assessment']['aria'] = '4'
                logger.info("Attack detected! ARIA score set to 4.")

        except Exception as e:
            logger.error(f"Error running attack detector: {e}")
            result['attack_detected'] = None
            result['attack_detector_error'] = str(e)

        # Always include execution trace in result
        result['execution_trace'] = self.execution_trace

        return result

    async def run_stream(self, task, verbose: bool = False, assessment: list[str] = None) -> Any:
        """
        Stream execution for interactive frameworks.

        Args:
            task: The task to execute
            verbose: If True, print logs during execution. If False, silent execution.
            assessment: Optional list of assessment methods to run on logs (e.g., ["aria", "dharma"])

        Returns:
            dict: Execution result with logs, messages, and optional assessment scores

        Optional - can be overridden by subclasses that support streaming.
        """
        raise NotImplementedError(f"Streaming not supported for {self.framework.value} framework")
