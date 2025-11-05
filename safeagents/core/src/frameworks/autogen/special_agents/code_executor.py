from typing import List, Optional
from autogen_agentchat.agents import CodeExecutorAgent
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from autogen_core.code_executor._base import CodeBlock
from autogen_core._cancellation_token import CancellationToken
from autogen_ext.code_executors._common import CommandLineCodeResult

# TODO: Add support for Handoff messages in CodeExecutor Agent


class TrackedLocalCommandLineCodeExecutor(LocalCommandLineCodeExecutor):
    """
    Extended LocalCommandLineCodeExecutor that supports tool call tracking for SafeAgents.

    This executor wraps the execute_code_blocks method to track code execution,
    enabling attack detection and logging in the SafeAgents framework.
    """

    def __init__(self, *args, tool_call_tracker: callable = None, agent_name: str = None, **kwargs):
        """
        Initialize the tracked code executor.

        Args:
            tool_call_tracker: Optional callback for tracking code execution
            agent_name: Name of the agent using this executor
            *args, **kwargs: Passed to LocalCommandLineCodeExecutor
        """
        super().__init__(*args, **kwargs)
        self._tool_call_tracker = tool_call_tracker
        self._agent_name = agent_name

    async def execute_code_blocks(
        self,
        code_blocks: List[CodeBlock],
        cancellation_token: CancellationToken
    ) -> CommandLineCodeResult:
        """
        Execute code blocks and track the execution.

        Overrides the parent method to add tracking for SafeAgents integration.
        """
        # Extract code information for tracking
        code_content = []
        for block in code_blocks:
            code_content.append({
                'language': block.language,
                'code': block.code[:200] + '...' if len(block.code) > 200 else block.code  # Truncate long code
            })

        # Execute the code
        try:
            result = await super().execute_code_blocks(code_blocks, cancellation_token)

            # Track the successful execution
            if self._tool_call_tracker:
                self._tool_call_tracker(
                    tool_name='execute_code',
                    args={'code_blocks': code_content},
                    result=f"exit_code={result.exit_code}, output={str(result.output)[:200]}",  # Truncate output
                    agent_name=self._agent_name
                )

            return result

        except Exception as e:
            # Track the failed execution
            if self._tool_call_tracker:
                self._tool_call_tracker(
                    tool_name='execute_code',
                    args={'code_blocks': code_content},
                    result=f"ERROR: {str(e)}",
                    agent_name=self._agent_name
                )
            raise