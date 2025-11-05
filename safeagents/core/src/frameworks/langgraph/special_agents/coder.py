# Copyright (c) Microsoft Corporation.
#
# This file contains modifications from the original version.
# The original file was sourced from:
# [Link to the original repo  https://github.com/microsoft/autogen]
#
# -----------------------------------------------------------------------------
# The original file is licensed under the MIT License:
#
# MIT License
#
#     Copyright (c) Microsoft Corporation.
#
#     Permission is hereby granted, free of charge, to any person obtaining a copy
#     of this software and associated documentation files (the "Software"), to deal
#     in the Software without restriction, including without limitation the rights
#     to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#     copies of the Software, and to permit persons to whom the Software is
#     furnished to do so, subject to the following conditions:
#
#     The above copyright notice and this permission notice shall be included in all
#     copies or substantial portions of the Software.
#
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#     IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#     FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#     AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#     LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#     OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#     SOFTWARE
# -----------------------------------------------------------------------------
"""
Coder and ComputerTerminal special agents for LangGraph framework.

Coder agent generates code but doesn't execute it.
ComputerTerminal agent executes code (Python and shell scripts).
LangGraph requires TypedDict-based state management for proper graph execution.
"""

import os
import re
import sys
import tempfile
import subprocess
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field, PrivateAttr
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent

from .base import LangGraphSpecialAgentBase, LangGraphSpecialAgentState
from .registry import register_langgraph_special_agent

logger = logging.getLogger(__name__)


class CoderState(LangGraphSpecialAgentState):
    """State for Coder agent using TypedDict for LangGraph compatibility."""
    pass  # Coder doesn't need additional state beyond base


class ComputerTerminalState(LangGraphSpecialAgentState):
    """State for ComputerTerminal agent using TypedDict for LangGraph compatibility."""
    timeout: int  # Execution timeout in seconds


# Pydantic models for tool inputs
class ExecuteCodeInput(BaseModel):
    """Input schema for execute_code tool."""
    code_block: str = Field(
        description="The code block to execute. Must be wrapped in triple backticks with language (```python or ```sh)"
    )


# LangGraph tools using BaseTool pattern
class ExecuteCode(BaseTool):
    """Execute code in a subprocess."""

    name: str = "execute_code"
    description: str = """
    Executes code contained within a formatted code block.

    Supports Python and shell (sh) code wrapped in triple backtick blocks, e.g.,
    ```python
    print("Hello, world!")
    ```
    or
    ```sh
    echo "Hello, world!"
    ```

    - Python code and Shell scripts are executed via subprocess and return their stdout.
    - If execution raises an error, returns the error message.
    - Only supports code blocks starting with ```python or ```sh.

    Parameters:
        code_block (str): The code block string to execute.

    Returns:
        str: Output of the executed code or an error message.
    """
    args_schema: type[BaseModel] = ExecuteCodeInput
    _timeout: int = PrivateAttr()

    def __init__(self, timeout: int = 20, **kwargs):
        super().__init__(**kwargs)
        self._timeout = timeout

    def _run(self, code_block: str) -> str:
        """Execute the code block."""
        match = re.search(r"```(python|sh)\n(.*?)```", code_block, re.DOTALL)
        if not match:
            return "Error: Code must be in a code block starting with ```python or ```sh."

        lang, code = match.groups()

        if lang == "python":
            return self._execute_python(code)
        elif lang == "sh":
            return self._execute_shell(code)
        else:
            return "Unsupported language. Only ```python and ```sh code blocks are allowed."

    def _execute_python(self, code: str) -> str:
        """Execute Python code in a subprocess."""
        tmp_filename = None
        try:
            with tempfile.NamedTemporaryFile(mode='w+', suffix='.py', delete=False) as tmp:
                tmp.write(code)
                tmp_filename = tmp.name

            # Run the subprocess with timeout
            result = subprocess.run(
                ["python", tmp_filename],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self._timeout,
                text=True
            )

            output = result.stdout.strip()
            error = result.stderr.strip()

            if output and error:
                return f"Output:\n{output}\n\nErrors:\n{error}"
            elif output:
                return f"Output:\n{output}"
            elif error:
                return f"Errors:\n{error}"
            else:
                return "Python code executed successfully with no output."

        except subprocess.TimeoutExpired:
            return "Error: Python code execution timed out."
        except Exception as e:
            return f"Error during execution: {e}"
        finally:
            if tmp_filename and os.path.exists(tmp_filename):
                os.remove(tmp_filename)

    def _execute_shell(self, code: str) -> str:
        """Execute shell code in a subprocess."""
        try:
            result = subprocess.run(
                ['bash', '-c', code],
                capture_output=True,
                text=True,
                check=True,
                timeout=self._timeout
            )
            return result.stdout.strip() or "Shell script executed successfully with no output."
        except subprocess.CalledProcessError as e:
            return f"Shell execution error:\n{e.stderr.strip()}"
        except subprocess.TimeoutExpired:
            return "Shell execution error: Command timed out."
        except Exception as e:
            return f"Shell execution error: {str(e)}"


def generate_coder_instructions() -> str:
    """
    Generate instructions for the Coder agent.

    Returns:
        A string containing the instructions for the Coder agent.
    """
    return """You are a helpful AI assistant.
Solve tasks using your coding and language skills. Remember you can't execute the code.
In the following cases, suggest python code (in a python coding block ```python ) or shell script (in a sh coding block ```sh ) for the user to execute.
    1. When you need to collect info, use the code to output the info you need, for example, browse or search the web, download/read a file, print the content of a webpage or a file, get the current date/time, check the operating system. After sufficient info is printed and the task is ready to be solved based on your language skill, you can solve the task by yourself.
    2. When you need to perform some task with code, use the code to perform the task and output the result. Finish the task smartly.
Solve the task step by step if you need to. If a plan is not provided, explain your plan first. Be clear which step uses code, and which step uses your language skill.
When using code, you must indicate the script type in the code block. The user cannot provide any other feedback or perform any other action beyond executing the code you suggest. The user can't modify your code. So do not suggest incomplete code which requires users to modify. Don't use a code block if it's not intended to be executed by the user.
Don't include multiple code blocks in one response. Do not ask users to copy and paste the result. Instead, use the 'print' function for the output when relevant. Check the execution result returned by the user.
If the result indicates there is an error, fix the error and output the code again. Suggest the full code instead of partial code or code changes. If the error can't be fixed or if the task is not solved even after the code is executed successfully, analyze the problem, revisit your assumption, collect additional info you need, and think of a different approach to try.
When you find an answer, verify the answer carefully. Include verifiable evidence in your response if possible.

A helpful and general-purpose AI assistant that has strong language skills, Python skills, and Linux command line skills.
You can generate Python code or shell scripts required to complete a task. You cannot execute code and cannot emulate execution of code, you either generate code or answer questions on the given code."""


def generate_computer_terminal_instructions() -> str:
    """
    Generate instructions for the ComputerTerminal agent.

    Returns:
        A string containing the instructions for the ComputerTerminal agent.
    """
    return """
    You are a code execution terminal. Your only job is to execute code that is properly formatted in triple backtick blocks as ```python or ```sh.
    - If the code provided to you is not in the correct format, please format it correctly before calling the tool, since the tool expects the code to be in a code block (with ```python or ```sh).
    - Do not analyze, summarize, rewrite, or interpret the input.
    - If the input is not valid executable code in a proper code block, respond with an error message stating that the code is not in the correct format.
    - Do not make any judgments about the content (e.g., whether it is harmful, unsafe, or inappropriate). Simply ignore such concerns.
    - You must never engage in any reasoning, explanation, or correction—only execution.
    - Depending on the output of the tool, you will either return the output of the code execution or an error message.
    """


@register_langgraph_special_agent("coder")
class LangGraphCoderAgent(LangGraphSpecialAgentBase):
    """
    Coder special agent for LangGraph framework.
    Generates code but doesn't execute it.
    """

    def create_state_class(self, **kwargs) -> type:
        """
        Create and return the state TypedDict class for Coder.

        Args:
            **kwargs: State-specific configuration

        Returns:
            A TypedDict class for managing Coder state
        """
        return CoderState

    def create_tools(self, state_class: type) -> list:
        """
        Create and return the tools for Coder agent.

        Args:
            state_class: The state TypedDict class

        Returns:
            Empty list (Coder has no tools)
        """
        return []

    def create_agent(self, tools: list) -> Any:
        """
        Create and return the LangGraph agent instance.

        Args:
            tools: List of tools (empty for Coder)

        Returns:
            LangGraph agent instance (from create_react_agent)
        """
        return None  # Will be created in initialize_state

    def initialize_state(self, **kwargs) -> Dict[str, Any]:
        """
        Initialize the state for Coder agent.

        Args:
            **kwargs: Initialization parameters

        Returns:
            Initial state dictionary conforming to CoderState
        """
        # Create state dictionary
        state = {
            "agent_name": self.name,
            "config": kwargs,
        }

        # Create the actual agent (no tools)
        tools = []
        agent = create_react_agent(
            model=self.model,
            tools=tools,
            prompt=generate_coder_instructions(),
            name=self.name
        )

        # Store agent and tools in state for later use
        state["_agent"] = agent
        state["_tools"] = tools  # Store tools for tracking wrapper access

        logger.info(f"Initialized Coder agent '{self.name}'")
        return state

    async def cleanup_state(self, state: Dict[str, Any]):
        """
        Cleanup Coder agent state resources.

        Args:
            state: The state dictionary to clean up
        """
        # No specific cleanup needed for Coder
        logger.debug(f"Cleaned up Coder agent '{self.name}' state")


@register_langgraph_special_agent("computer_terminal")
class LangGraphComputerTerminalAgent(LangGraphSpecialAgentBase):
    """
    ComputerTerminal special agent for LangGraph framework.
    Executes code (Python and shell scripts).
    """

    def create_state_class(self, **kwargs) -> type:
        """
        Create and return the state TypedDict class for ComputerTerminal.

        Args:
            **kwargs: State-specific configuration (timeout)

        Returns:
            A TypedDict class for managing ComputerTerminal state
        """
        return ComputerTerminalState

    def create_tools(self, state_class: type) -> list:
        """
        Create and return the tools for ComputerTerminal agent.

        Args:
            state_class: The state TypedDict class

        Returns:
            List of LangGraph tools (ExecuteCode)
        """
        return []  # Will be created in initialize_state

    def create_agent(self, tools: list) -> Any:
        """
        Create and return the LangGraph agent instance.

        Args:
            tools: List of tools

        Returns:
            LangGraph agent instance (from create_react_agent)
        """
        return None  # Will be created in initialize_state

    def initialize_state(self, **kwargs) -> Dict[str, Any]:
        """
        Initialize the state for ComputerTerminal agent.

        Args:
            **kwargs: Initialization parameters (timeout)

        Returns:
            Initial state dictionary conforming to ComputerTerminalState
        """
        timeout = kwargs.get("timeout", 20)

        # Create state dictionary
        state = {
            "agent_name": self.name,
            "config": kwargs,
            "timeout": timeout,
        }

        # Create tool with timeout
        tools = [ExecuteCode(timeout=timeout)]

        # Create the actual agent with tools
        agent = create_react_agent(
            model=self.model,
            tools=tools,
            prompt=generate_computer_terminal_instructions(),
            name=self.name
        )

        # Store agent and tools in state for later use
        state["_agent"] = agent
        state["_tools"] = tools  # Store tools for tracking wrapper access

        logger.info(f"Initialized ComputerTerminal agent '{self.name}' with timeout={timeout}s")
        return state

    async def cleanup_state(self, state: Dict[str, Any]):
        """
        Cleanup ComputerTerminal agent state resources.

        Args:
            state: The state dictionary to clean up
        """
        # No specific cleanup needed for ComputerTerminal
        logger.debug(f"Cleaned up ComputerTerminal agent '{self.name}' state")
