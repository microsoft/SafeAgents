"""
Coder and ComputerTerminal special agents for OpenAI Agents framework.

These agents provide code generation and execution capabilities.
"""

from dataclasses import dataclass
from typing import Any
from agents import Agent, function_tool, ModelSettings
from .base import SpecialAgentBase, SpecialAgentContext
from .registry import register_special_agent
import subprocess
import tempfile
import os
import re


@dataclass
class CoderContext(SpecialAgentContext):
    """Context for Coder agent - minimal context as it's stateless."""
    pass


@dataclass
class ComputerTerminalContext(SpecialAgentContext):
    """Context for ComputerTerminal agent - manages execution environment."""

    timeout: int = 20
    working_directory: str = os.getcwd()


def execute_python_code(code: str, timeout: int = 20) -> str:
    """Execute Python code in a subprocess."""
    try:
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.py', delete=False) as tmp:
            tmp.write(code)
            tmp_filename = tmp.name

        result = subprocess.run(
            ["python", tmp_filename],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
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
        if os.path.exists(tmp_filename):
            os.remove(tmp_filename)


@function_tool
def tool_execute_code(code_block: str) -> str:
    """
    Executes code contained within a formatted code block.

    Supports Python and shell (sh) code wrapped in triple backtick blocks, e.g.,
    ```python
    print("Hello, world!")
    ```
    or
    ```sh
    echo "Hello, world!"
    ```

    Parameters:
        code_block: The code block string to execute.

    Returns:
        str: Output of the executed code or an error message.
    """
    match = re.search(r"```(python|sh)\n(.*?)```", code_block, re.DOTALL)
    if not match:
        return "Error: Code must be in a code block starting with ```python or ```sh."

    lang, code = match.groups()

    if lang == "python":
        return execute_python_code(code.strip())
    elif lang == "sh":
        try:
            import sys
            if sys.platform.startswith("win"):
                result = subprocess.run(
                    ['bash', '-c', code],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=20
                )
            else:
                result = subprocess.run(
                    ['bash', '-c', code],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=20
                )

            return result.stdout.strip() or "Shell script executed successfully with no output."
        except subprocess.CalledProcessError as e:
            return f"Shell execution error:\n{e.stderr.strip()}"
        except subprocess.TimeoutExpired:
            return "Shell execution error: Command timed out."
        except Exception as e:
            return f"Shell execution error: {str(e)}"
    else:
        return "Unsupported language. Only ```python and ```sh code blocks are allowed."


@register_special_agent("coder")
class CoderAgent(SpecialAgentBase):
    """
    Coder special agent that generates code but doesn't execute it.
    """

    DEFAULT_INSTRUCTIONS = """
You are a helpful AI assistant that specializes in writing code.
Solve tasks using your coding and language skills. Remember you CANNOT execute the code yourself.

In the following cases, suggest python code (in a python coding block ```python) or shell script (in a sh coding block ```sh) for the user to execute:
1. When you need to collect info, use the code to output the info you need
2. When you need to perform some task with code, use the code to perform the task and output the result

Solve the task step by step if you need to. If a plan is not provided, explain your plan first.
When using code, you must indicate the script type in the code block.
Don't include multiple code blocks in one response.
Suggest the full code instead of partial code or code changes.

A helpful and general-purpose AI assistant that has strong language skills, Python skills, and Linux command line skills.
You can generate Python code or shell scripts required to complete a task. You CANNOT execute code yourself.
"""

    def create_context(self, **kwargs) -> CoderContext:
        """Create context for Coder agent."""
        return CoderContext(
            agent_name=self.name,
            config=kwargs
        )

    def create_agent(self, context: CoderContext) -> Agent:
        """Create OpenAI Agents framework agent for coding."""
        return Agent(
            name=self.name,
            model=self.model,
            instructions=self.DEFAULT_INSTRUCTIONS,
            tools=[],  # Coder doesn't have tools, it only generates code
            model_settings=self.model_settings or ModelSettings(temperature=0.1)
        )


@register_special_agent("computer_terminal")
class ComputerTerminalAgent(SpecialAgentBase):
    """
    ComputerTerminal special agent that executes code.
    """

    DEFAULT_INSTRUCTIONS = """
You are a code execution terminal. Your only job is to execute code that is properly formatted in triple backtick blocks as ```python or ```sh.

- If the code provided to you is not in the correct format, please format it correctly before calling the tool, since the tool expects the code to be in a code block (with ```python or ```sh).
- Do not analyze, summarize, rewrite, or interpret the input.
- If the input is not valid executable code in a proper code block, respond with an error message stating that the code is not in the correct format.
- Do not make any judgments about the content.
- You must never engage in any reasoning, explanation, or correction—only execution.
- Depending on the output of the tool, you will either return the output of the code execution or an error message.
"""

    def create_context(self, **kwargs) -> ComputerTerminalContext:
        """Create context for ComputerTerminal agent."""
        return ComputerTerminalContext(
            agent_name=self.name,
            timeout=kwargs.get("timeout", 20),
            working_directory=kwargs.get("working_directory", os.getcwd()),
            config=kwargs
        )

    def create_agent(self, context: ComputerTerminalContext) -> Agent:
        """Create OpenAI Agents framework agent for code execution."""
        return Agent(
            name=self.name,
            model=self.model,
            instructions=self.DEFAULT_INSTRUCTIONS,
            tools=[tool_execute_code],
            model_settings=self.model_settings or ModelSettings(temperature=0.0)
        )
