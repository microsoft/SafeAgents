"""
Attack detection utilities for SafeAgents.

Provides helper functions and a flexible API for users to detect attacks
during agent execution. Users can check if certain tools were called,
run bash scripts to detect side-effects, or write custom detection logic.

Usage:
    # Simple: Check if tools were called
    attack_detector = tools_called(['delete_file', 'send_email'])

    # Bash script: Check filesystem changes
    attack_detector = bash_returns_true('./check_changes.sh')

    # Combine multiple checks
    attack_detector = any_of(
        tools_called(['rm_file']),
        bash_returns_true('./check.sh')
    )

    # Custom logic
    def my_detector(trace):
        # Your logic here
        return True if attack else False
"""

import subprocess
import re
from typing import Callable, List, Dict, Any, Optional
from pathlib import Path

from ...logger import logger


# Type alias for detector functions
AttackDetector = Callable[[Dict[str, Any]], bool]


def tools_called(tool_names: List[str]) -> AttackDetector:
    """
    Check if any of the specified tools were called during execution.

    Args:
        tool_names: List of tool names to check for (e.g., ['delete_file', 'send_email'])

    Returns:
        AttackDetector: Function that returns True if any tool was called

    Example:
        >>> detector = tools_called(['delete_file', 'rm_file'])
        >>> result = team.run(task="...", attack_detector=detector)
    """
    def detector(trace: Dict[str, Any]) -> bool:
        if not trace.get('tool_calls'):
            return False

        called_tools = {call['name'] for call in trace['tool_calls']}
        return any(tool in called_tools for tool in tool_names)

    return detector


def tool_called_with_args(tool_name: str, arg_checks: Dict[str, Any]) -> AttackDetector:
    """
    Check if a specific tool was called with specific argument values.

    Args:
        tool_name: Name of the tool to check
        arg_checks: Dict of argument names and values/patterns to match

    Returns:
        AttackDetector: Function that returns True if tool was called with matching args

    Example:
        >>> # Check if delete_file was called with path="/"
        >>> detector = tool_called_with_args('delete_file', {'path': '/'})
        >>> # Check if send_email was called with 'admin' in recipient
        >>> detector = tool_called_with_args('send_email', {'to': lambda x: 'admin' in x})
    """
    def detector(trace: Dict[str, Any]) -> bool:
        if not trace.get('tool_calls'):
            return False

        for call in trace['tool_calls']:
            if call['name'] != tool_name:
                continue

            # Check if all arg_checks match
            call_args = call.get('args', {})
            match = True

            for arg_name, expected in arg_checks.items():
                if arg_name not in call_args:
                    match = False
                    break

                actual = call_args[arg_name]

                # If expected is callable, use it as a checker function
                if callable(expected):
                    if not expected(actual):
                        match = False
                        break
                # Otherwise check for equality
                elif actual != expected:
                    match = False
                    break

            if match:
                return True

        return False

    return detector


def bash_returns_true(script_path: str, timeout: Optional[int] = 10) -> AttackDetector:
    """
    Run a bash script and check if it returns exit code 0 (indicating attack).

    The bash script should:
    - Return exit code 0 if attack is detected
    - Return exit code 1 (or non-zero) if no attack detected

    Args:
        script_path: Path to bash script to execute
        timeout: Timeout in seconds (default: 10)

    Returns:
        AttackDetector: Function that returns True if script exits with 0

    Example:
        >>> # check_deletions.sh returns 0 if files were deleted
        >>> detector = bash_returns_true('./check_deletions.sh')
        >>> result = team.run(task="...", attack_detector=detector)
    """
    def detector(trace: Dict[str, Any]) -> bool:
        try:
            result = subprocess.run(
                [script_path],
                capture_output=True,
                timeout=timeout,
                text=True
            )
            # Exit code 0 means attack detected
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            logger.warning(f"Attack detection script timed out: {script_path}")
            return False
        except Exception as e:
            logger.warning(f"Error running attack detection script: {e}")
            return False

    return detector


def bash_output_contains(script_path: str, pattern: str, timeout: Optional[int] = 10) -> AttackDetector:
    """
    Run a bash script and check if its output contains a pattern.

    Args:
        script_path: Path to bash script to execute
        pattern: String or regex pattern to search for in output
        timeout: Timeout in seconds (default: 10)

    Returns:
        AttackDetector: Function that returns True if pattern found in output

    Example:
        >>> # Check if output contains "ATTACK"
        >>> detector = bash_output_contains('./check.sh', 'ATTACK')
    """
    def detector(trace: Dict[str, Any]) -> bool:
        try:
            result = subprocess.run(
                [script_path],
                capture_output=True,
                timeout=timeout,
                text=True
            )
            output = result.stdout + result.stderr
            return re.search(pattern, output) is not None
        except subprocess.TimeoutExpired:
            logger.warning(f"Attack detection script timed out: {script_path}")
            return False
        except Exception as e:
            logger.warning(f"Error running attack detection script: {e}")
            return False

    return detector


def log_contains(pattern: str, case_sensitive: bool = True) -> AttackDetector:
    """
    Check if execution logs contain a specific pattern.

    Args:
        pattern: String or regex pattern to search for
        case_sensitive: Whether search is case sensitive (default: True)

    Returns:
        AttackDetector: Function that returns True if pattern found in logs

    Example:
        >>> # Check for deletion messages
        >>> detector = log_contains(r'deleted \d+ files')
        >>> # Case-insensitive search
        >>> detector = log_contains('error', case_sensitive=False)
    """
    def detector(trace: Dict[str, Any]) -> bool:
        logs = trace.get('logs', '')
        if not logs:
            return False

        flags = 0 if case_sensitive else re.IGNORECASE
        return re.search(pattern, logs, flags) is not None

    return detector


def any_of(*detectors: AttackDetector) -> AttackDetector:
    """
    Combine multiple detectors with OR logic.

    Returns True if ANY detector returns True.

    Args:
        *detectors: Variable number of detector functions

    Returns:
        AttackDetector: Combined detector function

    Example:
        >>> detector = any_of(
        ...     tools_called(['delete_file']),
        ...     bash_returns_true('./check.sh'),
        ...     log_contains('attack')
        ... )
    """
    def detector(trace: Dict[str, Any]) -> bool:
        return any(d(trace) for d in detectors)

    return detector


def all_of(*detectors: AttackDetector) -> AttackDetector:
    """
    Combine multiple detectors with AND logic.

    Returns True if ALL detectors return True.

    Args:
        *detectors: Variable number of detector functions

    Returns:
        AttackDetector: Combined detector function

    Example:
        >>> # Attack only if BOTH conditions are met
        >>> detector = all_of(
        ...     tools_called(['send_email']),
        ...     log_contains('attachment.*sensitive')
        ... )
    """
    def detector(trace: Dict[str, Any]) -> bool:
        return all(d(trace) for d in detectors)

    return detector


def none_of(*detectors: AttackDetector) -> AttackDetector:
    """
    Combine multiple detectors with NOR logic.

    Returns True if NONE of the detectors return True.

    Args:
        *detectors: Variable number of detector functions

    Returns:
        AttackDetector: Combined detector function

    Example:
        >>> # Attack if none of these safe tools were used
        >>> detector = none_of(
        ...     tools_called(['read_file', 'search'])
        ... )
    """
    def detector(trace: Dict[str, Any]) -> bool:
        return not any(d(trace) for d in detectors)

    return detector


def count_tool_calls(tool_name: str, threshold: int) -> AttackDetector:
    """
    Check if a tool was called more than a threshold number of times.

    Args:
        tool_name: Name of the tool to count
        threshold: Maximum allowed calls (attack if exceeded)

    Returns:
        AttackDetector: Function that returns True if threshold exceeded

    Example:
        >>> # Attack if delete_file called more than 5 times
        >>> detector = count_tool_calls('delete_file', threshold=5)
    """
    def detector(trace: Dict[str, Any]) -> bool:
        if not trace.get('tool_calls'):
            return False

        count = sum(1 for call in trace['tool_calls'] if call['name'] == tool_name)
        return count > threshold

    return detector


# Example custom detector for documentation
def example_custom_detector(trace: Dict[str, Any]) -> bool:
    """
    Example of a custom attack detector function.

    Users can write their own detector functions that:
    1. Take execution_trace dict as input
    2. Return bool (True = attack, False = safe)

    The trace dict contains:
        - tool_calls: List of {name, args, result, timestamp, agent}
        - bash_commands: List of {command, output, exit_code, timestamp}
        - logs: Full execution logs as string
        - messages: All messages exchanged
        - framework: Framework name
        - task: Task string

    Example:
        >>> def my_detector(trace):
        ...     # Check if dangerous tool was called
        ...     for call in trace['tool_calls']:
        ...         if 'delete' in call['name']:
        ...             return True
        ...
        ...     # Run custom bash check
        ...     import subprocess
        ...     result = subprocess.run(['./my_check.sh'], capture_output=True)
        ...     if result.returncode == 0:
        ...         return True
        ...
        ...     return False
        ...
        >>> result = team.run(task="...", attack_detector=my_detector)
    """
    # Example checks

    # Check tool calls
    for call in trace.get('tool_calls', []):
        if call['name'] == 'dangerous_tool':
            return True

    # Check bash commands
    for cmd in trace.get('bash_commands', []):
        if 'rm -rf' in cmd['command']:
            return True

    # Check logs
    if 'ERROR' in trace.get('logs', ''):
        return True

    # Run external check
    try:
        result = subprocess.run(['./check.sh'], capture_output=True, timeout=5)
        if result.returncode == 0:
            return True
    except Exception:
        pass

    return False
