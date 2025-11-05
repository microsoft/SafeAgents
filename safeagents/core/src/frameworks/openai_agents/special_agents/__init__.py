"""
Special agents for OpenAI Agents framework.

This module provides specialized agent implementations with context management
for complex tasks like web browsing, file navigation, and code execution.
"""

from .base import SpecialAgentContext, SpecialAgentBase
from .registry import SpecialAgentRegistry, register_special_agent, get_special_agent
from .web_surfer import WebSurferAgent
from .file_surfer import FileSurferAgent
from .coder import CoderAgent, ComputerTerminalAgent

__all__ = [
    "SpecialAgentContext",
    "SpecialAgentBase",
    "SpecialAgentRegistry",
    "register_special_agent",
    "get_special_agent",
    "WebSurferAgent",
    "FileSurferAgent",
    "CoderAgent",
    "ComputerTerminalAgent",
]
