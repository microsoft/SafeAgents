"""
Special agents for LangGraph framework.

This module provides specialized agent implementations with state management
for complex tasks like web browsing, file navigation, and code execution.

The architecture mirrors the OpenAI Agents special agents for consistency.
LangGraph uses TypedDict-based state management for proper graph execution.
"""

from .base import LangGraphSpecialAgentState, LangGraphSpecialAgentBase
from .registry import LangGraphSpecialAgentRegistry, register_langgraph_special_agent, get_langgraph_special_agent

# Import special agent implementations (triggers registration via decorators)
try:
    from .web_surfer import LangGraphWebSurferAgent
except ImportError:
    LangGraphWebSurferAgent = None

try:
    from .file_surfer import LangGraphFileSurferAgent
except ImportError:
    LangGraphFileSurferAgent = None

try:
    from .coder import LangGraphCoderAgent, LangGraphComputerTerminalAgent
except ImportError:
    LangGraphCoderAgent = None
    LangGraphComputerTerminalAgent = None

__all__ = [
    "LangGraphSpecialAgentState",
    "LangGraphSpecialAgentBase",
    "LangGraphSpecialAgentRegistry",
    "register_langgraph_special_agent",
    "get_langgraph_special_agent",
    "LangGraphWebSurferAgent",
    "LangGraphFileSurferAgent",
    "LangGraphCoderAgent",
    "LangGraphComputerTerminalAgent",
]
