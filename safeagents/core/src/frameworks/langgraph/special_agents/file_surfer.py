# Copyright (c) Microsoft Corporation.
#
# This file contains modifications from the original version.
# The original file was sourced from:
# [Link to the original repo   https://github.com/microsoft/autogen]
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
File Surfer special agent for LangGraph framework.

This agent provides file navigation capabilities with a markdown-based file browser.
LangGraph requires TypedDict-based state management for proper graph execution.
"""

import os
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

# Import file surfer components from the same directory
try:
    from .markdown_file_browser import MarkdownFileBrowser
    FILE_SURFER_AVAILABLE = True
except ImportError:
    FILE_SURFER_AVAILABLE = False
    MarkdownFileBrowser = None
    logger.warning("MarkdownFileBrowser not available. FileSurfer agent will not work.")


class FileSurferState(LangGraphSpecialAgentState):
    """State for FileSurfer agent using TypedDict for LangGraph compatibility."""
    markdown_filebrowser: Any  # MarkdownFileBrowser instance
    viewport_size: int
    base_path: str
    cwd: Optional[str]


# Pydantic models for tool inputs
class OpenPathInput(BaseModel):
    """Input schema for open_path tool."""
    path: str = Field(description="The relative or absolute path of a local file or directory to open")


class FindOnPageInput(BaseModel):
    """Input schema for find_on_page tool."""
    search_string: str = Field(description="The string to search for. Supports wildcards like '*'")


# LangGraph tools using BaseTool pattern
class OpenPath(BaseTool):
    """Open a local file or directory at a path in the text-based file browser."""

    name: str = "open_path"
    description: str = "Open a local file or directory at a path in the text-based file browser and return current viewport content."
    args_schema: type[BaseModel] = OpenPathInput
    _state_wrapper: Dict[str, Any] = PrivateAttr()

    def __init__(self, state_wrapper: Dict[str, Any], **kwargs):
        super().__init__(**kwargs)
        self._state_wrapper = state_wrapper

    def _run(self, path: str) -> str:
        """Execute the tool."""
        browser = self._state_wrapper['markdown_filebrowser']
        browser.open_path(path)
        header, content = browser._get_browser_state()
        return header.strip() + "\n=======================\n" + content


class PageUp(BaseTool):
    """Scroll the viewport up one page-length in the current file."""

    name: str = "page_up"
    description: str = "Scroll the viewport UP one page-length in the current file and return the new viewport content."
    _state_wrapper: Dict[str, Any] = PrivateAttr()

    def __init__(self, state_wrapper: Dict[str, Any], **kwargs):
        super().__init__(**kwargs)
        self._state_wrapper = state_wrapper

    def _run(self) -> str:
        """Execute the tool."""
        browser = self._state_wrapper['markdown_filebrowser']
        browser.page_up()
        header, content = browser._get_browser_state()
        return header.strip() + "\n=======================\n" + content


class PageDown(BaseTool):
    """Scroll the viewport down one page-length in the current file."""

    name: str = "page_down"
    description: str = "Scroll the viewport DOWN one page-length in the current file and return the new viewport content."
    _state_wrapper: Dict[str, Any] = PrivateAttr()

    def __init__(self, state_wrapper: Dict[str, Any], **kwargs):
        super().__init__(**kwargs)
        self._state_wrapper = state_wrapper

    def _run(self) -> str:
        """Execute the tool."""
        browser = self._state_wrapper['markdown_filebrowser']
        browser.page_down()
        header, content = browser._get_browser_state()
        return header.strip() + "\n=======================\n" + content


class FindOnPageCtrlF(BaseTool):
    """Scroll the viewport to the first occurrence of the search string (Ctrl+F)."""

    name: str = "find_on_page_ctrl_f"
    description: str = "Scroll the viewport to the first occurrence of the search string. Equivalent to Ctrl+F. Supports wildcards like '*'."
    args_schema: type[BaseModel] = FindOnPageInput
    _state_wrapper: Dict[str, Any] = PrivateAttr()

    def __init__(self, state_wrapper: Dict[str, Any], **kwargs):
        super().__init__(**kwargs)
        self._state_wrapper = state_wrapper

    def _run(self, search_string: str) -> str:
        """Execute the tool."""
        browser = self._state_wrapper['markdown_filebrowser']
        browser.find_on_page(search_string)
        header, content = browser._get_browser_state()
        return header.strip() + "\n=======================\n" + content


class FindNext(BaseTool):
    """Scroll the viewport to the next occurrence of the current search string."""

    name: str = "find_next"
    description: str = "Scroll the viewport to the next occurrence of the current search string."
    _state_wrapper: Dict[str, Any] = PrivateAttr()

    def __init__(self, state_wrapper: Dict[str, Any], **kwargs):
        super().__init__(**kwargs)
        self._state_wrapper = state_wrapper

    def _run(self) -> str:
        """Execute the tool."""
        browser = self._state_wrapper['markdown_filebrowser']
        browser.find_next()
        header, content = browser._get_browser_state()
        return header.strip() + "\n=======================\n" + content


def generate_file_surfer_instructions(state: Dict[str, Any]) -> str:
    """
    Generate dynamic instructions for the File Surfer agent based on current state.

    Args:
        state: The current state dictionary

    Returns:
        A string containing the instructions for the File Surfer agent.
    """
    try:
        browser = state.get('markdown_filebrowser')
        if browser and hasattr(browser, 'page_title') and hasattr(browser, 'path'):
            current_path = browser.path
            page_title = browser.page_title
        else:
            current_path = "not yet opened"
            page_title = "not yet opened"
    except Exception:
        current_path = "not yet opened"
        page_title = "not yet opened"

    return (
        "You are a File Surfer agent. You can navigate local files and directories. "
        f"Your file viewer is currently at: '{page_title}' with path '{current_path}'.\n\n"
        "Core Rules:\n"
        "- Before every tool call, explain your reasoning to the user\n"
        "- Always use absolute paths whenever needed, don't use relative paths"
    )


@register_langgraph_special_agent("file_surfer")
class LangGraphFileSurferAgent(LangGraphSpecialAgentBase):
    """
    File Surfer special agent for LangGraph framework.
    Provides file navigation capabilities with TypedDict-based state management.
    """

    def create_state_class(self, **kwargs) -> type:
        """
        Create and return the state TypedDict class for FileSurfer.

        Args:
            **kwargs: State-specific configuration (viewport_size, base_path, cwd)

        Returns:
            A TypedDict class for managing FileSurfer state
        """
        if not FILE_SURFER_AVAILABLE:
            raise ImportError(
                "MarkdownFileBrowser dependencies are not available. "
                "Cannot create FileSurfer agent."
            )

        # Return the FileSurferState TypedDict
        return FileSurferState

    def create_tools(self, state_class: type) -> list:
        """
        Create and return the tools for FileSurfer agent.

        Args:
            state_class: The state TypedDict class (not used directly, tools access state at runtime)

        Returns:
            List of LangGraph tools (BaseTool instances)
        """
        # Note: In LangGraph, tools need access to state at runtime
        # We'll pass a state wrapper when creating tools
        # This will be properly initialized in create_agent
        return []  # Will be created in create_agent with proper state access

    def create_agent(self, tools: list) -> Any:
        """
        Create and return the LangGraph agent instance.

        Args:
            tools: List of tools (will be created here with proper state access)

        Returns:
            LangGraph agent instance (from create_react_agent)
        """
        # Tools will be created with access to state in the graph execution
        # For now, return None as the actual agent is created in initialize_state
        return None

    def initialize_state(self, **kwargs) -> Dict[str, Any]:
        """
        Initialize the state for FileSurfer agent.

        Args:
            **kwargs: Initialization parameters (viewport_size, base_path, cwd)

        Returns:
            Initial state dictionary conforming to FileSurferState
        """
        viewport_size = kwargs.get("viewport_size", 1024 * 8)
        base_path = kwargs.get("base_path", os.getcwd())
        cwd = kwargs.get("cwd")

        # Create the markdown file browser
        markdown_filebrowser = MarkdownFileBrowser(
            viewport_size=viewport_size,
            base_path=base_path,
            cwd=cwd
        )

        # Create state dictionary
        state = {
            "agent_name": self.name,
            "config": kwargs,
            "markdown_filebrowser": markdown_filebrowser,
            "viewport_size": viewport_size,
            "base_path": base_path,
            "cwd": cwd,
        }

        # Create tools with access to this state
        state_wrapper = {"markdown_filebrowser": markdown_filebrowser}
        tools = [
            OpenPath(state_wrapper),
            PageUp(state_wrapper),
            PageDown(state_wrapper),
            FindOnPageCtrlF(state_wrapper),
            FindNext(state_wrapper),
        ]

        # Create the actual agent with tools
        agent = create_react_agent(
            model=self.model,
            tools=tools,
            prompt=generate_file_surfer_instructions(state),
            name=self.name
        )

        # Store agent and tools in state for later use
        state["_agent"] = agent
        state["_tools"] = tools  # Store tools for tracking wrapper access

        logger.info(f"Initialized FileSurfer agent '{self.name}' with base_path='{base_path}'")
        return state

    async def cleanup_state(self, state: Dict[str, Any]):
        """
        Cleanup FileSurfer agent state resources.

        Args:
            state: The state dictionary to clean up
        """
        # No specific cleanup needed for file browser
        logger.debug(f"Cleaned up FileSurfer agent '{self.name}' state")
