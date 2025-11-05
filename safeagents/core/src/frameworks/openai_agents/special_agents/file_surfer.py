"""
File Surfer special agent for OpenAI Agents framework.

This agent provides file navigation capabilities with a markdown-based file browser.
"""

from dataclasses import dataclass
from typing import Any, Optional
from agents import Agent, function_tool, ModelSettings, RunContextWrapper
from .base import SpecialAgentBase, SpecialAgentContext
from .registry import register_special_agent
import os


# Import file surfer components from the same directory
try:
    from .markdown_file_browser import MarkdownFileBrowser
    FILE_SURFER_AVAILABLE = True
except ImportError:
    FILE_SURFER_AVAILABLE = False
    MarkdownFileBrowser = None


@dataclass
class FileSurferContext(SpecialAgentContext):
    """Context for FileSurfer agent containing file browser state."""

    markdown_filebrowser: Optional[Any] = None
    viewport_size: int = 1024 * 8
    base_path: Optional[str] = None
    cwd: Optional[str] = None

    def __post_init__(self):
        """Initialize the file browser."""
        super().__post_init__()
        if FILE_SURFER_AVAILABLE and MarkdownFileBrowser:
            self.markdown_filebrowser = MarkdownFileBrowser(
                viewport_size=self.viewport_size,
                base_path=self.base_path,
                cwd=self.cwd
            )


# Define file surfer tools
@function_tool
def tool_open_path(wrapper: RunContextWrapper[FileSurferContext], path: str) -> str:
    """
    Open a local file or directory at a path in the text-based file browser and return current viewport content.

    Args:
        path: The relative or absolute path of a local file to visit.

    Returns:
        The content of the current viewport after opening the file or directory.
    """
    wrapper.context.markdown_filebrowser.open_path(path)
    header, content = wrapper.context.markdown_filebrowser._get_browser_state()
    final_response = header.strip() + "\n=======================\n" + content
    return final_response


@function_tool
def tool_page_up(wrapper: RunContextWrapper[FileSurferContext]) -> str:
    """
    Scroll the viewport UP one page-length in the current file and return the new viewport content.

    Returns:
        The content of the current viewport after scrolling up.
    """
    wrapper.context.markdown_filebrowser.page_up()
    header, content = wrapper.context.markdown_filebrowser._get_browser_state()
    final_response = header.strip() + "\n=======================\n" + content
    return final_response


@function_tool
def tool_page_down(wrapper: RunContextWrapper[FileSurferContext]) -> str:
    """
    Scroll the viewport DOWN one page-length in the current file and return the new viewport content.

    Returns:
        The content of the current viewport after scrolling down.
    """
    wrapper.context.markdown_filebrowser.page_down()
    header, content = wrapper.context.markdown_filebrowser._get_browser_state()
    final_response = header.strip() + "\n=======================\n" + content
    return final_response


@function_tool
def tool_find_on_page(wrapper: RunContextWrapper[FileSurferContext], search_string: str) -> str:
    """
    Scroll the viewport to the first occurrence of the search string. This is equivalent to Ctrl+F.

    Args:
        search_string: The string to search for on the page. This search string supports wildcards like '*'.

    Returns:
        The content of the current viewport after searching for the string.
    """
    wrapper.context.markdown_filebrowser.find_on_page(search_string)
    header, content = wrapper.context.markdown_filebrowser._get_browser_state()
    final_response = header.strip() + "\n=======================\n" + content
    return final_response


@function_tool
def tool_find_next(wrapper: RunContextWrapper[FileSurferContext]) -> str:
    """
    Scroll the viewport to next occurrence of the search string.

    Returns:
        The content of the current viewport after finding the next occurrence.
    """
    wrapper.context.markdown_filebrowser.find_next()
    header, content = wrapper.context.markdown_filebrowser._get_browser_state()
    final_response = header.strip() + "\n=======================\n" + content
    return final_response


def generate_file_surfer_instructions(wrapper, agent) -> str:
    """
    Generate dynamic instructions for the File Surfer agent based on current state.

    Returns:
        A string containing the instructions for the File Surfer agent.
    """
    # Handle case where context might not be initialized yet
    try:
        if hasattr(wrapper, 'context') and wrapper.context and hasattr(wrapper.context, 'markdown_filebrowser'):
            current_path = wrapper.context.markdown_filebrowser.path if wrapper.context.markdown_filebrowser else "unknown"
            page_title = wrapper.context.markdown_filebrowser.page_title if wrapper.context.markdown_filebrowser else "unknown"
        else:
            current_path = "not yet opened"
            page_title = "not yet opened"
    except:
        current_path = "not yet opened"
        page_title = "not yet opened"

    return (
        "You are a File Surfer agent. You can navigate local files and directories. "
        f"Your file viewer is currently at: '{page_title}' with path '{current_path}'.\n\n"
        "Core Rules:\n"
        "- Before every tool call, explain your reasoning to the user\n"
        "- Always use absolute paths whenever needed, don't use relative paths"
    )


@register_special_agent("file_surfer")
class FileSurferAgent(SpecialAgentBase):
    """
    File Surfer special agent that provides file navigation capabilities.
    """

    def create_context(self, **kwargs) -> FileSurferContext:
        """
        Create context for FileSurfer agent.

        Args:
            **kwargs: Configuration options (viewport_size, base_path, cwd)

        Returns:
            FileSurferContext: The created context
        """
        if not FILE_SURFER_AVAILABLE:
            raise ImportError(
                "MarkdownFileBrowser dependencies are not available. "
                "Cannot create FileSurfer agent."
            )

        config = {
            "agent_name": self.name,
            "viewport_size": kwargs.get("viewport_size", 1024 * 8),
            "base_path": kwargs.get("base_path", os.getcwd()),
            "cwd": kwargs.get("cwd"),
            "config": kwargs,
        }

        return FileSurferContext(**config)

    def create_agent(self, context: FileSurferContext) -> Agent:
        """
        Create OpenAI Agents framework agent for file surfing.

        Args:
            context: The FileSurferContext

        Returns:
            Agent: OpenAI Agents framework agent instance
        """
        tools = [
            tool_open_path,
            tool_page_up,
            tool_page_down,
            tool_find_on_page,
            tool_find_next,
        ]

        return Agent(
            name=self.name,
            model=self.model,
            instructions=generate_file_surfer_instructions,
            tools=tools,
            model_settings=self.model_settings or ModelSettings(temperature=0.1)
        )
