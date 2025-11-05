"""
Web Surfer special agent for LangGraph framework.

This agent provides web browsing capabilities using Playwright.
LangGraph requires TypedDict-based state management for proper graph execution.
"""

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

# Import playwright-based web surfer components from the same directory
try:
    from .playwright_controller import PlaywrightController
    from .web_surfer_browser import MultiModalWebsurfer
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    PlaywrightController = None
    MultiModalWebsurfer = None
    logger.warning("Playwright dependencies not available. WebSurfer agent will not work.")


class WebSurferState(LangGraphSpecialAgentState):
    """State for WebSurfer agent using TypedDict for LangGraph compatibility."""
    multimodal_websurfer: Any  # MultiModalWebsurfer instance
    headless: bool
    start_page: str
    downloads_folder: Optional[str]
    debug_dir: Optional[str]
    to_save_screenshots: bool
    use_ocr: bool


# Pydantic models for tool inputs
class ClickInput(BaseModel):
    """Input schema for click tool."""
    reasoning: str = Field(description="Explanation for why this action is being taken")
    target_id: int = Field(description="The numeric id of the target to click")


class VisitUrlInput(BaseModel):
    """Input schema for visit_url tool."""
    reasoning: str = Field(description="Explanation for why this action is being taken")
    url: str = Field(description="The URL to visit")


class WebSearchInput(BaseModel):
    """Input schema for web_search tool."""
    reasoning: str = Field(description="Explanation for why this action is being taken")
    query: str = Field(description="The search query")


class InputTextInput(BaseModel):
    """Input schema for input_text tool."""
    reasoning: str = Field(description="Explanation for why this action is being taken")
    target_id: int = Field(description="The numeric id of the textbox target")
    text: str = Field(description="The text to input")


class ReasoningInput(BaseModel):
    """Input schema for tools that only require reasoning."""
    reasoning: str = Field(description="Explanation for why this action is being taken")


# LangGraph tools using BaseTool pattern
class Click(BaseTool):
    """Click on a target element on the page."""

    name: str = "click"
    description: str = "Clicks the mouse on the target with the given id."
    args_schema: type[BaseModel] = ClickInput
    _state_wrapper: Dict[str, Any] = PrivateAttr()

    def __init__(self, state_wrapper: Dict[str, Any], **kwargs):
        super().__init__(**kwargs)
        self._state_wrapper = state_wrapper

    async def _arun(self, reasoning: str, target_id: int) -> str:
        """Execute the tool asynchronously."""
        browser = self._state_wrapper['multimodal_websurfer']
        if not browser.did_lazy_init:
            await browser._lazy_init()
        assert browser._page is not None
        rects = await browser._playwright_controller.get_interactive_rects(browser._page)
        return await browser._execute_tool(name="click", args={"target_id": target_id}, rects=rects)


class VisitUrl(BaseTool):
    """Visit a specified URL."""

    name: str = "visit_url"
    description: str = "Visits the specified URL."
    args_schema: type[BaseModel] = VisitUrlInput
    _state_wrapper: Dict[str, Any] = PrivateAttr()

    def __init__(self, state_wrapper: Dict[str, Any], **kwargs):
        super().__init__(**kwargs)
        self._state_wrapper = state_wrapper

    async def _arun(self, reasoning: str, url: str) -> str:
        """Execute the tool asynchronously."""
        browser = self._state_wrapper['multimodal_websurfer']
        if not browser.did_lazy_init:
            await browser._lazy_init()
        assert browser._page is not None
        rects = await browser._playwright_controller.get_interactive_rects(browser._page)
        return await browser._execute_tool(name="visit_url", args={"url": url}, rects=rects)


class WebSearch(BaseTool):
    """Perform a web search."""

    name: str = "web_search"
    description: str = "Performs a web search using the given query."
    args_schema: type[BaseModel] = WebSearchInput
    _state_wrapper: Dict[str, Any] = PrivateAttr()

    def __init__(self, state_wrapper: Dict[str, Any], **kwargs):
        super().__init__(**kwargs)
        self._state_wrapper = state_wrapper

    async def _arun(self, reasoning: str, query: str) -> str:
        """Execute the tool asynchronously."""
        browser = self._state_wrapper['multimodal_websurfer']
        if not browser.did_lazy_init:
            await browser._lazy_init()
        assert browser._page is not None
        rects = await browser._playwright_controller.get_interactive_rects(browser._page)
        return await browser._execute_tool(name="web_search", args={"query": query}, rects=rects)


class ScrollDown(BaseTool):
    """Scroll the page down."""

    name: str = "scroll_down"
    description: str = "Scrolls the entire page down."
    args_schema: type[BaseModel] = ReasoningInput
    _state_wrapper: Dict[str, Any] = PrivateAttr()

    def __init__(self, state_wrapper: Dict[str, Any], **kwargs):
        super().__init__(**kwargs)
        self._state_wrapper = state_wrapper

    async def _arun(self, reasoning: str) -> str:
        """Execute the tool asynchronously."""
        browser = self._state_wrapper['multimodal_websurfer']
        if not browser.did_lazy_init:
            await browser._lazy_init()
        assert browser._page is not None
        rects = await browser._playwright_controller.get_interactive_rects(browser._page)
        return await browser._execute_tool(name="scroll_down", args={}, rects=rects)


class ScrollUp(BaseTool):
    """Scroll the page up."""

    name: str = "scroll_up"
    description: str = "Scrolls the entire page up."
    args_schema: type[BaseModel] = ReasoningInput
    _state_wrapper: Dict[str, Any] = PrivateAttr()

    def __init__(self, state_wrapper: Dict[str, Any], **kwargs):
        super().__init__(**kwargs)
        self._state_wrapper = state_wrapper

    async def _arun(self, reasoning: str) -> str:
        """Execute the tool asynchronously."""
        browser = self._state_wrapper['multimodal_websurfer']
        if not browser.did_lazy_init:
            await browser._lazy_init()
        assert browser._page is not None
        rects = await browser._playwright_controller.get_interactive_rects(browser._page)
        return await browser._execute_tool(name="scroll_up", args={}, rects=rects)


class InputText(BaseTool):
    """Input text into a textbox on the page."""

    name: str = "input_text"
    description: str = "Inputs text into a textbox with the given id."
    args_schema: type[BaseModel] = InputTextInput
    _state_wrapper: Dict[str, Any] = PrivateAttr()

    def __init__(self, state_wrapper: Dict[str, Any], **kwargs):
        super().__init__(**kwargs)
        self._state_wrapper = state_wrapper

    async def _arun(self, reasoning: str, target_id: int, text: str) -> str:
        """Execute the tool asynchronously."""
        browser = self._state_wrapper['multimodal_websurfer']
        if not browser.did_lazy_init:
            await browser._lazy_init()
        assert browser._page is not None
        rects = await browser._playwright_controller.get_interactive_rects(browser._page)
        return await browser._execute_tool(
            name="input_text",
            args={"target_id": target_id, "text": text},
            rects=rects
        )


class PageUp(BaseTool):
    """Scroll up one page-length."""

    name: str = "page_up"
    description: str = "Scrolls the viewport up one page-length in the current page."
    args_schema: type[BaseModel] = ReasoningInput
    _state_wrapper: Dict[str, Any] = PrivateAttr()

    def __init__(self, state_wrapper: Dict[str, Any], **kwargs):
        super().__init__(**kwargs)
        self._state_wrapper = state_wrapper

    async def _arun(self, reasoning: str) -> str:
        """Execute the tool asynchronously."""
        browser = self._state_wrapper['multimodal_websurfer']
        if not browser.did_lazy_init:
            await browser._lazy_init()
        assert browser._page is not None
        rects = await browser._playwright_controller.get_interactive_rects(browser._page)
        return await browser._execute_tool(name="page_up", args={}, rects=rects)


class PageDown(BaseTool):
    """Scroll down one page-length."""

    name: str = "page_down"
    description: str = "Scrolls the viewport down one page-length in the current page."
    args_schema: type[BaseModel] = ReasoningInput
    _state_wrapper: Dict[str, Any] = PrivateAttr()

    def __init__(self, state_wrapper: Dict[str, Any], **kwargs):
        super().__init__(**kwargs)
        self._state_wrapper = state_wrapper

    async def _arun(self, reasoning: str) -> str:
        """Execute the tool asynchronously."""
        browser = self._state_wrapper['multimodal_websurfer']
        if not browser.did_lazy_init:
            await browser._lazy_init()
        assert browser._page is not None
        rects = await browser._playwright_controller.get_interactive_rects(browser._page)
        return await browser._execute_tool(name="page_down", args={}, rects=rects)


def generate_web_surfer_instructions(state: Dict[str, Any]) -> str:
    """
    Generate dynamic instructions for the Web Surfer agent based on current state.

    Args:
        state: The current state dictionary

    Returns:
        A string containing the instructions for the Web Surfer agent.
    """
    start_page = state.get('start_page', 'https://www.bing.com/')

    return (
        "You are a Web Surfer agent. You can browse the web, search, and interact with web pages. "
        f"Your browser will start at: {start_page}\n\n"
        "Core Rules:\n"
        "- Before every tool call, provide reasoning for your action\n"
        "- Use web_search for searching information\n"
        "- Use visit_url to navigate to specific URLs\n"
        "- Use click to interact with page elements\n"
        "- Use input_text to fill in form fields\n"
        "- Use scroll commands to navigate long pages"
    )


@register_langgraph_special_agent("web_surfer")
class LangGraphWebSurferAgent(LangGraphSpecialAgentBase):
    """
    Web Surfer special agent for LangGraph framework.
    Provides web browsing capabilities with TypedDict-based state management.
    """

    def create_state_class(self, **kwargs) -> type:
        """
        Create and return the state TypedDict class for WebSurfer.

        Args:
            **kwargs: State-specific configuration (headless, start_page, etc.)

        Returns:
            A TypedDict class for managing WebSurfer state
        """
        if not PLAYWRIGHT_AVAILABLE:
            raise ImportError(
                "Playwright dependencies are not available. "
                "Install with: pip install playwright && playwright install chromium"
            )

        # Return the WebSurferState TypedDict
        return WebSurferState

    def create_tools(self, state_class: type) -> list:
        """
        Create and return the tools for WebSurfer agent.

        Args:
            state_class: The state TypedDict class (not used directly, tools access state at runtime)

        Returns:
            List of LangGraph tools (BaseTool instances)
        """
        # Note: Tools will be created in initialize_state with proper state access
        return []

    def create_agent(self, tools: list) -> Any:
        """
        Create and return the LangGraph agent instance.

        Args:
            tools: List of tools (will be created in initialize_state)

        Returns:
            LangGraph agent instance (from create_react_agent)
        """
        # Agent will be created in initialize_state with proper tools
        return None

    def initialize_state(self, **kwargs) -> Dict[str, Any]:
        """
        Initialize the state for WebSurfer agent.

        Args:
            **kwargs: Initialization parameters (headless, start_page, etc.)

        Returns:
            Initial state dictionary conforming to WebSurferState
        """
        headless = kwargs.get("headless", True)
        start_page = kwargs.get("start_page", "https://www.bing.com/")
        downloads_folder = kwargs.get("downloads_folder")
        debug_dir = kwargs.get("debug_dir")
        to_save_screenshots = kwargs.get("to_save_screenshots", False)
        use_ocr = kwargs.get("use_ocr", False)

        # Create the multimodal web surfer
        multimodal_websurfer = MultiModalWebsurfer(
            name=self.name,
            headless=headless,
            start_page=start_page,
            downloads_folder=downloads_folder,
            debug_dir=debug_dir,
            to_save_screenshots=to_save_screenshots,
            use_ocr=use_ocr,
        )

        # Create state dictionary
        state = {
            "agent_name": self.name,
            "config": kwargs,
            "multimodal_websurfer": multimodal_websurfer,
            "headless": headless,
            "start_page": start_page,
            "downloads_folder": downloads_folder,
            "debug_dir": debug_dir,
            "to_save_screenshots": to_save_screenshots,
            "use_ocr": use_ocr,
        }

        # Create tools with access to this state
        state_wrapper = {"multimodal_websurfer": multimodal_websurfer}
        tools = [
            VisitUrl(state_wrapper),
            WebSearch(state_wrapper),
            Click(state_wrapper),
            InputText(state_wrapper),
            ScrollDown(state_wrapper),
            ScrollUp(state_wrapper),
            PageDown(state_wrapper),
            PageUp(state_wrapper),
        ]

        # Create the actual agent with tools
        agent = create_react_agent(
            model=self.model,
            tools=tools,
            prompt=generate_web_surfer_instructions(state),
            name=self.name
        )

        # Store agent and tools in state for later use
        state["_agent"] = agent
        state["_tools"] = tools  # Store tools for tracking wrapper access

        logger.info(f"Initialized WebSurfer agent '{self.name}' with start_page='{start_page}' headless={headless}")
        return state

    async def cleanup_state(self, state: Dict[str, Any]):
        """
        Cleanup WebSurfer agent state resources.

        Args:
            state: The state dictionary to clean up
        """
        # Close the browser session
        multimodal_websurfer = state.get('multimodal_websurfer')
        if multimodal_websurfer:
            await multimodal_websurfer.close()
            logger.debug(f"Closed WebSurfer agent '{self.name}' browser session")
