"""
Web Surfer special agent for OpenAI Agents framework.

This agent provides web browsing capabilities using Playwright.
It maintains a persistent browser session as context.
"""

from dataclasses import dataclass, field
from typing import Any, Optional
from agents import Agent, function_tool, ModelSettings, RunContextWrapper
from .base import SpecialAgentBase, SpecialAgentContext
from .registry import register_special_agent


# Import playwright-based web surfer components from the same directory
try:
    from .playwright_controller import PlaywrightController
    from .web_surfer_browser import MultiModalWebsurfer
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    PlaywrightController = None
    MultiModalWebsurfer = None


@dataclass
class WebSurferContext(SpecialAgentContext):
    """Context for WebSurfer agent containing browser session state."""

    multimodal_websurfer: Optional[Any] = None
    headless: bool = True
    start_page: str = "https://www.bing.com/"
    downloads_folder: Optional[str] = None
    debug_dir: Optional[str] = None
    to_save_screenshots: bool = False
    use_ocr: bool = False

    def __post_init__(self):
        """Initialize the web surfer browser."""
        super().__post_init__()
        if PLAYWRIGHT_AVAILABLE and MultiModalWebsurfer:
            self.multimodal_websurfer = MultiModalWebsurfer(
                name=self.agent_name,
                headless=self.headless,
                start_page=self.start_page,
                downloads_folder=self.downloads_folder,
                debug_dir=self.debug_dir,
                to_save_screenshots=self.to_save_screenshots,
                use_ocr=self.use_ocr,
            )

    async def cleanup(self):
        """Close the browser session."""
        if self.multimodal_websurfer:
            await self.multimodal_websurfer.close()


# Define web surfer tools that use context
@function_tool
async def tool_click(wrapper: RunContextWrapper[WebSurferContext], reasoning: str, target_id: int):
    """
    Clicks the mouse on the target with the given id.

    Args:
        reasoning: Explanation for why this action is being taken.
        target_id: The numeric id of the target to click.
    """
    if not wrapper.context.multimodal_websurfer.did_lazy_init:
        await wrapper.context.multimodal_websurfer._lazy_init()
    assert wrapper.context.multimodal_websurfer._page is not None
    rects = await wrapper.context.multimodal_websurfer._playwright_controller.get_interactive_rects(
        wrapper.context.multimodal_websurfer._page
    )
    return await wrapper.context.multimodal_websurfer._execute_tool(
        name="click",
        args={"target_id": target_id},
        rects=rects
    )


@function_tool
async def tool_visit_url(wrapper: RunContextWrapper[WebSurferContext], reasoning: str, url: str):
    """
    Visits the specified URL.

    Args:
        reasoning: Explanation for why this action is being taken.
        url: The URL to visit.
    """
    if not wrapper.context.multimodal_websurfer.did_lazy_init:
        await wrapper.context.multimodal_websurfer._lazy_init()
    assert wrapper.context.multimodal_websurfer._page is not None
    rects = await wrapper.context.multimodal_websurfer._playwright_controller.get_interactive_rects(
        wrapper.context.multimodal_websurfer._page
    )
    return await wrapper.context.multimodal_websurfer._execute_tool(
        name="visit_url",
        args={"url": url},
        rects=rects
    )


@function_tool
async def tool_web_search(wrapper: RunContextWrapper[WebSurferContext], reasoning: str, query: str):
    """
    Performs a web search using the given query.

    Args:
        reasoning: Explanation for why this action is being taken.
        query: The search query.
    """
    if not wrapper.context.multimodal_websurfer.did_lazy_init:
        await wrapper.context.multimodal_websurfer._lazy_init()
    assert wrapper.context.multimodal_websurfer._page is not None
    rects = await wrapper.context.multimodal_websurfer._playwright_controller.get_interactive_rects(
        wrapper.context.multimodal_websurfer._page
    )
    return await wrapper.context.multimodal_websurfer._execute_tool(
        name="web_search",
        args={"query": query},
        rects=rects
    )


@function_tool
async def tool_scroll_down(wrapper: RunContextWrapper[WebSurferContext], reasoning: str):
    """
    Scrolls the entire page down.

    Args:
        reasoning: Explanation for why this action is being taken.
    """
    if not wrapper.context.multimodal_websurfer.did_lazy_init:
        await wrapper.context.multimodal_websurfer._lazy_init()
    assert wrapper.context.multimodal_websurfer._page is not None
    rects = await wrapper.context.multimodal_websurfer._playwright_controller.get_interactive_rects(
        wrapper.context.multimodal_websurfer._page
    )
    return await wrapper.context.multimodal_websurfer._execute_tool(
        name="scroll_down",
        args={},
        rects=rects
    )


@function_tool
async def tool_scroll_up(wrapper: RunContextWrapper[WebSurferContext], reasoning: str):
    """
    Scrolls the entire page up.

    Args:
        reasoning: Explanation for why this action is being taken.
    """
    if not wrapper.context.multimodal_websurfer.did_lazy_init:
        await wrapper.context.multimodal_websurfer._lazy_init()
    assert wrapper.context.multimodal_websurfer._page is not None
    rects = await wrapper.context.multimodal_websurfer._playwright_controller.get_interactive_rects(
        wrapper.context.multimodal_websurfer._page
    )
    return await wrapper.context.multimodal_websurfer._execute_tool(
        name="scroll_up",
        args={},
        rects=rects
    )


@function_tool
async def tool_type(wrapper: RunContextWrapper[WebSurferContext], reasoning: str, input_field_id: int, text_value: str):
    """
    Types the given text value into the specified field.

    Args:
        reasoning: Explanation for why this action is being taken.
        input_field_id: The numeric id of the input field to receive the text.
        text_value: The text to type into the input field.
    """
    if not wrapper.context.multimodal_websurfer.did_lazy_init:
        await wrapper.context.multimodal_websurfer._lazy_init()
    assert wrapper.context.multimodal_websurfer._page is not None
    rects = await wrapper.context.multimodal_websurfer._playwright_controller.get_interactive_rects(
        wrapper.context.multimodal_websurfer._page
    )
    return await wrapper.context.multimodal_websurfer._execute_tool(
        name="input_text",
        args={"input_field_id": input_field_id, "text_value": text_value},
        rects=rects
    )


@register_special_agent("web_surfer")
class WebSurferAgent(SpecialAgentBase):
    """
    Web Surfer special agent that provides web browsing capabilities.
    """

    DEFAULT_INSTRUCTIONS = """
    A helpful assistant with access to a web browser.
    You can perform web searches, open pages, and interact with content (e.g., clicking links, scrolling the viewport, filling in form fields, etc.).
    """

    def create_context(self, **kwargs) -> WebSurferContext:
        """
        Create context for WebSurfer agent.

        Args:
            **kwargs: Configuration options (headless, start_page, downloads_folder, etc.)

        Returns:
            WebSurferContext: The created context
        """
        if not PLAYWRIGHT_AVAILABLE:
            raise ImportError(
                "Playwright and web_surfer dependencies are not available. "
                "Cannot create WebSurfer agent."
            )

        config = {
            "agent_name": self.name,
            "headless": kwargs.get("headless", True),
            "start_page": kwargs.get("start_page", "https://www.bing.com/"),
            "downloads_folder": kwargs.get("downloads_folder"),
            "debug_dir": kwargs.get("debug_dir"),
            "to_save_screenshots": kwargs.get("to_save_screenshots", False),
            "use_ocr": kwargs.get("use_ocr", False),
            "config": kwargs,
        }

        return WebSurferContext(**config)

    def create_agent(self, context: WebSurferContext) -> Agent:
        """
        Create OpenAI Agents framework agent for web surfing.

        Args:
            context: The WebSurferContext

        Returns:
            Agent: OpenAI Agents framework agent instance
        """
        tools = [
            tool_click,
            tool_visit_url,
            tool_web_search,
            tool_scroll_down,
            tool_scroll_up,
            tool_type,
        ]

        return Agent(
            name=self.name,
            model=self.model,
            instructions=self.DEFAULT_INSTRUCTIONS,
            tools=tools,
            model_settings=self.model_settings or ModelSettings(temperature=0.1)
        )
