# Copyright (c) Microsoft Corporation.

# This file contains modifications from the original version.
# The original file was sourced from:
# [Link to the original repo : https://github.com/microsoft/autogen]

# -----------------------------------------------------------------------------
# The original file is licensed under the MIT License:

# MIT License

#     Copyright (c) Microsoft Corporation.

#     Permission is hereby granted, free of charge, to any person obtaining a copy
#     of this software and associated documentation files (the "Software"), to deal
#     in the Software without restriction, including without limitation the rights
#     to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#     copies of the Software, and to permit persons to whom the Software is
#     furnished to do so, subject to the following conditions:

#     The above copyright notice and this permission notice shall be included in all
#     copies or substantial portions of the Software.

#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#     IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#     FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#     AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#     LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#     OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#     SOFTWARE
# -----------------------------------------------------------------------------
"""
MultiModal Web Surfer implementation for browser automation.

This module provides the core web browsing functionality using Playwright.
"""

from typing import Dict, Any, List, Optional
from playwright.async_api import BrowserContext, Download, Page, Playwright, async_playwright
from .playwright_controller import PlaywrightController, add_set_of_mark, InteractiveRegion
import os
import time
import base64
import hashlib
import io
import re
from urllib.parse import quote_plus
import aiofiles


class MultiModalWebsurfer:
    """
    A multimodal web surfer that can interact with web pages using Playwright.
    """

    DEFAULT_DESCRIPTION = """
    A helpful assistant with access to a web browser.
    Ask them to perform web searches, open pages, and interact with content (e.g., clicking links, scrolling the viewport, filling in form fields, etc.).
    It can also summarize the entire page, or answer questions based on the content of the page.
    It can also be asked to sleep and wait for pages to load, in cases where the page seems not yet fully loaded.
    """
    DEFAULT_START_PAGE = "https://www.bing.com/"

    # Viewport dimensions
    VIEWPORT_HEIGHT = 900
    VIEWPORT_WIDTH = 1440

    # Size of the image we send to the MLM
    MLM_HEIGHT = 765
    MLM_WIDTH = 1224

    SCREENSHOT_TOKENS = 1105

    def __init__(
        self,
        name: str,
        downloads_folder: str | None = None,
        description: str = DEFAULT_DESCRIPTION,
        debug_dir: str | None = None,
        headless: bool = True,
        start_page: str | None = DEFAULT_START_PAGE,
        animate_actions: bool = False,
        to_save_screenshots: bool = False,
        use_ocr: bool = False,
        browser_channel: str | None = None,
        browser_data_dir: str | None = None,
        to_resize_viewport: bool = True,
        playwright: Playwright | None = None,
        context: BrowserContext | None = None,
    ):
        self.headless = headless
        self.browser_channel = browser_channel
        self.browser_data_dir = browser_data_dir
        self.start_page = start_page or self.DEFAULT_START_PAGE
        self.downloads_folder = downloads_folder
        self.debug_dir = debug_dir
        self.to_save_screenshots = to_save_screenshots
        self.use_ocr = use_ocr
        self.to_resize_viewport = to_resize_viewport
        self.animate_actions = animate_actions

        # Initialize state
        self._playwright: Playwright | None = playwright
        self._context: BrowserContext | None = context
        self._page: Page | None = None
        self._last_download: Download | None = None
        self._prior_metadata_hash: str | None = None
        self.did_lazy_init: bool = False

        # Define the download handler
        def _download_handler(download: Download) -> None:
            self._last_download = download

        self._download_handler = _download_handler

        # Define the Playwright controller that handles the browser interactions
        self._playwright_controller = PlaywrightController(
            animate_actions=self.animate_actions,
            downloads_folder=self.downloads_folder,
            viewport_width=self.VIEWPORT_WIDTH,
            viewport_height=self.VIEWPORT_HEIGHT,
            _download_handler=self._download_handler,
            to_resize_viewport=self.to_resize_viewport,
        )

    async def _lazy_init(self) -> None:
        """Initialize the browser on first call."""
        self._last_download = None
        self._prior_metadata_hash = None

        # Create the playwright instance
        launch_args: Dict[str, Any] = {"headless": self.headless}
        if self.browser_channel is not None:
            launch_args["channel"] = self.browser_channel
        if self._playwright is None:
            self._playwright = await async_playwright().start()

        # Create the context
        if self._context is None:
            if self.browser_data_dir is None:
                browser = await self._playwright.chromium.launch(**launch_args)
                self._context = await browser.new_context(
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 Edg/122.0.0.0"
                )
            else:
                self._context = await self._playwright.chromium.launch_persistent_context(
                    self.browser_data_dir, **launch_args
                )

        # Create the page
        self._context.set_default_timeout(60000)  # One minute
        self._page = await self._context.new_page()
        assert self._page is not None
        self._page.on("download", self._download_handler)
        if self.to_resize_viewport:
            await self._page.set_viewport_size({"width": self.VIEWPORT_WIDTH, "height": self.VIEWPORT_HEIGHT})
        await self._page.add_init_script(
            path=os.path.join(os.path.abspath(os.path.dirname(__file__)), "page_script.js")
        )
        await self._page.goto(self.start_page)
        await self._page.wait_for_load_state()

        # Prepare the debug directory
        await self._set_debug_dir(self.debug_dir)
        self.did_lazy_init = True

    def _format_target_list(self, ids: List[str], rects: Dict[str, InteractiveRegion]) -> List[str]:
        """Format the list of targets in the webpage as a string."""
        targets: List[str] = []
        for r in list(set(ids)):
            if r in rects:
                aria_role = rects[r].get("role", "").strip()
                if len(aria_role) == 0:
                    aria_role = rects[r].get("tag_name", "").strip()

                aria_name = re.sub(r"[\n\r]+", " ", rects[r].get("aria_name", "")).strip()

                actions = ['"click", "hover"']
                if rects[r]["role"] in ["textbox", "searchbox", "search"]:
                    actions = ['"input_text"']
                actions_str = "[" + ",".join(actions) + "]"

                targets.append(f'{{"id": {r}, "name": "{aria_name}", "role": "{aria_role}", "tools": {actions_str} }}')

        return targets

    def _target_name(self, target: str, rects: Dict[str, InteractiveRegion]) -> str | None:
        try:
            return rects[target]["aria_name"].strip()
        except KeyError:
            return None

    async def close(self) -> None:
        """Close the browser and page."""
        if self._page is not None:
            await self._page.close()
            self._page = None
        if self._context is not None:
            await self._context.close()
            self._context = None
        if self._playwright is not None:
            await self._playwright.stop()
            self._playwright = None

    async def _set_debug_dir(self, debug_dir: str | None) -> None:
        assert self._page is not None
        if self.debug_dir is None:
            return

        if not os.path.isdir(self.debug_dir):
            os.mkdir(self.debug_dir)

        if self.to_save_screenshots:
            current_timestamp = "_" + int(time.time()).__str__()
            screenshot_png_name = "screenshot" + current_timestamp + ".png"
            await self._page.screenshot(path=os.path.join(self.debug_dir, screenshot_png_name))

    async def _get_state_description(self) -> str:
        assert self._playwright_controller is not None
        assert self._page is not None

        # Describe the viewport
        viewport = await self._playwright_controller.get_visual_viewport(self._page)
        percent_visible = int(viewport["height"] * 100 / viewport["scrollHeight"])
        percent_scrolled = int(viewport["pageTop"] * 100 / viewport["scrollHeight"])
        if percent_scrolled < 1:
            position_text = "at the top of the page"
        elif percent_scrolled + percent_visible >= 99:
            position_text = "at the bottom of the page"
        else:
            position_text = str(percent_scrolled) + "% down from the top of the page"

        visible_text = await self._playwright_controller.get_visible_text(self._page)

        page_title = await self._page.title()
        message_content = f"web browser is open to the page [{page_title}]({self._page.url}).\nThe viewport shows {percent_visible}% of the webpage, and is positioned {position_text}\n"
        message_content += f"The following text is visible in the viewport:\n\n{visible_text}"
        return message_content

    async def _execute_tool(
        self,
        name: str,
        args: Dict[str, Any],
        rects: Dict[str, InteractiveRegion],
        tool_names: str = None,
    ):
        """Execute a browser action tool."""
        if name == "visit_url":
            url = args.get("url")
            action_description = f"I typed '{url}' into the browser address bar."
            if url.startswith(("https://", "http://", "file://", "about:")):
                reset_prior_metadata, reset_last_download = await self._playwright_controller.visit_page(
                    self._page, url
                )
            elif " " in url:
                reset_prior_metadata, reset_last_download = await self._playwright_controller.visit_page(
                    self._page, f"https://www.bing.com/search?q={quote_plus(url)}&FORM=QBLH"
                )
            else:
                reset_prior_metadata, reset_last_download = await self._playwright_controller.visit_page(
                    self._page, "https://" + url
                )
            if reset_last_download and self._last_download is not None:
                self._last_download = None
            if reset_prior_metadata and self._prior_metadata_hash is not None:
                self._prior_metadata_hash = None

        elif name == "history_back":
            action_description = "I clicked the browser back button."
            await self._playwright_controller.back(self._page)

        elif name == "web_search":
            query = args.get("query")
            action_description = f"I typed '{query}' into the browser search bar."
            reset_prior_metadata, reset_last_download = await self._playwright_controller.visit_page(
                self._page, f"https://www.bing.com/search?q={quote_plus(query)}&FORM=QBLH"
            )
            if reset_last_download and self._last_download is not None:
                self._last_download = None
            if reset_prior_metadata and self._prior_metadata_hash is not None:
                self._prior_metadata_hash = None

        elif name == "scroll_up":
            action_description = "I scrolled up one page in the browser."
            await self._playwright_controller.page_up(self._page)

        elif name == "scroll_down":
            action_description = "I scrolled down one page in the browser."
            await self._playwright_controller.page_down(self._page)

        elif name == "click":
            target_id = str(args.get("target_id"))
            target_name = self._target_name(target_id, rects)
            if target_name:
                action_description = f"I clicked '{target_name}'."
            else:
                action_description = "I clicked the control."
            new_page_tentative = await self._playwright_controller.click_id(self._page, target_id)
            if new_page_tentative is not None:
                self._page = new_page_tentative
                self._prior_metadata_hash = None

        elif name == "input_text":
            input_field_id = str(args.get("input_field_id"))
            text_value = str(args.get("text_value"))
            input_field_name = self._target_name(input_field_id, rects)
            if input_field_name:
                action_description = f"I typed '{text_value}' into '{input_field_name}'."
            else:
                action_description = f"I input '{text_value}'."
            await self._playwright_controller.fill_id(self._page, input_field_id, text_value)

        elif name == "hover":
            target_id = str(args.get("target_id"))
            target_name = self._target_name(target_id, rects)
            if target_name:
                action_description = f"I hovered over '{target_name}'."
            else:
                action_description = "I hovered over the control."
            await self._playwright_controller.hover_id(self._page, target_id)

        elif name == "sleep":
            action_description = "I am waiting a short period of time before taking further action."
            await self._playwright_controller.sleep(self._page, 3)

        else:
            raise ValueError(f"Unknown tool '{name}'. Please choose from:\n\n{tool_names}")

        await self._page.wait_for_load_state()
        await self._playwright_controller.sleep(self._page, 3)

        # Handle downloads
        if self._last_download is not None and self.downloads_folder is not None:
            fname = os.path.join(self.downloads_folder, self._last_download.suggested_filename)
            await self._last_download.save_as(fname)
            page_body = f"<html><head><title>Download Successful</title></head><body style=\"margin: 20px;\"><h1>Successfully downloaded '{self._last_download.suggested_filename}' to local path:<br><br>{fname}</h1></body></html>"
            await self._page.goto(
                "data:text/html;base64," + base64.b64encode(page_body.encode("utf-8")).decode("utf-8")
            )
            await self._page.wait_for_load_state()

        # Handle metadata
        page_metadata = await self._playwright_controller.get_page_metadata(self._page)
        import json
        page_metadata_str = json.dumps(page_metadata, indent=4)
        metadata_hash = hashlib.md5(page_metadata_str.encode("utf-8")).hexdigest()
        if metadata_hash != self._prior_metadata_hash:
            page_metadata_text = "\n\nThe following metadata was extracted from the webpage:\n\n" + page_metadata_str.strip() + "\n"
        else:
            page_metadata_text = ""
        self._prior_metadata_hash = metadata_hash

        new_screenshot = await self._page.screenshot()
        if self.to_save_screenshots:
            current_timestamp = "_" + int(time.time()).__str__()
            screenshot_png_name = "screenshot" + current_timestamp + ".png"
            async with aiofiles.open(os.path.join(self.debug_dir, screenshot_png_name), "wb") as file:
                await file.write(new_screenshot)

        som_screenshot, visible_rects, rects_above, rects_below = add_set_of_mark(new_screenshot, rects)
        focused = await self._playwright_controller.get_focused_rect_id(self._page)
        focused_hint = ""
        if focused:
            name = self._target_name(focused, rects)
            if name:
                name = f"(and name '{name}') "
            else:
                name = ""

            role = "control"
            try:
                role = rects[focused]["role"]
            except KeyError:
                pass

            focused_hint = f"\nThe {role} with ID {focused} {name}currently has the input focus.\n\n"

        # Everything visible
        visible_targets = "\n".join(self._format_target_list(visible_rects, rects)) + "\n\n"

        # Everything else
        other_targets: List[str] = []
        other_targets.extend(self._format_target_list(rects_above, rects))
        other_targets.extend(self._format_target_list(rects_below, rects))

        if len(other_targets) > 0:
            if len(other_targets) > 30:
                other_targets = other_targets[0:30]
                other_targets.append("...")
            other_targets_str = (
                "Additional valid interaction targets include (but are not limited to):\n"
                + "\n".join(other_targets)
                + "\n\n"
            )
        else:
            other_targets_str = ""

        # Return the complete observation
        state_description = "The " + await self._get_state_description()

        message_content = (
            f"{action_description}\n\n" + state_description + page_metadata_text
            + f"You have also identified the following interactive components:{visible_targets}{other_targets_str}{focused_hint}"
        )

        return re.sub(r"(\n\s*){3,}", "\n\n", message_content)
