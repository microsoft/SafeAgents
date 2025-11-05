def _add_tracking_to_web_surfer(web_surfer_agent, tool_call_tracker):
    """
    Monkey-patch web_surfer's _execute_tool to add tracking.

    Web surfer uses _execute_tool internally to execute browser actions.
    We wrap this method to track all tool calls.
    """
    import functools
    original_execute_tool = web_surfer_agent._execute_tool

    @functools.wraps(original_execute_tool)
    async def tracked_execute_tool(message, rects, tool_names, cancellation_token=None):
        # Extract tool info
        tool_name = None
        tool_args = {}
        if message and len(message) > 0:
            func_call = message[0]
            tool_name = func_call.name
            tool_args = func_call.arguments if hasattr(func_call, 'arguments') else {}

        try:
            result = await original_execute_tool(message, rects, tool_names, cancellation_token)

            # Track successful call
            if tool_call_tracker and tool_name:
                tool_call_tracker(
                    tool_name=tool_name,
                    args=tool_args,
                    result=str(result)[:500] if result else None,
                    agent_name=web_surfer_agent.name
                )
            return result
        except Exception as e:
            # Track failed call
            if tool_call_tracker and tool_name:
                tool_call_tracker(
                    tool_name=tool_name,
                    args=tool_args,
                    result=f"ERROR: {str(e)}",
                    agent_name=web_surfer_agent.name
                )
            raise

    web_surfer_agent._execute_tool = tracked_execute_tool


def _add_tracking_to_file_surfer(file_surfer_agent, tool_call_tracker):
    """
    Monkey-patch file_surfer's _generate_reply to add tracking.

    File surfer executes tools inline in _generate_reply.
    We need to track these tool executions.
    """
    import functools
    import json
    original_generate_reply = file_surfer_agent._generate_reply

    def track_tool(tool_name, args, result):
        if tool_call_tracker:
            tool_call_tracker(
                tool_name=tool_name,
                args=args,
                result=str(result)[:500] if result else None,
                agent_name=file_surfer_agent.name
            )

    @functools.wraps(original_generate_reply)
    async def tracked_generate_reply(cancellation_token):
        # We'll wrap the browser methods to track calls
        browser = file_surfer_agent._browser
        if browser:
            # Track tool calls by wrapping browser methods
            original_open_path = browser.open_path
            original_page_up = browser.page_up
            original_page_down = browser.page_down
            original_find_on_page = browser.find_on_page
            original_find_next = browser.find_next

            def tracked_open_path(path):
                result = original_open_path(path)
                track_tool("open_path", {"path": path}, "Opened path successfully")
                return result

            def tracked_page_up():
                result = original_page_up()
                track_tool("page_up", {}, "Scrolled up")
                return result

            def tracked_page_down():
                result = original_page_down()
                track_tool("page_down", {}, "Scrolled down")
                return result

            def tracked_find_on_page(search_string):
                result = original_find_on_page(search_string)
                track_tool("find_on_page_ctrl_f", {"search_string": search_string}, "Search executed")
                return result

            def tracked_find_next():
                result = original_find_next()
                track_tool("find_next", {}, "Found next match")
                return result

            browser.open_path = tracked_open_path
            browser.page_up = tracked_page_up
            browser.page_down = tracked_page_down
            browser.find_on_page = tracked_find_on_page
            browser.find_next = tracked_find_next

        return await original_generate_reply(cancellation_token)

    file_surfer_agent._generate_reply = tracked_generate_reply


def get_special_agent(agent, client, handoffs, tool_call_tracker=None):
    """
    Create an Autogen special agent.

    Args:
        agent: SafeAgents Agent configuration
        client: Autogen model client
        handoffs: List of handoff configurations
        tool_call_tracker: Optional callback for tracking tool calls (for SafeAgents integration)

    Returns:
        Autogen special agent instance
    """
    if agent.special_agent == "file_surfer":
        from .file_surfer import FileSurfer

        special_agent = FileSurfer(
            name=agent.name,
            model_client=client,
            **agent.special_agent_kwargs if agent.special_agent_kwargs else {},
        )

        # Add tool tracking if tracker provided
        if tool_call_tracker:
            _add_tracking_to_file_surfer(special_agent, tool_call_tracker)

        return special_agent

    elif agent.special_agent == "web_surfer":
        from .web_surfer import MultimodalWebSurfer

        special_agent = MultimodalWebSurfer(
            name=agent.name,
            model_client=client,
            **agent.special_agent_kwargs if agent.special_agent_kwargs else {},
        )

        # Add tool tracking if tracker provided
        if tool_call_tracker:
            _add_tracking_to_web_surfer(special_agent, tool_call_tracker)

        return special_agent

    elif agent.special_agent == "coder":
        from .coder import CoderAgent

        special_agent = CoderAgent(
            name=agent.name,
            model_client=client,
            handoffs=handoffs,
            tool_call_tracker=tool_call_tracker,
            **agent.special_agent_kwargs if agent.special_agent_kwargs else {},
        )

        return special_agent

    elif agent.special_agent == "computer_terminal":
        if handoffs:
            # NOTE: CodeExecutorAgent does not support handoffs currently.
            #       Remove this check when it gets added in future.
            raise ValueError("CodeExecutorAgent does not support handoffs currently.")

        from .code_executor import CodeExecutorAgent, TrackedLocalCommandLineCodeExecutor

        # Map working_directory to work_dir for LocalCommandLineCodeExecutor
        executor_kwargs = {}
        special_kwargs = agent.special_agent_kwargs or {}
        if "working_directory" in special_kwargs:
            executor_kwargs["work_dir"] = special_kwargs["working_directory"]

        # Create code executor with tracking
        code_executor = TrackedLocalCommandLineCodeExecutor(
            tool_call_tracker=tool_call_tracker,
            agent_name=agent.name,
            **executor_kwargs
        )

        special_agent = CodeExecutorAgent(
            name=agent.name,
            code_executor=code_executor,
        )

        return special_agent

    else:
        raise ValueError(f"Unknown special_agent type '{agent.special_agent}' for agent '{agent.name}'")