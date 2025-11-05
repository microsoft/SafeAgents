import importlib
import inspect
import functools

class Tool:
    """
    Represents a tool that an agent can use to perform tasks.
    """

    def __init__(self, name, func):
        self.name = name
        self.func = func

    async def execute(self, query):
        """
        Execute the tool's function with the given query.
        """
        return await self.func(query)

    @staticmethod
    def load_from_names(tool_names, module):
        """
        Load tool functions by name from a given module.
        """
        tools_module = importlib.import_module(module)
        tools = []
        for tool_name in tool_names:
            if hasattr(tools_module, tool_name):
                func = getattr(tools_module, tool_name)
                tools.append(Tool(tool_name, func))
        return tools


def tool(name=None):
    """
    Decorator to wrap a function into a Tool instance.
    Usage:
        @tool()
        def my_tool(input):
            ...
        @tool(name="custom_name")
        def another_tool(input):
            ...
    """
    def decorator(func):
        tool_name = name or func.__name__

        # Wrap sync functions into async for compatibility
        if not inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return Tool(tool_name, async_wrapper)
        return Tool(tool_name, func)
    return decorator
