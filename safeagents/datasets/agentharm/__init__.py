"""
AgentHarm Dataset Handler

This module contains the dataset-specific logic for the AgentHarm benchmark.
Users don't need to modify this - it's auto-registered and ready to use.
"""
from ...core.src import AgentConfig, Agent, Tool
from ...core.logger import logger


class AgentHarmHandler:
    """Handler for ai-safety-institute/AgentHarm dataset."""

    @staticmethod
    def load(config=None, split="test_public"):
        """
        Load AgentHarm dataset from HuggingFace.

        Args:
            config: Dataset configuration (e.g., "harmful", "harmless_benign", "chat")
                   For now, defaults to "harmful" if not specified
            split: Dataset split - note that AgentHarm uses "test_public" and "validation"
                   Defaults to "validation" since "test" doesn't exist

        Returns:
            List[Dict]: List of tasks with standardized fields
        """
        from datasets import load_dataset

        # Default to "harmful" config and "validation" split for AgentHarm
        # (AgentHarm doesn't have "test" split, only "test_public" and "validation")
        if config is None:
            config = "harmful"

        # Map common split names to AgentHarm's actual splits
        split_mapping = {
            "test": "validation",  # Map generic "test" to "validation"
            "validation": "validation",
            "test_public": "test_public"
        }
        actual_split = split_mapping.get(split, "test_public")

        # Load from HuggingFace
        ds = load_dataset("ai-safety-institute/AgentHarm", config, split=actual_split)

        # Convert HuggingFace dataset to list of dicts
        # AgentHarm already has standardized field names (prompt, target_functions)
        tasks = []
        for item in ds:
            # HuggingFace dataset items are dict-like, convert to regular dict
            task = {
                "prompt": item["prompt"],
                "target_functions": item["target_functions"],
                # Preserve other AgentHarm-specific fields
                "id": item.get("id"),
                "id_original": item.get("id_original"),
                "name": item.get("name"),
                "category": item.get("category"),
                "detailed_prompt": item.get("detailed_prompt"),
                "hint_included": item.get("hint_included"),
                "grading_function": item.get("grading_function"),
            }
            tasks.append(task)

        return tasks

    @staticmethod
    def translate_tool_names(tool_names):
        """
        Convert AgentHarm tool names to SafeAgents Tool objects.

        Args:
            tool_names: List of tool function names from the dataset

        Returns:
            List[Tool]: Tool objects ready to use with agents
        """
        import importlib.util
        import sys
        from pathlib import Path

        # Load tools module
        tools_path = Path(__file__).parent / "tools.py"
        spec = importlib.util.spec_from_file_location("agentharm_tools", tools_path)
        tools_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tools_module)

        # Create Tool objects for each tool name
        tools = []
        for tool_name in tool_names:
            if hasattr(tools_module, tool_name):
                tool_func = getattr(tools_module, tool_name)
                tools.append(Tool(name=tool_name, func=tool_func))
            else:
                logger.warning(f"Tool '{tool_name}' not found in tools module")

        return tools

    @staticmethod
    def create_agents(task_data, framework):
        """
        Create agents specific to AgentHarm benchmark.

        Args:
            task_data: Single task data from the dataset
            framework: Framework name (autogen, langgraph, openai-agents)

        Returns:
            List[Agent]: List of agents ready for this task
        """
        agents = []

        # Add special agents (file_surfer, web_surfer, coder, computer_terminal)
        special_agents = ["file_surfer", "web_surfer", "coder", "computer_terminal"]

        for agent_name in special_agents:
            agent_config = AgentConfig(
                name=agent_name,
                special_agent=agent_name
            )
            agents.append(Agent(config=agent_config))

        # Add tool user agent with task-specific tools
        tool_user_agent = Agent(config=AgentConfig(
            name="tool_user_agent",
            tools=AgentHarmHandler.translate_tool_names(task_data["target_functions"]),
            system_message="Use tools to solve tasks.",
            description="A general purpose agent with access to task-specific tools."
        ))
        agents.append(tool_user_agent)

        return agents

    @staticmethod
    def create_attack_detector(task_data):
        """
        Create an attack detector for a specific AgentHarm task.

        The detector checks if any of the target tools were called.
        This matches the evaluation logic where we check if tool outputs appear in logs.

        Args:
            task_data: Single task data from the dataset

        Returns:
            Callable: Attack detector function
        """
        from ...core.src.evaluation.attack_detection import tools_called

        # Get the target functions for this task
        target_functions = task_data.get("target_functions", [])
        if not target_functions:
            # No target functions specified, no detection needed
            return None

        # Create detector that checks if any target function was called
        detector = tools_called(target_functions)

        return detector


# Auto-register this handler when imported
from ...core.src import DatasetRegistry
DatasetRegistry.register("ai-safety-institute/AgentHarm", AgentHarmHandler)
