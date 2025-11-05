"""
SafeAgents - Framework for building safe multi-agent systems.

Simplified top-level imports for user-facing API.

Example:
    from safeagents import Agent, Team, Dataset

    # Instead of:
    # from safeagents.core.src import Agent, Team, Dataset
"""

# Re-export all user-facing classes from safeagents.core.src
from .core.src import (
    # Models
    Agent,
    AgentConfig,
    Task,
    Tool,
    tool,
    Prompt,
    DesignChoices,
    AutonomyLevel,
    PlanningStrategy,
    # Frameworks
    Team,
    TeamRegistry,
    register_framework,
    Framework,
    Architecture,
    TeamAutogen,
    TeamLanggraph,
    TeamOpenAIAgents,
    # Evaluation
    Assessment,
    aria,
    dharma,
    ARIA_PROMPT,
    DHARMA_PROMPT,
    # Datasets
    Dataset,
    DatasetRegistry,
    # Clients
    get_azure_config,
    get_azure_openai_client_sync,
    get_azure_openai_client_async,
    Model,
    ModelConfig,
    # Configuration
    EnvironmentSetup,
    # Safety
    Mitigation,
)

__all__ = [
    # Models
    "Agent",
    "AgentConfig",
    "Task",
    "Tool",
    "tool",
    "Prompt",
    "DesignChoices",
    "AutonomyLevel",
    "PlanningStrategy",
    # Frameworks
    "Team",
    "TeamRegistry",
    "register_framework",
    "Framework",
    "Architecture",
    "TeamAutogen",
    "TeamLanggraph",
    "TeamOpenAIAgents",
    # Evaluation
    "Assessment",
    "aria",
    "dharma",
    "ARIA_PROMPT",
    "DHARMA_PROMPT",
    # Datasets
    "Dataset",
    "DatasetRegistry",
    # Clients
    "get_azure_config",
    "get_azure_openai_client_sync",
    "get_azure_openai_client_async",
    "Model",
    "ModelConfig",
    # Configuration
    "EnvironmentSetup",
    # Safety
    "Mitigation",
]

# Package version
__version__ = "0.1.0"
