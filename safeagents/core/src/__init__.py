"""
SafeAgents Core - Framework for building safe multi-agent systems.
"""

# Core models
from .models import (
    Agent,
    AgentConfig,
    Task,
    Tool,
    tool,
    Prompt,
    DesignChoices,
    AutonomyLevel,
    PlanningStrategy,
)

# Frameworks
from .frameworks import (
    Team,
    TeamRegistry,
    register_framework,
    Framework,
    Architecture,
    TeamAutogen,
    TeamLanggraph,
    TeamOpenAIAgents,
)

# Evaluation
from .evaluation import (
    Assessment,
    aria,
    dharma,
    ARIA_PROMPT,
    DHARMA_PROMPT,
)

# Datasets
from .datasets import (
    Dataset,
    DatasetRegistry,
)

# Clients
from .clients import (
    get_azure_config,
    get_azure_openai_client_sync,
    get_azure_openai_client_async,
    Model,
    ModelConfig,
)

# Configuration
from .config import (
    EnvironmentSetup,
)

# Safety
from .safety import (
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

# Auto-import dataset handlers AFTER all imports are complete
# This avoids circular import issues
try:
    import safeagents.datasets  # noqa: F401
except ImportError:
    pass  # Datasets package not available
