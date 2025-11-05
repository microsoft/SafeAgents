"""
SafeAgents Datasets

This module contains dataset-specific handlers for various benchmarks.
Each dataset has its own subdirectory with:
- tools.py: Dataset-specific tool implementations
- __init__.py: Handler class that auto-registers with DatasetRegistry

To add a new dataset:
1. Create a new directory: safeagents/datasets/your_dataset/
2. Add tools.py with your tool functions
3. Add __init__.py with YourDatasetHandler class
4. Import your dataset here to enable auto-registration
5. The handler will auto-register when this module is imported

Available datasets are automatically registered on import.

Example structure:
    safeagents/datasets/
    ├── __init__.py          (this file - imports all dataset handlers)
    ├── agentharm/
    │   ├── __init__.py      (AgentHarmHandler + auto-registration)
    │   └── tools.py         (AgentHarm-specific tools)
    └── your_dataset/
        ├── __init__.py      (YourDatasetHandler + auto-registration)
        └── tools.py         (Your dataset-specific tools)
"""

# Auto-import all dataset handlers to register them
from . import agentharm
from . import asb

__all__ = ['agentharm', 'asb']
