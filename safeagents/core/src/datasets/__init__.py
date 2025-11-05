"""
Dataset handling for SafeAgents.

Dataset handlers are auto-imported by safeagents.core.src.__init__.py
Users don't need to import them manually.
"""

from .dataset import Dataset, DatasetRegistry

__all__ = [
    "Dataset",
    "DatasetRegistry",
]
