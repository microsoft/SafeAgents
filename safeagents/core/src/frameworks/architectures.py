"""
Architecture types for multi-agent teams.
"""
from enum import Enum


class Architecture(Enum):
    """Defines the organizational structure of a multi-agent team."""
    CENTRALIZED = "centralized"
    DECENTRALIZED = "decentralized"
