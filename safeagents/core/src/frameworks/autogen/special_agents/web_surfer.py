"""
Web Surfer wrapper for Autogen framework with tool tracking support.

This module simply re-exports MultimodalWebSurfer from autogen_ext.
Tool tracking is handled via monkey-patching in the agents factory.
"""

from autogen_ext.agents.web_surfer import MultimodalWebSurfer

__all__ = ["MultimodalWebSurfer"]
