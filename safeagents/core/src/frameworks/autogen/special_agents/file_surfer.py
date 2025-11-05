"""
File Surfer wrapper for Autogen framework with tool tracking support.

This module simply re-exports FileSurfer from autogen_ext.
Tool tracking is handled via monkey-patching in the agents factory.
"""

from autogen_ext.agents.file_surfer import FileSurfer

__all__ = ["FileSurfer"]
