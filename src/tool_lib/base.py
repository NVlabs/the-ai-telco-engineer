# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Base classes for tool providers.
"""

from abc import ABC, abstractmethod
from langchain_core.tools import BaseTool


class ToolProvider(ABC):
    """Base class for objects that provide LangChain tools.

    Subclasses must implement :meth:`get_tools`.  Override :meth:`build`
    when the tool requires expensive one-time setup (e.g. building a
    vector-store index) that should happen before worker processes are
    spawned.
    """

    @abstractmethod
    def get_tools(self) -> list[BaseTool]:
        """Return a list of tools this provider offers."""
        ...

    @classmethod
    def build(cls, tools_config) -> None:
        """One-time pre-spawn setup for this tool provider.

        Called by the manager process before worker processes are created.
        Use this for expensive, shared initialisation that should only
        happen once (e.g. building a FAISS index on disk).

        The default implementation does nothing.

        Args:
            tools_config: A :class:`~config.ToolsConfig` instance with
                the current run's tool parameters.
        """
        pass
