# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tool factory for the channel estimation task.

Provides API documentation tools for Sionna, DrJIT, and Mitsuba libraries.
"""

from tool_lib.base import ToolProvider
from config import ToolsConfig
from tool_lib.sionna_doc import SionnaDoc
from langchain_core.tools import BaseTool


class ToolFactory(ToolProvider):
    """Factory for task-specific tools.

    Creates API documentation tools for the libraries needed by this task.
    """

    TOOL_TYPES = [SionnaDoc]

    def __init__(self, tools_config: ToolsConfig):
        """Initialize the tool factory.

        Args:
            tools_config: Configuration for tools (includes sionna_doc_config etc.).
        """
        cfg = tools_config.get("sionna_doc_config", {})
        self.api_docs = [
            SionnaDoc(
                embedding_model=cfg.get("embedding_model", ""),
                embedding_base_url=cfg.get("embedding_base_url", ""),
                reranker_model=cfg.get("reranker_model", ""),
                reranker_base_url=cfg.get("reranker_base_url", ""),
                retrieve_k=cfg.get("retrieve_k", 12),
                rerank_top_n=cfg.get("rerank_top_n", 4),
                cache_dir=cfg.get("cache_dir_path", "api_doc_cache"),
            ),
        ]
        self._workspace = None

    def get_tools(self) -> list[BaseTool]:
        """Get all tools created by this factory."""
        tools = []
        for api_doc in self.api_docs:
            tools += api_doc.get_tools()
        return tools

    def set_workspace(self, workspace):
        """Set the workspace for all tools."""
        self._workspace = workspace
        for api_doc in self.api_docs:
            api_doc.set_workspace(workspace)
