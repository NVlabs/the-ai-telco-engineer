# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import printer


@dataclass
class LLMConfig:
    """Configuration for creating an LLM instance."""

    api_key: str = ""
    base_url: str = ""
    model: str = ""
    temperature: float = 0.7
    top_p: float = 0.95
    model_kwargs: dict[str, Any] = field(default_factory=dict)

@dataclass
class WorkspaceConfig:
    """Configuration for workspace Docker containers."""

    path: str = "workspaces"
    docker_image: str = "agent_container"
    memory_limit: str = "16g"
    pids_limit: int = 2048
    use_gpu: bool = True


class ToolsConfig:
    """Dynamic configuration for tools parameters.
    
    This class wraps a dictionary and provides attribute-style access.
    Any parameters defined in the JSON's tools_config section are available.
    
    Example JSON:
        "tools_config": {
            "cache_dir_path": "api_doc_cache",
            "custom_param": "value"
        }
    
    Usage:
        config.tools_config.cache_dir_path  # "api_doc_cache"
        config.tools_config.custom_param    # "value"
        config.tools_config.get("missing", "default")  # "default"
    """
    
    def __init__(self, data: Optional[dict[str, Any]] = None):
        """Initialize with a dictionary of parameters.
        
        Args:
            data: Dictionary of tool configuration parameters.
        """
        self._data = data if data is not None else {}
    
    def __getattr__(self, name: str) -> Any:
        """Get a configuration value by attribute name."""
        if name.startswith("_"):
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")
        if name not in self._data:
            raise AttributeError(f"ToolsConfig has no parameter '{name}'")
        return self._data[name]
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value with a default."""
        return self._data.get(key, default)
    
    def __contains__(self, key: str) -> bool:
        """Check if a key exists in the configuration."""
        return key in self._data
    
    def __repr__(self) -> str:
        return f"ToolsConfig({self._data})"
    
    def to_dict(self) -> dict[str, Any]:
        """Return the underlying dictionary."""
        return self._data.copy()


@dataclass
class Config:
    """Configuration for the agent manager and optimization run."""

    # LLM configuration: both required (use LLMConfig for each)
    agent_llm: LLMConfig = field(default_factory=LLMConfig)   # Used by agents (workers)
    manager_llm: LLMConfig = field(default_factory=LLMConfig)  # Used by manager (ideas, summaries)

    # Workspace configuration
    workspace: WorkspaceConfig = field(default_factory=WorkspaceConfig)
    tools_config: ToolsConfig = field(default_factory=ToolsConfig)

    # Worker configuration
    num_workers: int = 10

    # Metric optimization direction
    higher_is_better: bool = False

    # Optimization parameters
    population_size: int = 20
    num_generations: int = 5
    num_ideas: int = 3  # Number of distinct algorithmic approaches per generation
    timeout: int = 900  # seconds per agent
    task_submit_delay: float = 30.0  # seconds between task submissions

    # Prompt file
    prompt_path: str = "prompt.md"


def load_config(config_path: Union[str, Path]) -> Config:
    """Load configuration from a JSON file.
    
    Args:
        config_path: Path to the JSON configuration file.
        
    Returns:
        Config instance with the loaded configuration.
    """
    config_path = Path(config_path)
    with open(config_path) as f:
        # Remove comments (lines starting with //) for JSON parsing
        lines = f.readlines()
        clean_lines = [line for line in lines if not line.strip().startswith("//")]
        data = json.loads("".join(clean_lines))

    # Get API key from environment variable (required)
    api_key = os.environ.get("MODEL_API_KEY")
    if not api_key:
        printer.section("Error: MODEL_API_KEY environment variable is not set.",
                        "Please set it with: export MODEL_API_KEY=<your-api-key>")
        sys.exit(1)

    # Extract agent LLM config (required)
    if "agent_llm" not in data:
        printer.log("Error: config must contain 'agent_llm' section (LLM used by agents).")
        sys.exit(1)
    agent_llm_section = data["agent_llm"]
    agent_llm = LLMConfig(
        api_key=api_key,
        base_url=agent_llm_section.get("base_url", ""),
        model=agent_llm_section.get("model", ""),
        temperature=agent_llm_section.get("temperature", 0.7),
        top_p=agent_llm_section.get("top_p", 0.95),
        model_kwargs=agent_llm_section.get("model_kwargs", {}),
    )

    # Extract manager LLM config (required)
    if "manager_llm" not in data:
        printer.log("Error: config must contain 'manager_llm' section (LLM used for ideas and summaries).")
        sys.exit(1)
    manager_llm_section = data["manager_llm"]
    manager_llm = LLMConfig(
        api_key=api_key,
        base_url=manager_llm_section.get("base_url", ""),
        model=manager_llm_section.get("model", ""),
        temperature=manager_llm_section.get("temperature", 0.0),
        top_p=manager_llm_section.get("top_p", 0.95),
        model_kwargs=manager_llm_section.get("model_kwargs", {}),
    )

    # Extract workspace config from nested section
    workspace_section = data.get("workspace", {})
    workspace = WorkspaceConfig(
        path=str(config_path.parent / workspace_section.get("path", "workspaces")),
        docker_image=workspace_section.get("docker_image", "agent_container"),
        memory_limit=workspace_section.get("memory_limit", "16g"),
        pids_limit=workspace_section.get("pids_limit", 2048),
        use_gpu=workspace_section.get("use_gpu", True),
    )

    # Extract tools config from nested section (dynamic - any keys allowed)
    tools_config = ToolsConfig(data.get("tools_config", {}))

    return Config(
        agent_llm=agent_llm,
        manager_llm=manager_llm,
        workspace=workspace,
        tools_config=tools_config,
        num_workers=data.get("num_workers", 10),
        higher_is_better=data.get("higher_is_better", False),
        population_size=data.get("population_size", 20),
        num_generations=data.get("num_generations", 5),
        num_ideas=data.get("num_ideas", 3),
        timeout=data.get("timeout", 900),
        task_submit_delay=data.get("task_submit_delay", 30.0),
        prompt_path=data.get("prompt_path", "prompt.md"),
    )
