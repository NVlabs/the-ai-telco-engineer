# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Launch script for optimization agents.

Takes a task folder path as argument. The task folder must contain:
- config.json: Configuration file for the agent manager
- eval_tool.py: Evaluation tool module with an `EvalTool` class
- Any other files requires to perform the optimization

Usage:
    python launch.py <task_folder>

Example:
    python launch.py tasks/chn_est
"""

import sys
import argparse
import importlib.util
from pathlib import Path

# Add src/ to path before importing from it
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agent_manager import AgentManager
from config import load_config


def load_eval_tool(task_folder: Path):
    """Dynamically load the EvalTool class from the task folder.

    Args:
        task_folder: Path to the task folder containing eval_tool.py

    Returns:
        The EvalTool class from the task's eval_tool.py module.
    """
    eval_tool_path = task_folder / "eval_tool.py"
    if not eval_tool_path.exists():
        raise FileNotFoundError(f"eval_tool.py not found in {task_folder}")

    # Load the module dynamically
    spec = importlib.util.spec_from_file_location("eval_tool", eval_tool_path)
    module = importlib.util.module_from_spec(spec)

    # Add task folder to path so eval_tool can import local modules
    sys.path.insert(0, str(task_folder))

    # Register module in sys.modules so pickle can find it
    sys.modules["eval_tool"] = module

    spec.loader.exec_module(module)

    # Get the EvalTool class
    if not hasattr(module, "EvalTool"):
        raise AttributeError(f"eval_tool.py must define an 'EvalTool' class")

    return module.EvalTool


def load_tool_factory(task_folder: Path):
    """Optionally load the ToolFactory class from the task folder.

    Args:
        task_folder: Path to the task folder potentially containing tool_factory.py

    Returns:
        The ToolFactory class if found, None otherwise.
    """
    tool_factory_path = task_folder / "tool_factory.py"
    if not tool_factory_path.exists():
        return None

    # Load the module dynamically
    spec = importlib.util.spec_from_file_location("tool_factory", tool_factory_path)
    module = importlib.util.module_from_spec(spec)

    # Register module in sys.modules so pickle can find it
    sys.modules["tool_factory"] = module

    spec.loader.exec_module(module)

    # Get the ToolFactory class if it exists
    if not hasattr(module, "ToolFactory"):
        print(f"Warning: tool_factory.py exists but doesn't define 'ToolFactory' class")
        return None

    return module.ToolFactory


def main():
    parser = argparse.ArgumentParser(
        description="Launch optimization agents for a task",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python launch.py tasks/chn_est
    python launch.py tasks/link_adaptation
        """
    )
    parser.add_argument(
        "task_folder",
        type=str,
        help="Path to the task folder containing config.json and eval_tool.py"
    )
    args = parser.parse_args()

    # Resolve task folder path
    task_folder = Path(args.task_folder).resolve()
    if not task_folder.exists():
        print(f"Error: Task folder not found: {task_folder}")
        sys.exit(1)

    # Load configuration
    config_path = task_folder / "config.json"
    if not config_path.exists():
        print(f"Error: config.json not found in {task_folder}")
        sys.exit(1)
    config = load_config(config_path)

    # Load the evaluation tool (required)
    try:
        EvalTool = load_eval_tool(task_folder)
    except (FileNotFoundError, AttributeError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Load the tool factory (optional)
    ToolFactory = load_tool_factory(task_folder)

    print(f"Task folder: {task_folder}")
    print(f"Config: {config_path}")
    print(f"Prompt: {task_folder / config.prompt_path}")
    print(f"Eval tool: {EvalTool.__name__}")
    if ToolFactory:
        print(f"Tool factory: {ToolFactory.__name__}")

    # Create and run agent manager
    agent_manager = AgentManager(
        config,
        evaluation_tool_type=EvalTool,
        task_folder=task_folder,
        tool_factory_type=ToolFactory
    )

    with agent_manager as manager:
        manager.run()


if __name__ == "__main__":
    main()
