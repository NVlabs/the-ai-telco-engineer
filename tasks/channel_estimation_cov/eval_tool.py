# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import shlex
from pathlib import Path

from tool_lib.base import ToolProvider
from tool_lib.workspace import Workspace
from langchain_core.tools import tool, BaseTool


# Paths to evaluation scripts
_EVAL_SCRIPT_PATH = Path(__file__).parent / "eval/eval.py"
_LINK_CONFIG_PATH = Path(__file__).parent / "eval/link_config.py"
_BLER_PERF_CSI_PATH = Path(__file__).parent / "eval/bler_perf_csi.pkl"
_CHANNEL_COV_PATH = Path(__file__).parent / "eval/channel_cov.pkl"


class EvalTool(ToolProvider):
    """Tool provider for evaluating MIMO detector implementations.

    Provides a tool that evaluates a MIMO detector for a 4x16 MIMO link
    using the CDL channel model. The metric is the SNR (dB) at which BLER
    equals 0.001 (higher is better).

    Output format (first line only is parsed by the framework):
    ``SUCCESS, <metric>`` or ``FAILURE,`` followed by a newline and optional
    human-readable text.

    The agent must create a file called 'draft.py' in the workspace that
    implements the mimo_detector() function.
    """

    def __init__(self, eval_timeout: int = 240):
        self._eval_timeout = eval_timeout
        self._create_tools()
        self._workspace = None

    _DEFAULT_SOURCE_FILE = "draft.py"

    def _evaluate_from_file(self, source_file: str) -> str:
        """Run evaluation loading the mimo_detector from the given workspace file. Not exposed to the agent."""
        if self._workspace is None:
            raise ValueError("Workspace not set")

        # Copy supporting files to the workspace just before running
        with open(_LINK_CONFIG_PATH, "r") as f:
            self._workspace._write_file("link_config.py", f.read())
        with open(_EVAL_SCRIPT_PATH, "r") as f:
            self._workspace._write_file("eval.py", f.read())
        with open(_BLER_PERF_CSI_PATH, "rb") as f:
            self._workspace._write_file_binary("bler_perf_csi.pkl", f.read())
        with open(_CHANNEL_COV_PATH, "rb") as f:
            self._workspace._write_file_binary("channel_cov.pkl", f.read())
        # Run eval.py with source filename as argument
        cmd = f"python eval.py {shlex.quote(source_file)}"
        success, output = self._workspace._exec(cmd, timeout=self._eval_timeout)
        result = output if success else f"Error:\n{output}"

        # Clean up supporting files from the workspace
        self._workspace._delete("eval.py")
        self._workspace._delete("link_config.py")
        self._workspace._delete("bler_perf_csi.pkl")
        self._workspace._delete("channel_cov.pkl")
        return result

    def run_evaluation(self, filename: str) -> str:
        """Run evaluation using the given solution filename. Used by POST_EVAL; not a LangChain tool."""
        return self._evaluate_from_file(filename)

    def _evaluate_mimo_detector(self) -> str:
        """Evaluate the MIMO detector implementation.

        Expects `draft.py` in the workspace defining `mimo_detector(y, no)`.
        See the task prompt for signature, link_config symbols, and constraints.
        Runs Monte Carlo simulation; link_config.py is injected at evaluation time.

        Returns:
            str: First line ``SUCCESS, <metric>`` (SNR in dB at BLER=0.001) or
            ``FAILURE,``; remaining lines are optional human-readable text.
        """
        return self._evaluate_from_file(self._DEFAULT_SOURCE_FILE)

    def _create_tools(self):
        """Create tools from bound methods."""
        self.evaluate_mimo_detector = tool(self._evaluate_mimo_detector)

    def get_tools(self) -> list[BaseTool]:
        """Get the tools provided by this class."""
        return [self.evaluate_mimo_detector]

    def set_workspace(self, workspace: Workspace):
        """Set the workspace for the evaluator."""
        self._workspace = workspace
