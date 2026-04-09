# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tool provider for evaluating MCS selection implementations.

Wraps the link adaptation evaluation pipeline as a LangChain tool so that
an agent can submit its ``draft.py`` and receive a performance report.
"""

import shlex
from pathlib import Path

from tool_lib.base import ToolProvider
from tool_lib.workspace import Workspace
from langchain_core.tools import tool, BaseTool


# Paths to evaluation scripts and supporting files
_EVAL_SCRIPT_PATH = Path(__file__).parent / "eval/eval.py"
_UTILS_PATH = Path(__file__).parent / "eval/utils.py"
_DATA_DIR = Path(__file__).parent / "eval/data"


class EvalTool(ToolProvider):
    """Tool provider for evaluating MCS selection implementations.

    Copies the evaluation harness (``eval.py``, ``utils.py``, and data files)
    into the agent's workspace, runs the evaluation script that imports the
    agent's ``draft.py``, and returns a string whose first line has
    the format ``SUCCESS, <metric>`` or ``FAILURE, <metric>``, followed by
    optional lines (BLER statistics, failure details).
    """

    def __init__(self, eval_timeout: int = 120):
        self._eval_timeout = eval_timeout
        self._create_tools()
        self._workspace = None

    _DEFAULT_SOURCE_FILE = "draft.py"

    def _evaluate_mcs_selection(self) -> str:
        """Evaluate the agent's MCS selection algorithm.

        Runs a link adaptation simulation across multiple SINR trajectory
        scenarios and measures long-term BLER and average spectral
        efficiency.

        Before calling this tool the agent must create a file called
        ``draft.py`` in the workspace that defines::

            def mcs_selection(is_nack_hist: np.ndarray,
                              mcs_ackned_hist: np.ndarray,
                              bler_target: float) -> int:

        Parameters
        ----------
        (none -- the function takes no arguments; it reads the agent's
        ``draft.py`` from the workspace.)

        Function contract for ``mcs_selection``
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        is_nack_hist : numpy.ndarray
            Binary HARQ feedback history (1 = NACK, 0 = ACK) in
            chronological order.
        mcs_ackned_hist : numpy.ndarray
            History of MCS indices whose HARQ feedback has already been
            received, in chronological order.
        bler_target : float
            Long-term BLER target (e.g., 0.1 for 10 %).

        The function must return an ``int`` — the MCS index for the next
        transmission.

        Evaluation criteria
        ~~~~~~~~~~~~~~~~~~~
        * The controller is tested across multiple scenarios with
          different SINR dynamics.
        * For each scenario the long-term BLER must be at most 10 %.
        * The metric is the average spectral efficiency (bits/s/Hz) —
          **higher is better**.
        * A scenario fails if the BLER constraint is violated, regardless
          of throughput.

        Returns
        -------
        str
            First line: ``SUCCESS, <metric>`` or ``FAILURE, <metric>``
            (metric = average spectral efficiency in bits/s/Hz; higher is
            better). Remaining lines are optional (e.g. BLER statistics in
            log10 scale, failure details).
        """
        return self._evaluate_from_file(self._DEFAULT_SOURCE_FILE)

    def _evaluate_from_file(self, source_file: str) -> str:
        """Run evaluation loading the mcs_selection function from the given workspace file. Not exposed to the agent."""
        if self._workspace is None:
            raise ValueError("Workspace not set")

        # Copy supporting files to the workspace just before running
        # This prevents the agent from modifying them
        supporting_files = [
            (_UTILS_PATH, "utils.py"),
            (_EVAL_SCRIPT_PATH, "eval.py"),
        ]

        for src_path, dst_name in supporting_files:
            with open(src_path, "r") as f:
                self._workspace._write_file(dst_name, f.read())

        # Copy data directory
        self._workspace._create_dir("data")
        for data_file in _DATA_DIR.iterdir():
            if data_file.is_file():
                # Use binary mode for .pkl files, text mode for others
                if data_file.suffix == ".pkl":
                    with open(data_file, "rb") as f:
                        self._workspace._write_file_binary(f"data/{data_file.name}", f.read())
                else:
                    with open(data_file, "r") as f:
                        self._workspace._write_file(f"data/{data_file.name}", f.read())

        # Run eval.py with optional source filename as argument
        cmd = f"python eval.py {shlex.quote(source_file)}"
        success, output = self._workspace._exec(cmd, timeout=self._eval_timeout)
        result = output if success else f"Error:\n{output}"

        # Clean up supporting files from the workspace
        for _, dst_name in supporting_files:
            self._workspace._delete(dst_name)
        self._workspace._delete("data")

        return result

    def run_evaluation(self, filename: str) -> str:
        """Run evaluation using the given solution filename. Used by POST_EVAL; not a LangChain tool."""
        return self._evaluate_from_file(filename)

    def _create_tools(self):
        """Register bound methods as LangChain tools."""
        self.evaluate_mcs_selection = tool(self._evaluate_mcs_selection)

    def get_tools(self) -> list[BaseTool]:
        """Return the list of LangChain tools exposed by this provider."""
        return [self.evaluate_mcs_selection]

    def set_workspace(self, workspace: Workspace):
        """Bind the provider to a :class:`Workspace` instance.

        Parameters
        ----------
        workspace : Workspace
            The agent's sandboxed workspace where ``draft.py``
            lives and where evaluation artefacts will be temporarily
            copied.
        """
        self._workspace = workspace
