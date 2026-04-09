# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Agent module - A LangGraph/LangChain agent.
"""

import os
import re
import signal
from dataclasses import asdict
from datetime import datetime
from typing import Optional

import numpy as np
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from config import LLMConfig, WorkspaceConfig, ToolsConfig
from tool_lib.base import ToolProvider
from tool_lib.workspace import Workspace


class AgentTimeoutError(BaseException):
    """Raised when agent execution exceeds the timeout.

    Inherits from BaseException (not Exception) so that it bypasses
    LangGraph's ToolNode and run_with_retry handlers, both of which
    catch ``except Exception`` and would silently swallow the timeout.
    The worker retry loop in _worker_fn catches it explicitly before
    the generic ``except Exception`` clause, so retry semantics are
    correct.
    """
    pass


DRAFT_FILE = "draft.py"
SOLUTION_FILE = "solution.py"

# Prompt template to enrich the user query
# Placeholders: original_query, assigned_approach_section, metric_direction, better_metric_description
RESULT_PROMPT_TEMPLATE = """
You are an implementing agent in an automated optimization loop. You receive one task and one assigned approach. There is no prior conversation: this message is your only context.

## Context
- You run in an isolated workspace with tools: read/write files, copy files, run code, and an evaluation tool.
- Optimization target: {metric_direction} values are BETTER.
- Your final deliverable is 'solution.py'. A post-run evaluation will score this file automatically.

## Two-file workflow
You work with TWO files:
1. **`draft.py`** — your scratch pad. The evaluation tool tests this file. Write, edit, and experiment here freely.
2. **`solution.py`** — your safe vault. Use `copy_file("draft.py", "solution.py")` to save your best result. This protects your work: if you are interrupted or time out mid-experiment, `solution.py` still holds your best solution.

**IMPORTANT — always save early.** The first time the evaluation tool returns SUCCESS on your own approach (not the baseline), immediately run `copy_file("draft.py", "solution.py")` — even if the metric is worse than the baseline. A saved solution you can improve on is always better than no solution at all. After that first save, overwrite `solution.py` only when you achieve a metric that is {better_metric_description} than your previously saved best.

Never experiment directly in `solution.py`. Always edit `draft.py`, evaluate, and only then copy to `solution.py`.

**Your goal is to achieve the best possible metric** — beat the baseline, then keep pushing further. Do NOT save the baseline code itself as your solution — only save your own novel implementations following the assigned approach.

## Task
{original_query}

## Assigned approach (you MUST follow this)
{assigned_approach_section}

Do not switch to a different strategy. If reference code is shown, use it to understand the idea — your goal is to beat it, not reproduce it.

## Workflow

### Step 1 — Understand the setup
Read the evaluation tool description to learn the expected function signature, available imports, and constraints.

### Step 2 — Implement your approach
Write your solution into `draft.py`. Use provided configuration objects; do not hardcode values. Keep all logic inline — importing installed libraries is fine, but do not import from other files you create.

### Step 3 — Evaluate
Call the evaluation tool. It returns a metric ({metric_direction} is better), optional hints, or an error.

### Step 4 — Save your result
If this is your **first successful evaluation** (no `solution.py` yet), run `copy_file("draft.py", "solution.py")` immediately — even if the metric is worse than the baseline. Do this **before** making any further changes. If `solution.py` already exists, only overwrite it when the new metric is {better_metric_description} than your previous best.

### Step 5 — Iterate
Keep improving: adjust parameters, add algorithmic refinements, re-evaluate. Repeat Steps 2–4. Always save before risky changes. Stop only when you run out of ideas or time.

## Rules
- Do not call the evaluation tool twice without changing code between calls.
- After the first success, keep iterating — a run with only 1-2 attempts wastes your budget.
- Do not deviate from the assigned approach.
- Do NOT save a verbatim copy of the baseline as your solution.
- CRITICAL: both `draft.py` and `solution.py` must be fully self-contained. All logic you write must be defined inline. Importing from installed libraries (e.g. `numpy`, `scipy`, `torch`) is fine, but do NOT import from other files you created (e.g. `from helper import solve`).

=== END INSTRUCTIONS ===
"""

# Evaluation tool output format: first line is "SUCCESS, <metric>" or "FAILURE, <metric>" or "FAILURE,"
# Optional lines follow. Parser does not use an LLM.
_EVAL_FIRST_LINE = re.compile(r"^(SUCCESS|FAILURE)\s*,\s*([+-]?[\d.]+)?\s*$", re.IGNORECASE)


def parse_eval_output(eval_output: str) -> tuple[bool, Optional[float], Optional[str]]:
    """Parse evaluation tool output. First line must be 'SUCCESS, <metric>' or 'FAILURE, <metric>' or 'FAILURE,'.

    Returns:
        (success, metric, info). metric is None if missing or parse fails.
        info contains any additional lines after the first (stripped), or None if empty.
    """
    if not eval_output or not isinstance(eval_output, str):
        return False, None, None
    lines = eval_output.strip().split("\n")
    first_line = lines[0].strip()
    m = _EVAL_FIRST_LINE.match(first_line)
    if not m:
        return False, None, None
    success = m.group(1).upper() == "SUCCESS"
    metric_str = m.group(2)
    info = "\n".join(lines[1:]).strip() or None
    if metric_str is None or metric_str == "":
        return success, None, info
    try:
        return success, float(metric_str), info
    except (ValueError, TypeError):
        return success, None, info


class Agent:
    """
    A LangGraph/LangChain agent that can process queries.
    """

    def __init__(self, llm_config: LLMConfig,
                workspace_config: WorkspaceConfig,
                evaluation_tool_type: type[ToolProvider],
                tool_factory_type: Optional[type[ToolProvider]],
                tools_config: ToolsConfig,
                higher_is_better: bool = False):
        """
        Initialize the agent.

        Args:
            llm_config: The configuration for the language model to use.
            workspace_config: Configuration for workspace Docker containers.
            tools_config: Configuration for tools.
            evaluation_tool_type: The type of evaluation tool provider to use.
            tool_factory_type: Optional factory class for creating additional tools.
            higher_is_better: If True, higher metric values are better (e.g., throughput).
                             If False, lower metric values are better (e.g., error rate).
        """
        self.workspace_config = workspace_config
        self.higher_is_better = higher_is_better
        self._current_journal_path = None  # Set during run() for timeout logging

        # Build the LLM
        self.llm = ChatOpenAI(**asdict(llm_config))

        # Create task-specific tools from factory (if provided)
        self.tool_factory = None
        if tool_factory_type is not None:
            self.tool_factory = tool_factory_type(tools_config)

        # Evaluation tool
        eval_timeout = tools_config.get("eval_timeout", 120)
        self.evaluation_tool = evaluation_tool_type(eval_timeout=eval_timeout)

    def enrich_query(self, query: str, assigned_approach_section: str = "") -> str:
        """
        Enrich the original query with instructions and the assigned approach section.

        The assigned approach section is built by the agent manager (idea description
        followed by its reference code). This method only injects it into the template.

        Args:
            query: The original user query.
            assigned_approach_section: Pre-built section from the manager (idea + its reference code).

        Returns:
            The enriched query with instructions and assigned approach section.
        """
        if not assigned_approach_section.strip():
            assigned_approach_section = "(No specific approach assigned.)"
        metric_direction = "HIGHER" if self.higher_is_better else "LOWER"
        better_metric_description = "higher" if self.higher_is_better else "lower"
        return RESULT_PROMPT_TEMPLATE.format(
            original_query=query,
            assigned_approach_section=assigned_approach_section,
            metric_direction=metric_direction,
            better_metric_description=better_metric_description,
        )

    def _log_message(self, journal_path: str, msg) -> None:
        """
        Log a message to the journal file.

        Args:
            journal_path: Path to the journal.log file.
            msg: The message object to log.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = []

        if msg.type == "ai":
            # AI message - may contain content and/or tool calls
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_name = tc.get("name", "unknown")
                    tool_args = tc.get("args", {})
                    tool_id = tc.get("id", "")
                    args_str = ", ".join(f"{k}={repr(v)}" for k, v in tool_args.items())
                    lines.append(f"[{timestamp}] [TOOL_CALL] {tool_name}({args_str}) [id: {tool_id}]")

            if hasattr(msg, "content") and msg.content:
                lines.append(f"[{timestamp}] [AI]\n{msg.content}")

        elif msg.type == "tool":
            # Tool response message
            tool_name = getattr(msg, "name", "unknown")
            tool_call_id = getattr(msg, "tool_call_id", "")
            content = getattr(msg, "content", "") or ""
            lines.append(f"[{timestamp}] [TOOL_RESPONSE] {tool_name} [id: {tool_call_id}]\n{content}")

        elif msg.type == "human":
            # Human/user message
            content = getattr(msg, "content", "") or ""
            lines.append(f"[{timestamp}] [USER]\n{content}")

        else:
            # Other message types
            content = getattr(msg, "content", "") or ""
            lines.append(f"[{timestamp}] [{msg.type.upper()}]\n{content}")

        if lines:
            with open(journal_path, "a", encoding="utf-8") as f:
                for line in lines:
                    f.write(line + "\n\n")

    def _timeout_handler(self, signum, frame) -> None:
        """
        Signal handler for timeout. Logs the timeout event and raises an exception.

        Args:
            signum: Signal number.
            frame: Current stack frame.
        """
        if self._current_journal_path:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(self._current_journal_path, "a", encoding="utf-8") as f:
                f.write(f"[{timestamp}] [TIMEOUT] Agent execution timed out\n\n")

        raise AgentTimeoutError("Agent execution timed out")

    def _run_post_agent_evaluation(self, workspace: Workspace, journal_path: str) -> None:
        """Run the evaluation tool in the workspace and save result to result.npy.

        Evaluation tools must implement run_evaluation(filename: str) -> str so that
        POST_EVAL always evaluates the configured solution file.

        If solution.py does not exist but draft.py does, draft.py is used as a fallback
        so that work-in-progress is not silently discarded on timeout.

        result.npy is saved as a dict {'success': bool, 'metric': float|None, 'info': str|None}:
        - On success: metric holds the parsed float, info holds any additional output lines.
        - On failure: metric is None, info holds the error message or full evaluation output.
        result.npy is not saved if neither solution.py nor draft.py exist.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if not hasattr(self.evaluation_tool, "run_evaluation") or not callable(
            getattr(self.evaluation_tool, "run_evaluation")
        ):
            with open(journal_path, "a", encoding="utf-8") as f:
                f.write(
                    f"[{timestamp}] [POST_EVAL] Evaluation tool must implement run_evaluation(filename); skipping\n\n"
                )
            return

        solution_path = workspace._host_workspace_path / SOLUTION_FILE
        draft_path = workspace._host_workspace_path / DRAFT_FILE

        # Determine which file to evaluate: prefer solution.py, fall back to draft.py
        if solution_path.exists():
            eval_file = SOLUTION_FILE
        elif draft_path.exists():
            eval_file = DRAFT_FILE
            with open(journal_path, "a", encoding="utf-8") as f:
                f.write(
                    f"[{timestamp}] [POST_EVAL] '{SOLUTION_FILE}' not found; falling back to '{DRAFT_FILE}'\n\n"
                )
        else:
            with open(journal_path, "a", encoding="utf-8") as f:
                f.write(
                    f"[{timestamp}] [POST_EVAL] Neither '{SOLUTION_FILE}' nor '{DRAFT_FILE}' found; not saving result.npy\n\n"
                )
            return

        def _save_result(result_dict: dict) -> None:
            try:
                np.save(str(workspace._host_workspace_path / "result.npy"), result_dict, allow_pickle=True)
            except Exception as save_e:
                with open(journal_path, "a", encoding="utf-8") as f:
                    f.write(f"[{timestamp}] [POST_EVAL] Save failed: {save_e}\n\n")

        try:
            eval_output = self.evaluation_tool.run_evaluation(eval_file)
            if not isinstance(eval_output, str):
                eval_output = str(eval_output)
        except Exception as e:
            error_msg = str(e)
            with open(journal_path, "a", encoding="utf-8") as f:
                f.write(f"[{timestamp}] [POST_EVAL] Evaluation tool failed: {error_msg}\n\n")
            _save_result({"success": False, "metric": None, "info": error_msg})
            with open(journal_path, "a", encoding="utf-8") as f:
                f.write(f"[{timestamp}] [POST_EVAL] Saved failure result.npy with error message\n\n")
            return

        with open(journal_path, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] [POST_EVAL] Evaluation output ({eval_file}):\n{eval_output}\n\n")
        success, metric, info = parse_eval_output(eval_output)
        if not success or metric is None:
            with open(journal_path, "a", encoding="utf-8") as f:
                f.write(f"[{timestamp}] [POST_EVAL] Evaluation did not succeed or no metric; saving failure result.npy\n\n")
            _save_result({"success": False, "metric": None, "info": eval_output})
            return
        _save_result({"success": True, "metric": metric, "info": info})
        with open(journal_path, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] [POST_EVAL] Saved metric {metric} to result.npy\n\n")

    def run(
        self,
        workspace_id: str,
        query: str,
        timeout: Optional[int] = None,
        assigned_approach_section: str = "",
    ) -> str:
        """
        Run the agent on a query and return the final response.

        Args:
            workspace_id: The workspace ID to use for this run.
            query: The user query to process.
            timeout: Optional timeout in seconds for the agent run.
                     If None, no timeout is applied.
            assigned_approach_section: Pre-built section from the manager (idea followed by its reference code).

        Returns:
            The final AI response as a string.

        Raises:
            AgentTimeoutError: If the agent execution exceeds the timeout.
        """
        # Enrich query with instructions and the assigned approach section (manager-built)
        effective_query = self.enrich_query(query, assigned_approach_section)

        # Set up journal log path in the workspace directory (needed for timeout logging)
        workspace_dir = os.path.join(self.workspace_config.path, workspace_id)
        journal_path = os.path.join(workspace_dir, "journal.log")
        self._current_journal_path = journal_path

        # Set up timeout if specified
        if timeout is not None:
            signal.signal(signal.SIGALRM, self._timeout_handler)
            signal.alarm(timeout)

        try:
            # Use context manager for automatic workspace cleanup after run
            # No workspace inheritance - each candidate starts fresh
            with Workspace(workspace_id,
                          host_workspace_path=self.workspace_config.path,
                          docker_image=self.workspace_config.docker_image,
                          memory_limit=self.workspace_config.memory_limit,
                          pids_limit=self.workspace_config.pids_limit,
                          use_gpu=self.workspace_config.use_gpu) as workspace:
                # Write the query as the first entry in the journal
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open(journal_path, "w", encoding="utf-8") as f:
                    f.write(f"[{timestamp}] [QUERY]\n{effective_query}\n\n")

                # Set workspace on tool factory (if provided)
                if self.tool_factory is not None:
                    self.tool_factory.set_workspace(workspace)

                # Set workspace on evaluation tool for tool invocation
                self.evaluation_tool.set_workspace(workspace)

                # Collect all tools
                tools = workspace.get_tools() + self.evaluation_tool.get_tools()
                if self.tool_factory is not None:
                    tools += self.tool_factory.get_tools()
                # Build the agent
                agent = create_agent(self.llm, tools)

                final_response = ""
                tool_call_count = 0
                last_ai_has_tool_calls = None
                last_ai_content_preview = ""
                last_ai_content_len = 0
                try:
                    for event in agent.stream({"messages": [("user", effective_query)]}):
                        for _, node_output in event.items():
                            if "messages" in node_output:
                                for msg in node_output["messages"]:
                                    # Log all messages to journal
                                    self._log_message(journal_path, msg)
                                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                                        tool_call_count += len(msg.tool_calls)
                                    if msg.type == "ai":
                                        if msg.content:
                                            final_response = msg.content
                                        last_ai_has_tool_calls = bool(
                                            getattr(msg, "tool_calls", None)
                                        )
                                        content = (msg.content or "").strip()
                                        last_ai_content_len = len(content)
                                        last_ai_content_preview = (
                                            content[:200] + "..." if len(content) > 200
                                            else content
                                        )
                except AgentTimeoutError:
                    # Timeout: cancel alarm and run POST_EVAL in same workspace, then re-raise.
                    if timeout is not None:
                        signal.alarm(0)
                    self._run_post_agent_evaluation(workspace, journal_path)
                    raise

                # Normal completion: run POST_EVAL (cancel alarm so it is not interrupted).
                if timeout is not None:
                    signal.alarm(0)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open(journal_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"[{timestamp}] [RUN_END] tool_calls={tool_call_count} | "
                        f"last_ai_has_tool_calls={last_ai_has_tool_calls} | "
                        f"last_ai_content_len={last_ai_content_len}\n"
                    )
                    if last_ai_content_preview:
                        f.write(f"[{timestamp}] [RUN_END] last_ai_content_preview: {last_ai_content_preview!r}\n")
                    f.write("\n")
                self._run_post_agent_evaluation(workspace, journal_path)
            return final_response

        finally:
            if timeout is not None:
                signal.alarm(0)
