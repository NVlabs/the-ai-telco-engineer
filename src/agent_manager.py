# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
AgentManager module - Manages a pool of LangChain agents running in parallel
using Python's multiprocessing module.

Idea-driven optimization:
- Manager LLM produces n distinct algorithmic ideas from the task query (or from
  previous generation results).
- Population of m agents is split across ideas (m/n per idea).
- When a task completes, manager LLM summarizes the solution in context of the idea.
- When a generation completes, manager LLM produces n ideas for the next generation.
"""

import json
import re
import signal
import time
import numpy as np
import multiprocessing as mp
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional
import random
from langchain_openai import ChatOpenAI

import printer
from config import Config, LLMConfig, WorkspaceConfig, ToolsConfig
from agent import Agent, AgentTimeoutError, SOLUTION_FILE, DRAFT_FILE
from tool_lib.base import ToolProvider
from leaderboard import Candidate, ClusteredLeaderboard
from utils import (
    INITIAL_BACKOFF_SECONDS,
    MAX_BACKOFF_SECONDS,
    MAX_RATE_LIMIT_RETRIES,
    invoke_llm_with_retry,
    is_rate_limit_error,
)

# Bracket pairs for JSON extraction from LLM output (handles strings and escape)
_JSON_BRACKETS = ("[", "]"), ("{", "}")


def _extract_json_fragment(text: str, open_char: str) -> Optional[str]:
    """Extract a JSON array or object from text (from first open_char to matching close). Returns None if not found."""
    close_char = next(c for o, c in _JSON_BRACKETS if o == open_char)
    start = text.find(open_char)
    if start == -1:
        return None
    depth = 0
    in_string = None
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if escape:
            escape = False
            continue
        if c == "\\" and in_string:
            escape = True
            continue
        if in_string:
            if c == in_string:
                in_string = None
            continue
        if c in ('"', "'"):
            in_string = c
            continue
        if c == open_char:
            depth += 1
        elif c == close_char:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


@dataclass
class Task:
    """A task with a workspace ID, base query, and assigned approach section (idea + its reference code, built by manager)."""
    workspace_id: str
    query: str
    assigned_approach_section: str = ""  # Idea description followed by its reference code; manager builds this
    timeout: Optional[int] = None
    generation: int = 0


@dataclass
class TaskResult:
    """Result from a completed task."""
    workspace_id: str
    query: str
    response: str
    success: bool
    error: Optional[str] = None
    generation: int = 0


def _worker_fn(worker_id: int,
               print_lock: mp.Lock,
               agent_llm: LLMConfig,
               workspace_config: WorkspaceConfig,
               evaluation_tool_type: type[ToolProvider],
               tool_factory_type: Optional[type[ToolProvider]],
               tools_config: ToolsConfig,
               higher_is_better: bool,
               task_queue: mp.Queue, result_queue: mp.Queue):
    """
    Worker function that runs in a subprocess.
    Creates an Agent and processes tasks with per-task workspaces.
    Includes exponential backoff retry for rate limit errors.
    """
    # Convert SIGTERM (from p.terminate()) into SystemExit so that Python
    # unwinds the stack and context managers (e.g. Workspace) run cleanup.
    def _sigterm_handler(signum, frame):
        raise SystemExit("Worker terminated")
    signal.signal(signal.SIGTERM, _sigterm_handler)

    printer.init(print_lock, f"WORKER-{worker_id}")

    # Create Agent once per worker (reuses LLM and API doc tools across tasks)
    agent = Agent(agent_llm, workspace_config, evaluation_tool_type,
                  tool_factory_type, tools_config, higher_is_better)

    while True:
        task = task_queue.get()

        # Check for shutdown signal
        if task is None:
            break

        workspace_id = task.workspace_id
        query = task.query
        printer.set_header(f"AGENT-{workspace_id}")

        # Retry loop with exponential backoff for rate limit errors
        last_error = None
        for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
            try:
                response = agent.run(
                    workspace_id,
                    query,
                    timeout=task.timeout,
                    assigned_approach_section=task.assigned_approach_section,
                )
                result = TaskResult(
                    workspace_id=workspace_id,
                    query=query,
                    response=response,
                    success=True,
                    generation=task.generation
                )
                break  # Success - exit retry loop

            except AgentTimeoutError as e:
                # Timeout errors should not be retried
                result = TaskResult(
                    workspace_id=workspace_id,
                    query=query,
                    response="",
                    success=False,
                    error=f"Timeout: {str(e)}",
                    generation=task.generation
                )
                break

            except Exception as e:
                last_error = e

                # Check if this is a rate limit error
                if is_rate_limit_error(e) and attempt < MAX_RATE_LIMIT_RETRIES:
                    # Calculate backoff with exponential increase and jitter
                    backoff = min(
                        INITIAL_BACKOFF_SECONDS * (2 ** attempt),
                        MAX_BACKOFF_SECONDS
                    )
                    # Add some jitter (0-25% of backoff)
                    jitter = random.uniform(0, backoff * 0.25)
                    wait_time = backoff + jitter

                    printer.log(f"Rate limit hit, attempt {attempt + 1}/{MAX_RATE_LIMIT_RETRIES}. "
                                f"Waiting {wait_time:.1f}s before retry...")
                    time.sleep(wait_time)
                    continue  # Retry

                # Non-rate-limit error or max retries exceeded
                result = TaskResult(
                    workspace_id=workspace_id,
                    query=query,
                    response="",
                    success=False,
                    error=str(e),
                    generation=task.generation
                )
                break

        result_queue.put(result)


class AgentManager:
    """
    Manages a pool of LangChain agents running as separate processes.

    Idea-driven optimization with a manager LLM for generating ideas and
    summarizing solutions; clusters are assigned by idea.
    """

    IDEAS_INITIAL_PROMPT = """You are a research lead. Your job is to decompose a single task into exactly {n} distinct algorithmic approaches. Each approach will be assigned to a separate implementing agent, so each idea must be actionable, self-contained, and different in kind (not minor variants of the same approach).

## Context
- There is no prior conversation: this is a standalone request.
- The task below is the full problem statement. Propose {n} genuinely different strategies (e.g. different algorithms, data structures, or solution paradigms).

## Task
{query}

## Your output (strict format)
Reply with **only** a JSON array of exactly {n} ideas. No other text before or after the array.

Each element is an object with one key:
- "description": string (one or more sentences describing the approach in enough detail that an implementer can follow it)

Example for n=2:
[{{"description": "Enumerate candidate solutions and evaluate each; keep the best."}}, {{"description": "Build the solution step by step with a local greedy rule at each step."}}]

Your response (JSON array only):"""

    IDEAS_FROM_RESULTS_PROMPT = """You are a research lead in an iterative optimization loop. One or more generations of agents have run. Each agent was assigned one idea (cluster); you see their results below, grouped by generation. There is no prior conversation: use only the information in this message.

## Task (problem to solve)
{query}

## Context
- We run multiple generations. Each generation, agents implement ideas and produce a metric.
- Metric interpretation: {metric_direction}
- Your job: propose exactly {n} ideas for the *next* generation. These can be refinements of previous ideas (with clearer or updated instructions), combinations, or new ideas.

## Results from all previous generations (grouped by generation, then cluster)
Each cluster had one assigned idea; under it are workspace ids with a short summary and metric for each candidate produced for that idea. Workspace ids (e.g. gen00-0001) can be used as reference code for the next round: if you list them in REFERENCE_WORKSPACES, that code will be shown to the agent for that new idea.

{results_text}

## Your output (strict format)
Reply with **only** a JSON array of exactly {n} ideas. No other text before or after the array.

Each element is an object with:
- "description": string (the approach for the next generation, one or more sentences)
- "reference_workspaces": optional string of comma-separated workspace ids from the results above (e.g. "gen00-0001, gen00-0003"), or omit if no reference code is needed. Use only workspace ids that appear in the results above.

Example for n=2:
[{{"description": "Enumerate candidate solutions and evaluate each; keep the best.", "reference_workspaces": "gen00-0001, gen00-0003"}}, {{"description": "Build the solution step by step with a local greedy rule."}}]

Your response (JSON array only):"""

    SUMMARIZE_PROMPT = """You are summarizing a single implementation for a research coordinator. There is no prior conversation: use only the information below.

## Context
- An agent was assigned one specific approach (idea). It produced the solution code below.
- Your summary will be used to decide the next set of ideas for future agents. Be factual and concise: what was actually implemented, and how does it relate to the assigned idea?

## Assigned idea (the approach this agent was supposed to follow)
{idea_description}

## Solution code produced by the agent
```python
{code}
```

## Your task
In 2–4 sentences, summarize: (1) what approach/techniques were used in the code, and (2) how they align with or deviate from the assigned idea. No preamble or meta-commentary.

Reply with **only** a JSON object with one key: "summary" (the summary text). Example: {{"summary": "The code implements ..."}}"""

    def __init__(self, config: Config,
                 evaluation_tool_type: type[ToolProvider],
                 task_folder: Path,
                 tool_factory_type: Optional[type[ToolProvider]] = None):
        """
        Initialize the AgentManager.

        Can be used as a context manager for automatic start/stop:
            with AgentManager(config, EvalTool, task_folder) as manager:
                manager.run()

        Args:
            config: Config dataclass containing all configuration.
            evaluation_tool_type: The evaluation tool class for this task.
            task_folder: Path to the task folder (used for loading prompt).
            tool_factory_type: Optional factory class for creating additional tools.
        """
        # Use 'spawn' instead of 'fork' to be compatible with CUDA
        mp.set_start_method('spawn', force=True)

        self.config = config
        self.task_folder = Path(task_folder)

        # Load the prompt from task folder
        prompt_path = self.task_folder / config.prompt_path
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        self.prompt = prompt_path.read_text()

        self.task_queue: mp.Queue = mp.Queue()
        self.result_queue: mp.Queue = mp.Queue()
        self._print_lock: mp.Lock = mp.Lock()
        printer.init(self._print_lock, "MANAGER")
        self.workers: list[mp.Process] = []
        self._candidate_counter = 0  # Counter for deterministic workspace IDs
        self._manager_llm = None  # LLM for ideas and summaries, initialized lazily
        self.evaluation_tool_type = evaluation_tool_type
        self.tool_factory_type = tool_factory_type
        # workspace_id -> (cluster_id, idea_description) for current generation
        self._workspace_to_idea: dict[str, tuple[int, str]] = {}

    @property
    def _host_workspace_path(self) -> Path:
        return Path(self.config.workspace.path)

    @property
    def _leaderboard_path(self) -> Path:
        return self._host_workspace_path / "leaderboard.json"

    @property
    def _manager_log_path(self) -> Path:
        return self._host_workspace_path / "manager.log"

    def __enter__(self):
        """Context manager entry - starts worker processes."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - stops worker processes."""
        self.stop()
        return False  # Don't suppress exceptions

    def start(self):
        """Start all worker processes.

        Calls :meth:`~tool_lib.base.ToolProvider.build` on the evaluation
        tool and the optional tool factory *before* spawning workers so
        that expensive one-time setup runs once in the manager process.
        """
        printer.log(f"Starting {self.config.num_workers} agent workers...")

        # Ensure workspace directory exists
        self._host_workspace_path.mkdir(parents=True, exist_ok=True)

        # One-time pre-spawn setup for tool providers
        self.evaluation_tool_type.build(self.config.tools_config)
        if self.tool_factory_type is not None:
            if not hasattr(self.tool_factory_type, "TOOL_TYPES"):
                raise AttributeError(
                    f"{self.tool_factory_type.__name__} must define a TOOL_TYPES "
                    "class attribute listing its ToolProvider types."
                )
            for tool_type in self.tool_factory_type.TOOL_TYPES:
                tool_type.build(self.config.tools_config)

        for i in range(self.config.num_workers):
            p = mp.Process(
                target=_worker_fn,
                args=(i, self._print_lock, self.config.agent_llm, self.config.workspace,
                      self.evaluation_tool_type, self.tool_factory_type, self.config.tools_config,
                      self.config.higher_is_better,
                      self.task_queue, self.result_queue)
            )
            p.start()
            self.workers.append(p)
            printer.log(f"Worker {i} started (PID: {p.pid})")

        printer.log(f"All {self.config.num_workers} workers ready.")

    def stop(self):
        """Stop all worker processes."""
        printer.log("Stopping workers...")

        # Send shutdown signal to each worker
        for _ in self.workers:
            self.task_queue.put(None)

        # Wait for workers to finish
        for i, p in enumerate(self.workers):
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
                printer.log(f"Worker {i} terminated")
            else:
                printer.log(f"Worker {i} stopped")

        self.workers.clear()
        printer.log("All workers stopped.")

    def run(self) -> ClusteredLeaderboard:
        """
        Run the optimization. Automatically resumes if a leaderboard exists,
        otherwise starts fresh.

        All optimization parameters (population_size, num_generations, etc.)
        are taken from the Config provided at initialization. The prompt is loaded
        from the task folder at initialization.

        Returns:
            The ClusteredLeaderboard with all candidates from all generations.
        """
        # Check if we should resume from existing leaderboard
        if self._leaderboard_path.exists():
            # Resume mode
            leaderboard = ClusteredLeaderboard.load(self._leaderboard_path)
            all_candidates = leaderboard.get_all_candidates()
            printer.log(f"Resuming: Loaded existing leaderboard with {len(all_candidates)} candidates "
                        f"in {len(leaderboard.clusters)} clusters")

            # Determine starting generation (continue from last)
            if all_candidates:
                start_generation = max(c.generation for c in all_candidates) + 1
                self._candidate_counter = len(all_candidates)
            else:
                start_generation = 0
                self._candidate_counter = 0

            return self._run(
                query=self.prompt,
                leaderboard=leaderboard,
                start_generation=start_generation
            )
        else:
            # Start fresh
            self._candidate_counter = 0
            return self._run(query=self.prompt)

    def _run(self, query: str,
            leaderboard: Optional[ClusteredLeaderboard] = None,
            start_generation: int = 0) -> ClusteredLeaderboard:
        """
        Run idea-driven optimization.

        Algorithm:
        1. Manager LLM produces n ideas from the task query (or from previous results).
        2. Create m tasks (m/n per idea); each agent gets one idea.
        3. When a task completes, manager LLM summarizes the solution in context of the idea.
        4. When the generation is done, manager LLM produces n ideas for the next generation.
        5. Repeat from step 2.

        Args:
            query: The original user query.
            leaderboard: Optional existing leaderboard to continue from.
            start_generation: Generation number to start from (default 0).

        Returns:
            The final ClusteredLeaderboard with all candidates from all generations.
        """
        # Get parameters from config
        cfg = self.config
        population_size = cfg.population_size
        num_generations = cfg.num_generations
        timeout = cfg.timeout
        task_submit_delay = cfg.task_submit_delay

        # Determine if we're starting fresh or continuing
        is_continuation = leaderboard is not None and start_generation > 0
        num_ideas = cfg.num_ideas

        # Print header
        header_lines = [
            "",
            "=" * 60,
            f"Continuing Optimization from Generation {start_generation}" if is_continuation
            else "Starting Idea-driven Optimization",
            "=" * 60,
            f"Query: {query[:100]}{'...' if len(query) > 100 else ''}",
        ]
        if is_continuation:
            header_lines.append(f"Existing candidates: {len(leaderboard.get_all_candidates())}")
            header_lines.append(f"Existing clusters: {len(leaderboard.clusters)}")
        header_lines.extend([
            f"Population size: {population_size}",
            f"Number of ideas: {num_ideas}",
            f"Generations: {num_generations}",
            f"Timeout per agent: {timeout}s",
            f"Task submit delay: {task_submit_delay}s",
            "=" * 60,
            "",
        ])
        printer.section(*header_lines)

        # Initialize or use existing leaderboard
        if leaderboard is None:
            leaderboard = ClusteredLeaderboard(query=query, higher_is_better=self.config.higher_is_better)
            leaderboard.save(self._leaderboard_path)
        ideas: list[tuple[int, str, list[str]]] = []  # (cluster_id, idea_description, ref_workspace_ids)
        self._all_gen_summaries: list[tuple[int, int, str, str, float, str]] = []  # (generation, cluster_id, idea_desc, summary, metric, workspace_id)

        # Run generations
        for gen_offset in range(num_generations):
            generation = start_generation + gen_offset

            printer.section("", "=" * 60, f"Generation {generation}", "=" * 60, "")

            # Use initial ideas only for gen 0 or first gen after resume (_all_gen_summaries empty then)
            use_initial_ideas = (
                generation == 0
                or (is_continuation and generation == start_generation and not self._all_gen_summaries)
            )
            try:
                if use_initial_ideas:
                    descriptions = self._generate_initial_ideas(query, num_ideas)
                    refs_per_idea: list[list[str]] = [[] for _ in descriptions]
                else:
                    descriptions, refs_per_idea = self._generate_ideas_from_results(
                        self._all_gen_summaries, num_ideas, query
                    )
            except Exception as e:
                phase = "initial ideas" if use_initial_ideas else "ideas from previous results"
                printer.log(f"ERROR: Manager LLM failed to generate {phase}: {e}")
                printer.log("Stopping optimization.")
                break

            ids = leaderboard.get_next_cluster_ids(len(descriptions))
            ideas = [(cid, desc, refs) for cid, desc, refs in zip(ids, descriptions, refs_per_idea)]
            idea_lines = [f"Ideas for this generation ({len(ideas)}):"]
            for cid, desc, _ in ideas:
                idea_lines.append(f"  [{cid}] {desc[:80]}{'...' if len(desc) > 80 else ''}")
            printer.section(*idea_lines)

            # Register clusters (and their descriptions) on the leaderboard for this generation
            for cluster_id, idea_desc, _ in ideas:
                leaderboard.add_cluster(cluster_id, idea_desc)

            # Run generation
            try:
                self._run_generation(
                    query=query,
                    ideas=ideas,
                    population_size=population_size,
                    generation=generation,
                    timeout=timeout,
                    leaderboard=leaderboard,
                    task_submit_delay=task_submit_delay
                )
            except Exception as e:
                printer.log(f"ERROR: Generation {generation} failed: {e}")
                printer.log("Stopping optimization.")
                break

            self._print_generation_summary(leaderboard, generation)

        self._print_final_summary(leaderboard)
        return leaderboard

    def _get_manager_llm(self) -> ChatOpenAI:
        """Get or create the LLM used for generating ideas and summarizing solutions."""
        if self._manager_llm is None:
            config_dict = asdict(self.config.manager_llm)
            self._manager_llm = ChatOpenAI(**config_dict)
        return self._manager_llm

    def _parse_initial_ideas_json(self, response_text: str, n: int) -> Optional[list[str]]:
        """Parse JSON array of initial ideas. Returns None if parsing fails."""
        text = (response_text or "").strip()
        code_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if code_match:
            text = code_match.group(1).strip()
        text = _extract_json_fragment(text, "[")
        if text is None:
            return None
        try:
            data = json.loads(text)
            if not isinstance(data, list) or len(data) < n:
                return None
            descriptions: list[str] = []
            for item in data[:n]:
                if not isinstance(item, dict):
                    return None
                desc = item.get("description")
                if desc is not None:
                    desc = str(desc).strip()
                descriptions.append(desc or "No description")
            return descriptions[:n]
        except (json.JSONDecodeError, TypeError, ValueError):
            return None

    def _parse_summary_json(self, response_text: str) -> Optional[str]:
        """Parse JSON object with 'summary' key. Returns None if parsing fails."""
        text = (response_text or "").strip()
        code_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if code_match:
            text = code_match.group(1).strip()
        text = _extract_json_fragment(text, "{")
        if text is None:
            return None
        try:
            data = json.loads(text)
            if not isinstance(data, dict):
                return None
            s = data.get("summary")
            return str(s).strip() if s is not None else None
        except (json.JSONDecodeError, TypeError, ValueError):
            return None

    def _log_manager(self, role: str, content: str) -> None:
        """Append a manager LLM turn to manager.log in the workspace root."""
        self._host_workspace_path.mkdir(parents=True, exist_ok=True)
        with open(self._manager_log_path, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*60}\n[{role}]\n{content}\n")

    def _generate_initial_ideas(self, query: str, n: int) -> list[str]:
        """Ask manager LLM for n initial ideas from the task query. Returns list of descriptions.

        Raises on LLM or parsing failure (rate-limit retries are handled internally by invoke_llm_with_retry).
        """
        llm = self._get_manager_llm()
        prompt = self.IDEAS_INITIAL_PROMPT.format(n=n, query=query)
        self._log_manager("PROMPT (initial ideas)", prompt)
        response = invoke_llm_with_retry(llm, prompt, context="initial ideas")
        content = response.content.strip()
        self._log_manager("RESPONSE (initial ideas)", content)
        parsed = self._parse_initial_ideas_json(content, n)
        if parsed is None:
            raise ValueError(f"Failed to parse initial ideas from manager LLM response:\n{content}")
        return parsed

    def _summarize_solution(self, code: str, idea_description: str) -> str:
        """Ask manager LLM for a summary of the solution in context of the idea.

        Raises on LLM failure (rate-limit retries are handled internally by invoke_llm_with_retry).
        Falls back to the raw LLM response if JSON parsing fails.
        """
        if not code.strip():
            return "No code provided."
        llm = self._get_manager_llm()
        prompt = self.SUMMARIZE_PROMPT.format(
            idea_description=idea_description,
            code=code
        )
        self._log_manager("PROMPT (summarize)", prompt)
        response = invoke_llm_with_retry(llm, prompt, context="summarize")
        content = response.content.strip() or "No summary."
        self._log_manager("RESPONSE (summarize)", content)
        summary = self._parse_summary_json(content)
        return summary if summary is not None else content

    def _generate_ideas_from_results(self,
                                     all_gen_summaries: list[tuple[int, int, str, str, float, str]],
                                     n: int,
                                     query: str = "",
                                     ) -> tuple[list[str], list[list[str]]]:
        """Generate ideas from all previous generation summaries. Returns (descriptions, ref_workspace_ids per idea)."""
        # all_gen_summaries: (generation, cluster_id, idea_desc, summary, metric, workspace_id)
        by_gen: dict[int, dict[int, list[tuple[str, float, str]]]] = {}  # generation -> cluster_id -> [(summary, metric, workspace_id), ...]
        idea_descriptions: dict[int, str] = {}
        for generation, cluster_id, idea_desc, summary, metric, workspace_id in all_gen_summaries:
            idea_descriptions[cluster_id] = idea_desc
            if generation not in by_gen:
                by_gen[generation] = {}
            if cluster_id not in by_gen[generation]:
                by_gen[generation][cluster_id] = []
            by_gen[generation][cluster_id].append((summary, metric, workspace_id))

        results_lines = []
        for generation in sorted(by_gen.keys()):
            results_lines.append(f"=== Generation {generation} ===")
            for cluster_id, pairs in by_gen[generation].items():
                desc = idea_descriptions.get(cluster_id, "")
                results_lines.append(f"  --- Cluster {cluster_id} ---")
                results_lines.append(f"  Description: {desc}")
                for summary, metric, ws_id in pairs:
                    results_lines.append(f"    Workspace {ws_id}: Summary: {summary}; Metric: {metric}")
                    _, _, info = self._read_result_metric(ws_id)
                    if info:
                        results_lines.append(f"      Additional evaluation info: {info}")
                results_lines.append("")
            results_lines.append("")

        metric_direction = (
            "Higher metric values are better (e.g. reward, throughput)."
            if self.config.higher_is_better
            else "Lower metric values are better (e.g. loss, error rate)."
        )
        llm = self._get_manager_llm()
        prompt = self.IDEAS_FROM_RESULTS_PROMPT.format(
            n=n,
            query=query,
            metric_direction=metric_direction,
            results_text="\n".join(results_lines),
        )
        self._log_manager("PROMPT (ideas from results)", prompt)
        response = invoke_llm_with_retry(llm, prompt, context="ideas from results")
        content = response.content.strip()
        self._log_manager("RESPONSE (ideas from results)", content)
        descriptions, refs_per_idea = self._parse_ideas_from_results_response(content, n)
        if not descriptions:
            raise ValueError(f"Failed to parse ideas from manager LLM response:\n{content}")
        return descriptions, refs_per_idea

    def _parse_ideas_from_results_response(
        self, response_text: str, n: int
    ) -> tuple[list[str], list[list[str]]]:
        """Parse LLM response as JSON array of ideas. On failure returns empty lists."""
        parsed = self._parse_ideas_from_results_json(response_text, n)
        if parsed is not None:
            return parsed
        return [], [[] for _ in range(n)]

    def _parse_ideas_from_results_json(
        self, response_text: str, n: int
    ) -> Optional[tuple[list[str], list[list[str]]]]:
        """Parse JSON array of ideas. Returns None if parsing fails."""
        text = (response_text or "").strip()
        code_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if code_match:
            text = code_match.group(1).strip()
        text = _extract_json_fragment(text, "[")
        if text is None:
            return None
        try:
            data = json.loads(text)
            if not isinstance(data, list) or len(data) < n:
                return None
            descriptions: list[str] = []
            refs_per_idea: list[list[str]] = []
            for item in data[:n]:
                if not isinstance(item, dict):
                    return None
                desc = item.get("description")
                if desc is not None:
                    desc = str(desc).strip()
                descriptions.append(desc or "No description")
                refs_raw = item.get("reference_workspaces")
                refs: list[str] = []
                if refs_raw is not None:
                    if isinstance(refs_raw, list):
                        refs = [str(x).strip() for x in refs_raw if str(x).strip()]
                    else:
                        refs = [
                            w.strip()
                            for w in str(refs_raw).replace(",", " ").split()
                            if w.strip()
                        ]
                refs_per_idea.append(refs)
            return descriptions[:n], refs_per_idea[:n]
        except (json.JSONDecodeError, TypeError, ValueError):
            return None

    def _run_generation(self, query: str,
                        ideas: list[tuple[int, str, list[str]]],  # (cluster_id, idea_description, reference_workspace_ids)
                        population_size: int,
                        generation: int,
                        timeout: int,
                        leaderboard: ClusteredLeaderboard,
                        task_submit_delay: float):
        """
        Run a single generation: m/n tasks per idea (m = population_size, n = len(ideas)).

        Args:
            query: The base task query.
            ideas: List of (cluster_id, idea_description, reference_workspace_ids).
            population_size: Total number of tasks (m).
            generation: The generation number.
            timeout: Timeout per agent in seconds.
            leaderboard: The leaderboard to update.
            task_submit_delay: Delay in seconds between task submissions.
        """
        n = len(ideas)
        tasks_per_idea = max(1, population_size // n)
        num_candidates = tasks_per_idea * n
        self._workspace_to_idea.clear()

        printer.log(f"Submitting {num_candidates} tasks ({tasks_per_idea} per idea, {task_submit_delay}s stagger)...")

        # Submit tasks: round-robin or batch per idea so we can track workspace_id -> idea
        submitted_ids = []
        for idx in range(num_candidates):
            idea_idx = idx % n
            cluster_id, idea_description, ref_workspace_ids = ideas[idea_idx]
            workspace_id = self._submit_task(
                query=query,
                idea_description=idea_description,
                reference_workspace_ids=ref_workspace_ids or [],
                timeout=timeout,
                generation=generation
            )
            self._workspace_to_idea[workspace_id] = (cluster_id, idea_description)
            submitted_ids.append(workspace_id)

            if idx < num_candidates - 1 and task_submit_delay > 0:
                time.sleep(task_submit_delay)

        printer.log(f"Submitted {len(submitted_ids)} tasks. Waiting for results...")

        completed = 0
        for _ in range(num_candidates):
            result = self._get_result()
            if result is None:
                printer.log(f"[{completed + 1}/{num_candidates}] ✗ Failed to retrieve result from queue")
                continue
            candidate = self._process_result(result, leaderboard)
            completed += 1
            status = "✓" if candidate.success else "✗"
            metric_str = f"{candidate.metric:.6f}" if candidate.success else "N/A"
            cluster_str = f"[{candidate.cluster}]" if candidate.cluster is not None else ""
            printer.log(f"[{completed}/{num_candidates}] {status} {result.workspace_id} "
                        f"{cluster_str} (metric: {metric_str})")

    def _submit_task(self, query: str,
                     idea_description: str = "",
                     reference_workspace_ids: Optional[list[str]] = None,
                     timeout: Optional[int] = None,
                     generation: int = 0) -> str:
        """Internal method to submit a task to be processed by a worker.

        Args:
            query: The base task query (no idea or reference code in it).
            idea_description: Assigned approach description (passed as task field).
            reference_workspace_ids: Optional workspace ids whose solution code is attached to this idea.
            timeout: Optional timeout in seconds for the agent run.
            generation: The generation number for this candidate.

        Returns:
            The workspace_id for this task.
        """
        workspace_id = f"gen{generation:02d}-{self._candidate_counter:04d}"
        self._candidate_counter += 1

        # Build assigned approach section: idea description immediately followed by its reference code.
        # Only include references that have a valid metric; skip failed/unevaluated runs.
        lines = [
            "=== ASSIGNED APPROACH (you MUST follow this) ===",
            idea_description.strip(),
        ]
        valid_ref_metrics: list[float] = []
        if reference_workspace_ids:
            for ref_id in reference_workspace_ids:
                ref_code = self._read_workspace_code(ref_id)
                metric_ok, metric_val, ref_info = self._read_result_metric(ref_id)
                if not metric_ok or not ref_code:
                    continue
                valid_ref_metrics.append(metric_val)
                lines.append("")
                lines.append(f"Reference code from workspace {ref_id} (metric: {metric_val:.6f}):")
                lines.append("```python")
                lines.append(ref_code.strip())
                lines.append("```")
                if ref_info:
                    lines.append("")
                    lines.append(f"Additional evaluation info for {ref_id}:")
                    lines.append(ref_info.strip())
        if valid_ref_metrics:
            best = min(valid_ref_metrics) if not self.config.higher_is_better else max(valid_ref_metrics)
            direction = "lower" if not self.config.higher_is_better else "higher"
            lines.append("")
            lines.append(
                f"Your target: achieve a metric strictly {direction} than {best:.6f} "
                f"(the best reference above). If your first result matches this value, "
                f"you have only reproduced the reference — you must improve further."
            )
        lines.append("")
        lines.append("=== END ASSIGNED APPROACH ===")
        assigned_approach_section = "\n".join(lines)

        task = Task(
            workspace_id=workspace_id,
            query=query,
            assigned_approach_section=assigned_approach_section,
            timeout=timeout,
            generation=generation,
        )
        self.task_queue.put(task)
        return workspace_id

    def _get_result(self, timeout: Optional[float] = None) -> Optional[TaskResult]:
        """Internal method to get a result from the result queue."""
        try:
            return self.result_queue.get(timeout=timeout)
        except Exception:
            return None

    def _process_result(self, result: TaskResult, leaderboard: ClusteredLeaderboard) -> Candidate:
        """
        Process a task result: read metric and code, get summary from manager LLM,
        add candidate to leaderboard with cluster = idea_id.

        A candidate is considered successful if result.npy exists and contains a valid scalar.
        """
        metric_success, metric, _ = self._read_result_metric(result.workspace_id)
        code = self._read_workspace_code(result.workspace_id)

        idea_id, idea_description = self._workspace_to_idea.get(
            result.workspace_id, (0, "Unknown idea")
        )

        # Manager LLM summarizes the solution in context of the idea
        summary = self._summarize_solution(code, idea_description)
        self._all_gen_summaries.append((result.generation, idea_id, idea_description, summary, metric, result.workspace_id))

        if metric_success:
            candidate = Candidate(
                workspace_id=result.workspace_id,
                metric=metric,
                generation=result.generation,
                code=code,
                success=True,
                error=None,
                cluster=idea_id,
            )
        else:
            worst_metric = float('-inf') if self.config.higher_is_better else float('inf')
            error_msg = "No result.npy found"
            if result.error:
                error_msg = f"{error_msg} ({result.error})"
            candidate = Candidate(
                workspace_id=result.workspace_id,
                metric=worst_metric,
                generation=result.generation,
                code=code,
                success=False,
                error=error_msg,
                cluster=idea_id,
            )

        leaderboard.add_candidate(candidate)
        leaderboard.save(self._leaderboard_path)
        return candidate

    def _read_workspace_code(self, workspace_id: str) -> str:
        """
        Read the solution code from a workspace.

        Prefers solution.py; falls back to draft.py if solution.py is absent
        (mirrors the fallback in Agent._run_post_agent_evaluation).

        Args:
            workspace_id: The workspace ID to read from.

        Returns:
            The code as a string, or empty string if not found.
        """
        workspace_path = self._host_workspace_path / workspace_id
        solution_path = workspace_path / SOLUTION_FILE
        draft_path = workspace_path / DRAFT_FILE
        try:
            if solution_path.exists():
                return solution_path.read_text()
            elif draft_path.exists():
                printer.log(f"Warning: {SOLUTION_FILE} not found in {workspace_id}; falling back to {DRAFT_FILE}")
                return draft_path.read_text()
            else:
                printer.log(f"Warning: Neither {SOLUTION_FILE} nor {DRAFT_FILE} found in {workspace_id}")
                return ""
        except Exception as e:
            printer.log(f"Warning: Failed to read code from {workspace_id}: {e}")
            return ""

    def _read_result_metric(self, workspace_id: str) -> tuple[bool, float, Optional[str]]:
        """
        Read the result metric from a workspace's result.npy file.

        Supports two formats:
        - New dict format (allow_pickle): {'success': bool, 'metric': float|None, 'info': str|None}
        - Legacy format: a scalar float64 (treated as a successful result with no info)

        Args:
            workspace_id: The workspace ID to read from.

        Returns:
            A tuple of (success, metric, info). If reading fails, returns (False, worst_metric, None)
            where worst_metric is -inf for higher_is_better or inf otherwise.
            success is True only if result.npy exists and contains a valid scalar metric from a
            successful evaluation. info holds optional additional information from the evaluation.
        """
        worst_metric = float('-inf') if self.config.higher_is_better else float('inf')
        result_path = self._host_workspace_path / workspace_id / "result.npy"
        try:
            if not result_path.exists():
                return False, worst_metric, None
            raw = np.load(result_path, allow_pickle=True)
            if raw.ndim == 0 and raw.dtype == object:
                data = raw.item()
                if isinstance(data, dict):
                    eval_success = bool(data.get("success", False))
                    info = data.get("info", None)
                    metric_val = data.get("metric", None)
                    if not eval_success or metric_val is None:
                        return False, worst_metric, info
                    return True, float(metric_val), info
            return True, float(raw), None
        except Exception as e:
            printer.log(f"Warning: Failed to read result.npy for {workspace_id}: {e}")
            return False, worst_metric, None

    def _print_final_summary(self, leaderboard: ClusteredLeaderboard):
        """Print the final summary after optimization completes."""
        all_candidates = leaderboard.get_all_candidates()
        successful_all = leaderboard.get_successful_candidates()

        lines = [
            "",
            "=" * 60,
            "Optimization Complete",
            "=" * 60,
            f"Total candidates evaluated: {len(all_candidates)}",
            f"Total successful: {len(successful_all)}",
            f"Total clusters: {len(leaderboard.clusters)}",
        ]

        if leaderboard.clusters:
            lines.append("")
            lines.append("Cluster Summary:")
            cluster_summary = leaderboard.get_cluster_summary()
            worst_default = float('-inf') if self.config.higher_is_better else float('inf')
            for cluster, info in sorted(
                cluster_summary.items(),
                key=lambda x: x[1]["best_metric"] if x[1]["best_metric"] is not None else worst_default,
                reverse=self.config.higher_is_better
            ):
                best_metric = f"{info['best_metric']:.6f}" if info['best_metric'] is not None else "N/A"
                lines.append(f"  [{cluster}]: {info['successful']}/{info['total']} successful, "
                             f"best: {best_metric}")

        lines.extend([
            "",
            f"Leaderboard saved to: {self._leaderboard_path}",
            "=" * 60,
            "",
        ])
        printer.section(*lines)

    def _print_generation_summary(self, leaderboard: ClusteredLeaderboard, generation: int):
        """Print summary for a completed generation."""
        gen_candidates = leaderboard.get_current_generation_candidates(generation)
        successful = [c for c in gen_candidates if c.success]

        # Count clusters in this generation
        clusters_this_gen = set(c.cluster for c in gen_candidates if c.cluster is not None)

        lines = [
            "",
            f"Generation {generation} Summary:",
            f"  Total candidates: {len(gen_candidates)}",
            f"  Successful: {len(successful)}",
            f"  Clusters used: {len(clusters_this_gen)}",
        ]

        if successful:
            if self.config.higher_is_better:
                best = max(successful, key=lambda c: c.metric)
            else:
                best = min(successful, key=lambda c: c.metric)
            lines.append(f"  Best metric: {best.metric:.6f} (workspace: {best.workspace_id}, "
                         f"cluster: {best.cluster})")

        printer.section(*lines)
