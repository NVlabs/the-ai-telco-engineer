# [The AI Telco Engineer](The%20AI%20Telco%20Engineer.pdf)

This framework deploys a swarm of parallel agents to autonomously design and optimize wireless communication algorithms for user-defined tasks, such as channel estimation, link adaptation, or LDPC decoding. Each agent is powered by a large language model (LLM) and operates within an isolated, containerized environment. Agents have access to a toolkit that includes file editing capabilities, Sionna documentation, and a task-specific evaluation tool that provides feedback on algorithmic performance.

The framework implements an idea-driven optimization loop. An orchestrator LLM proposes N distinct algorithmic approaches (ideas) for the task. A population of M agents is distributed across those ideas, with each agent implementing and improving one assigned approach in its own isolated workspace. When an agent completes, the orchestrator LLM summarizes its algorithm. At the end of each generation, the orchestrator reviews all summaries and metrics to propose N new ideas for the next generation, optionally referencing the best algorithms found so far as a starting point. Candidates are organized by their assigned idea on the leaderboard.

The system runs multiple LLM agents in parallel to explore and optimize algorithmic approaches. Each agent:

- Is assigned a distinct algorithmic approach by an orchestrator LLM
- Has access to tools for file operations, code execution, and Sionna documentation
- Runs in an isolated Docker container workspace
- Is evaluated with a task-specific evaluation tool
- Contributes to a leaderboard

## Setup

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Set API key

Set your LLM API key as an environment variable:

```bash
export MODEL_API_KEY=<your-api-key>
```

## Tasks

A **task** is a folder that provides everything needed to run the agentic framework for a given problem. Tasks live under `tasks/`.

### Launching a task

Run:

```bash
python launch.py <task_folder>
```

**Examples:**

```bash
# Channel estimation
python launch.py tasks/channel_estimation

# Link adaptation
python launch.py tasks/link_adaptation
```

The framework is bundled with example tasks in `tasks/`. Each includes a `visualize_results.ipynb` notebook that visualizes the algorithms found by the framework and compares them to baselines. You can run the notebook without running the framework; the notebooks use pre-copied algorithms produced by the agents.

### Bundled tasks

| Task | Metric | Direction | Description |
|------|--------|-----------|-------------|
| `channel_estimation` | Normalized Validation Error (NVE) | Lower is better | MIMO channel estimation using Sionna |
| `channel_estimation_cov` | Normalized Validation Error (NVE) | Lower is better | MIMO channel estimation with covariance information |
| `link_adaptation` | Spectral efficiency (bits/s/Hz) | Higher is better | MCS selection controller for link adaptation |

### Leaderboard

The **leaderboard** is the live record of the search: it lists all algorithms produced by the agents, their evaluation outcome (success or failure), and the task metric (e.g. BLER or spectral efficiency).

While a task is running (or after it has run), you can view the leaderboard in a web UI. From the repository root, run:

```bash
./view_leaderboard.py
```

Then open **http://localhost:8000** in your browser.

To point the viewer at a specific workspace folder (e.g. for a task that uses a custom workspace path):

```bash
./view_leaderboard.py --workspace path/to/workspaces
```

Use `./view_leaderboard.py --help` for more options (e.g. `--port`).

## Creating a New Task

Create a new subfolder under `tasks/` with the following:

- **Required:** `config.json`, `prompt.md`, `eval_tool.py`, and a `docker/` folder (Dockerfile and build script).
- **Optional:** `tool_factory.py` for extra tools (e.g. Sionna documentation search).

### 1. Create the task folder

```bash
mkdir -p tasks/my_task/eval
mkdir -p tasks/my_task/docker
```

### 2. Create the Docker container

Add a `Dockerfile` in `tasks/my_task/docker/` with the dependencies for your task, and a build script:

```bash
# tasks/my_task/docker/build_agent_container.sh
docker build -t agent_my_task -f dockerfile_agent_container .
```

This image is used to run agents in isolated workspaces. Agents can install additional packages via PyPI inside the container.

The image name must match `workspace.docker_image` in `config.json`.

### 3. Create required files

**`config.json`** — Task configuration. Copy from an existing task and adapt. Example:

```json
{
    "agent_llm": {
        "model": "<model-name>",
        "base_url": "<api-base-url>",
        "temperature": 0.7,
        "top_p": 0.95
    },
    "manager_llm": {
        "model": "<model-name>",
        "base_url": "<api-base-url>",
        "temperature": 0.0,
        "top_p": 0.95
    },
    "workspace": {
        "path": "workspaces",
        "docker_image": "agent_my_task",
        "memory_limit": "16g",
        "pids_limit": 2048,
        "use_gpu": true
    },
    "tools_config": {
        "eval_timeout": 120
    },
    "num_workers": 10,
    "higher_is_better": false,
    "population_size": 20,
    "num_ideas": 5,
    "num_generations": 5,
    "timeout": 900,
    "task_submit_delay": 30.0,
    "prompt_path": "prompt.md"
}
```

| Parameter | Description |
|-----------|-------------|
| `agent_llm.model` | LLM model used by agents (workers) |
| `agent_llm.base_url` | API base URL for the agent LLM |
| `agent_llm.temperature` | Sampling temperature for agents |
| `agent_llm.top_p` | Top-p (nucleus) sampling for agents (default: `0.95`) |
| `agent_llm.model_kwargs` | Optional extra model kwargs (e.g. `{"reasoning_effort": "high"}`) |
| `manager_llm.model` | LLM model used by the orchestrator (ideas and summaries) |
| `manager_llm.base_url` | API base URL for the orchestrator LLM |
| `manager_llm.temperature` | Sampling temperature for the orchestrator (typically 0.0) |
| `manager_llm.top_p` | Top-p (nucleus) sampling for the orchestrator (default: `0.95`) |
| `manager_llm.model_kwargs` | Optional extra model kwargs for the orchestrator |
| `workspace.path` | Directory for agent workspaces (relative to task folder) |
| `workspace.docker_image` | Docker image for agent containers |
| `workspace.memory_limit` | Memory limit per container (default: `"16g"`) |
| `workspace.pids_limit` | Max processes per container (default: `2048`) |
| `workspace.use_gpu` | Enable GPU access in containers (default: `true`); falls back to CPU if NVIDIA runtime is unavailable |
| `tools_config` | Configuration passed to `ToolFactory` and `EvalTool` |
| `tools_config.eval_timeout` | Timeout in seconds for each evaluation run (default: `120`) |
| `num_workers` | Number of parallel agent workers |
| `higher_is_better` | If true, higher metric values are better |
| `population_size` | Total number of candidates per generation |
| `num_ideas` | Number of distinct algorithmic approaches per generation |
| `num_generations` | Number of optimization generations |
| `timeout` | Timeout in seconds per agent |
| `task_submit_delay` | Delay between task submissions (rate limiting) |
| `prompt_path` | Path to the prompt file, relative to the task folder |

**`prompt.md`** — Task description for the agents. Describe in natural language the problem they should solve.

**`eval_tool.py`** — Must define an `EvalTool` class with:

1. **`run_evaluation(filename: str) -> str`** (required). The framework calls this after each run of an agent to score the algorithm. It must evaluate the workspace file `filename` and return the string format described below. If the file is missing or invalid, return `FAILURE,` optionally followed by message lines.

2. **Output format.** Both the agent-facing evaluation tool and `run_evaluation` must return a string in this format:
   - **First line:** `SUCCESS, <metric>` or `FAILURE, <metric>` or `FAILURE,`  
     - `<metric>` is a numeric value (e.g. `3.3687`, `12.5`). Use `FAILURE,` (nothing after the comma) when there is no meaningful metric (e.g. crash before any run).
   - **Remaining lines (optional):** Details for the agent and logs (e.g. error messages, statistics). The framework uses only the first line when recording the result.

Example first lines: `SUCCESS, 3.3687`, `FAILURE, 1.25`, `FAILURE,`

```python
from tool_lib.base import ToolProvider
from langchain_core.tools import tool, BaseTool

class EvalTool(ToolProvider):
    def __init__(self, eval_timeout: int = 120):
        self._eval_timeout = eval_timeout
        self._workspace = None
        self.evaluate = tool(self._evaluate)

    def run_evaluation(self, filename: str) -> str:
        """Evaluate the given file and return 'SUCCESS, <metric>' or 'FAILURE,'."""
        # Run your evaluation logic on the workspace file `filename`
        # Return first line "SUCCESS, <metric>" or "FAILURE," + optional lines
        return "SUCCESS, 3.14\nOptional details..."

    def _evaluate(self) -> str:
        """Evaluate the algorithm. Docstring becomes tool description for the agent."""
        return self.run_evaluation("draft.py")

    # --- ToolProvider interface ---

    def get_tools(self) -> list[BaseTool]:
        return [self.evaluate]

    def set_workspace(self, workspace):
        self._workspace = workspace
```

**`tool_factory.py`** (optional) — Provides additional tools (e.g. Sionna documentation search).

The class must define a `TOOL_TYPES` class attribute listing the `ToolProvider` types it uses. The framework calls `build()` on each type before spawning workers, allowing expensive one-time setup (e.g. building a vector-store index) to run once in the orchestrator process.

```python
from tool_lib.base import ToolProvider
from config import ToolsConfig

class ToolFactory(ToolProvider):
    TOOL_TYPES = [...]  # List of ToolProvider types used by this factory

    def __init__(self, tools_config: ToolsConfig):
        # Initialize tools using tools_config
        pass

    # --- ToolProvider interface ---

    def get_tools(self):
        return [...]

    def set_workspace(self, workspace):
        pass
```

### 4. Build the container and run

Build the Docker image, then launch the task:

```bash
python launch.py tasks/my_task
```

## Stopping

Press **Ctrl+C** to stop the agents gracefully. The leaderboard is saved after each candidate completes, so progress is preserved.

## Tool-Specific Configuration

Configure these only if your task uses the corresponding tools via `tool_factory.py`.

### Sionna Documentation (RAG-based documentation search)

The `SionnaDoc` tool indexes Sionna documentation for semantic search. It requires an embedding model and, optionally, a cross-encoder reranker. Indexing is performed once and cached to disk.

Configure the tool through `tools_config.sionna_doc_config` in `config.json`:

```json
{
    "tools_config": {
        "sionna_doc_config": {
            "cache_dir_path": "api_doc_cache",
            "embedding_model": "<embedding-model-name>",
            "embedding_base_url": "<embedding-server-url>",
            "reranker_model": "<reranker-model-name>",
            "reranker_base_url": "<reranker-server-url>",
            "retrieve_k": 12,
            "rerank_top_n": 4,
            "summarize_llm": {
                "model": "<summarization-model-name>",
                "base_url": "<summarization-api-url>",
                "temperature": 0.0
            }
        }
    }
}
```

| Parameter | Description |
|-----------|-------------|
| `cache_dir_path` | Directory for the FAISS index cache |
| `embedding_model` | Embedding model name (served via any OpenAI-compatible endpoint) |
| `embedding_base_url` | Base URL of the embedding server (e.g. TEI, Ollama `/v1`, vLLM) |
| `reranker_model` | Cross-encoder model for reranking (optional; leave empty to skip) |
| `reranker_base_url` | Base URL of the reranker server |
| `retrieve_k` | Number of documents to retrieve before reranking |
| `rerank_top_n` | Number of documents to return after reranking |
| `summarize_llm` | Optional LLM config for summarizing tutorials before indexing (omit or set to `{}` to skip) |
| `summarize_llm.model` | LLM model name for summarization |
| `summarize_llm.base_url` | API base URL for the summarization LLM |
| `summarize_llm.temperature` | Sampling temperature for summarization (default: `0.0`) |

The embedding and reranker endpoints must speak the OpenAI-compatible protocol (`/v1/embeddings` and `/v1/rerank`). You can serve them with [TEI](https://github.com/huggingface/text-embeddings-inference), [Ollama](https://ollama.com/), [vLLM](https://vllm.ai/), or any compatible server.

## How to Cite

If you use this software, please cite it as:


```bibtex
@software{the-ai-telco-engineer,
  title  = {The AI Telco Engineer},
  author = {{Aït Aoudia}, Fayçal and Hoydis, Jakob and Cammerer, Sebastian and Maggi, Lorenzo and Marti, Gian and Keller, Alexander},
  note   = {https://github.com/NVlabs/the-ai-telco-engineer},
  year   = {2026}
}
```
