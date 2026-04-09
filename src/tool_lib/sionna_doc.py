# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from pathlib import Path
import hashlib
import importlib
import inspect
import json
import logging
import os
import pickle
import pkgutil
import re
from concurrent.futures import ThreadPoolExecutor
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np

os.environ.setdefault("USER_AGENT", "api-doc-tool/1.0")

import faiss as faiss_lib
from langchain_core.tools import tool, BaseTool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.agents import create_agent
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from langchain_core.documents import Document


class HttpEmbeddings:
    """Embedding client using TEI's native ``/embed`` endpoint.

    Also works with any server that accepts
    ``POST /embed {"inputs": [...]}`` and returns ``[[float, ...], ...]``.
    """

    BATCH_SIZE = 32

    def __init__(self, model: str, base_url: str):
        self._model = model
        self._url = base_url.rstrip("/") + "/embed"

    def _post(self, inputs: list[str]) -> list[list[float]]:
        payload: dict = {"inputs": inputs}
        req = Request(self._url, data=json.dumps(payload).encode(),
                      headers={"Content-Type": "application/json"})
        try:
            with urlopen(req, timeout=120) as resp:
                return json.loads(resp.read())
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Embedding request failed ({exc.code}): {body}"
            ) from exc

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), self.BATCH_SIZE):
            batch = texts[i : i + self.BATCH_SIZE]
            all_embeddings.extend(self._post(batch))
        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        return self._post([text])[0]


class HttpReranker:
    """Reranking client using the TEI ``/rerank`` endpoint."""

    def __init__(self, model: str, base_url: str):
        self._model = model
        self._url = base_url.rstrip("/") + "/rerank"

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        query = pairs[0][0]
        texts = [p[1] for p in pairs]
        payload = {"query": query, "texts": texts}
        req = Request(self._url, data=json.dumps(payload).encode(),
                      headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=30) as resp:
            results = json.loads(resp.read())
        score_map = {r["index"]: r["score"] for r in results}
        return [score_map[i] for i in range(len(texts))]


def _fetch_urls(urls: list[str], max_workers: int = 8) -> list[Document]:
    """Fetch *urls* in parallel and return one Document per URL."""
    user_agent = os.environ.get("USER_AGENT", "api-doc-tool/1.0")

    def _fetch_one(url: str) -> Document:
        req = Request(url, headers={"User-Agent": user_agent})
        with urlopen(req, timeout=30) as resp:
            html = resp.read().decode("utf-8", errors="replace")
        return Document(page_content=html, metadata={"source": url})

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(_fetch_one, urls))


class FaissVectorStore:
    """FAISS vector store backed by langchain-compatible embeddings.
    """

    def __init__(self, embeddings, index, docstore,
                 index_to_id: dict[int, str]):
        self._embeddings = embeddings
        self._index = index
        self._index_to_id = index_to_id

        if isinstance(docstore, dict):
            self._docstore = docstore
        else:
            # Handle legacy caches built with langchain-community's
            # InMemoryDocstore (has a .store dict attribute).
            self._docstore = getattr(docstore, "store",
                                     getattr(docstore, "_dict", {}))

    @classmethod
    def from_documents(cls, documents: list[Document], embeddings):
        texts = [doc.page_content for doc in documents]
        vectors = np.array(embeddings.embed_documents(texts), dtype=np.float32)

        index = faiss_lib.IndexFlatL2(vectors.shape[1])
        index.add(vectors)

        docstore: dict[str, Document] = {}
        index_to_id: dict[int, str] = {}
        for i, doc in enumerate(documents):
            doc_id = str(i)
            docstore[doc_id] = doc
            index_to_id[i] = doc_id

        return cls(embeddings, index, docstore, index_to_id)

    def save_local(self, path: str) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        faiss_lib.write_index(self._index, str(p / "index.faiss"))
        with open(p / "index.pkl", "wb") as f:
            pickle.dump((self._docstore, self._index_to_id), f)

    def similarity_search(self, query: str, k: int = 4) -> list[Document]:
        vector = np.array([self._embeddings.embed_query(query)], dtype=np.float32)
        _, indices = self._index.search(vector, k)
        results: list[Document] = []
        for idx in indices[0]:
            if idx == -1:
                continue
            doc_id = self._index_to_id.get(int(idx))
            if doc_id is not None and doc_id in self._docstore:
                results.append(self._docstore[doc_id])
        return results


import printer
from .workspace import Workspace
from .base import ToolProvider


class SionnaDoc(ToolProvider):
    """
    A tool provider for Sionna documentation search, help, and listing.

    Call :meth:`build` once (in the manager process) before spawning workers
    to ensure the FAISS index is built on disk.  Each worker then calls
    ``__init__`` which memory-maps the index so that physical pages are
    shared across processes.
    """

    _VALID_SYMBOL_RE = re.compile(r"^[a-zA-Z_][\w.]*$")

    TUTORIAL_DOC_URLS = [
        # RT
        "https://nvlabs.github.io/sionna/rt/tutorials/Introduction.html",
        "https://nvlabs.github.io/sionna/rt/tutorials/Diffraction.html",
        "https://nvlabs.github.io/sionna/rt/tutorials/Mobility.html",
        "https://nvlabs.github.io/sionna/rt/tutorials/Radio-Maps.html",
        "https://nvlabs.github.io/sionna/rt/tutorials/Scattering.html",
        "https://nvlabs.github.io/sionna/rt/tutorials/Scene-Edit.html",
        # PHY
        "https://nvlabs.github.io/sionna/phy/tutorials/notebooks/Hello_World.html",
        "https://nvlabs.github.io/sionna/phy/tutorials/notebooks/Sionna_tutorial_part1.html",
        "https://nvlabs.github.io/sionna/phy/tutorials/notebooks/Sionna_tutorial_part2.html",
        "https://nvlabs.github.io/sionna/phy/tutorials/notebooks/Sionna_tutorial_part3.html",
        "https://nvlabs.github.io/sionna/phy/tutorials/notebooks/Sionna_tutorial_part4.html",
        "https://nvlabs.github.io/sionna/phy/tutorials/notebooks/Simple_MIMO_Simulation.html",
        "https://nvlabs.github.io/sionna/phy/tutorials/notebooks/Pulse_Shaping_Basics.html",
        "https://nvlabs.github.io/sionna/phy/tutorials/notebooks/Optical_Lumped_Amplification_Channel.html",
        "https://nvlabs.github.io/sionna/phy/tutorials/notebooks/5G_Channel_Coding_Polar_vs_LDPC_Codes.html",
        "https://nvlabs.github.io/sionna/phy/tutorials/notebooks/5G_NR_PUSCH.html",
        "https://nvlabs.github.io/sionna/phy/tutorials/notebooks/Bit_Interleaved_Coded_Modulation.html",
        "https://nvlabs.github.io/sionna/phy/tutorials/notebooks/MIMO_OFDM_Transmissions_over_CDL.html",
        "https://nvlabs.github.io/sionna/phy/tutorials/notebooks/Neural_Receiver.html",
        "https://nvlabs.github.io/sionna/phy/tutorials/notebooks/Realistic_Multiuser_MIMO_Simulations.html",
        "https://nvlabs.github.io/sionna/phy/tutorials/notebooks/OFDM_MIMO_Detection.html",
        "https://nvlabs.github.io/sionna/phy/tutorials/notebooks/Introduction_to_Iterative_Detection_and_Decoding.html",
        "https://nvlabs.github.io/sionna/phy/tutorials/notebooks/Autoencoder.html",
        "https://nvlabs.github.io/sionna/phy/tutorials/notebooks/Weighted_BP_Algorithm.html",
        "https://nvlabs.github.io/sionna/phy/tutorials/notebooks/Superimposed_Pilots.html",
        "https://nvlabs.github.io/sionna/phy/tutorials/notebooks/CIR_Dataset.html",
        "https://nvlabs.github.io/sionna/phy/tutorials/notebooks/Link_Level_Simulations_with_RT.html",
        # SYS
        "https://nvlabs.github.io/sionna/sys/tutorials/notebooks/PHY_Abstraction.html",
        "https://nvlabs.github.io/sionna/sys/tutorials/notebooks/LinkAdaptation.html",
        "https://nvlabs.github.io/sionna/sys/tutorials/notebooks/Scheduling.html",
        "https://nvlabs.github.io/sionna/sys/tutorials/notebooks/HexagonalGrid.html",
        "https://nvlabs.github.io/sionna/sys/tutorials/notebooks/Power_Control.html",
        "https://nvlabs.github.io/sionna/sys/tutorials/notebooks/SYS_Meets_RT.html",
        "https://nvlabs.github.io/sionna/sys/tutorials/notebooks/End-to-End_Example.html"
    ]

    SEARCH_EXAMPLES = [
        'sionna_search("LDPC encoder and decoder")',
        'sionna_search("OFDM modulation and demodulation")',
        'sionna_search("MIMO detection algorithms")',
        'sionna_search("5G NR channel coding")',
        'sionna_search("how to load a scene")',
        'sionna_search("compute radio map")',
        'sionna_search("antenna array configuration")',
        'sionna_search("ray tracing channel impulse response")',
    ]

    HELP_EXAMPLES = [
        'sionna_help("sionna.phy.fec.ldpc.LDPCEncoder")',
        'sionna_help("sionna.phy.ofdm.ResourceGrid")',
        'sionna_help("sionna.rt.Scene")',
        'sionna_help("sionna.rt.Transmitter")',
        'sionna_help("sionna.sys.PHYAbstraction")',
    ]

    LIST_EXAMPLES = [
        'sionna_list("sionna.phy")',
        'sionna_list("sionna.phy.fec")',
        'sionna_list("sionna.rt")',
        'sionna_list("sionna.sys")',
    ]

    # ── paths helpers ────────────────────────────────────────────────

    @staticmethod
    def _cache_path(cache_dir: str) -> Path:
        return Path(cache_dir) / "sionna"

    @classmethod
    def _vectorstore_path(cls, cache_dir: str) -> Path:
        return cls._cache_path(cache_dir) / "vectorstore"

    @classmethod
    def _markdown_path(cls, cache_dir: str) -> Path:
        return cls._cache_path(cache_dir) / "markdown"

    @classmethod
    def build(cls, tools_config) -> None:
        """Build the FAISS vectorstore on disk if it does not already exist.

        This is called once in the manager process before workers are
        spawned.  Worker processes then memory-map the same index file,
        sharing physical pages via the OS page cache.

        Args:
            tools_config: A :class:`~config.ToolsConfig` instance or plain
                dict.  SionnaDoc-specific settings are read from the
                ``sionna_doc_config`` nested key.
        """
        cfg = tools_config.get("sionna_doc_config", {})
        if not cfg:
            printer.log("SionnaDoc.build: No sionna_doc_config found — skipping.")
            return

        embedding_model = cfg.get("embedding_model", "")
        embedding_base_url = cfg.get("embedding_base_url", "")
        if not embedding_model or not embedding_base_url:
            raise ValueError(
                "SionnaDoc.build: 'embedding_model' and 'embedding_base_url' must be "
                "set in sionna_doc_config in your task's config.json."
            )

        cache_dir = cfg.get("cache_dir_path", "api_doc_cache")
        vectorstore_path = cls._vectorstore_path(cache_dir)

        if vectorstore_path.exists():
            printer.log(f"SionnaDoc.build: Vectorstore already cached at {vectorstore_path}")
            return

        printer.log("SionnaDoc.build: No cache found. Building vectorstore...")

        cache_path = cls._cache_path(cache_dir)
        markdown_path = cls._markdown_path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        markdown_path.mkdir(parents=True, exist_ok=True)

        embeddings = HttpEmbeddings(embedding_model, embedding_base_url)

        # Optional LLM for tutorial summarisation
        summarize_agent = cls._build_summarize_agent(cfg)

        # ── API docs: extract docstrings from installed sionna modules ──
        printer.log("SionnaDoc.build: Extracting API documentation from sionna modules...")
        api_documents = cls._extract_api_docs()
        printer.log(f"SionnaDoc.build: Extracted {len(api_documents)} API doc entries")

        # ── Tutorials: fetch from web, convert to markdown, summarize ──
        printer.log("SionnaDoc.build: Fetching tutorials from the web...")
        tutorial_html_docs = _fetch_urls(cls.TUTORIAL_DOC_URLS)
        printer.log(f"SionnaDoc.build: Fetched {len(tutorial_html_docs)} tutorials")

        tutorial_documents = []
        for doc in tutorial_html_docs:
            soup = BeautifulSoup(doc.page_content, 'html.parser')

            for tag in soup(["script", "style", "nav", "footer", "aside", "meta", "noscript",
                             "canvas", "video", "audio"]):
                tag.decompose()

            markdown_content = md(
                str(soup),
                heading_style="ATX",
                newlines_after_heading=1
            )
            cleaned_content = "\n".join(
                line for line in markdown_content.splitlines() if line.strip()
            )
            cleaned_content = re.sub(r'!\[.*?\]\(.*?\)', '', cleaned_content)

            md_hash = hashlib.md5(doc.metadata['source'].encode()).hexdigest()
            target_path = markdown_path / f"{md_hash}.md"
            with open(target_path, "w", encoding="utf-8") as f:
                f.write(cleaned_content)
            printer.log(f"SionnaDoc.build: Saved tutorial markdown to {target_path}")

            summary_path = target_path.with_suffix(".summary.md")
            if summary_path.exists():
                with open(summary_path, "r", encoding="utf-8") as f:
                    index_content = f.read()
                printer.log(f"SionnaDoc.build: Using cached summary from {summary_path}")
            else:
                index_content = cleaned_content

            if summarize_agent is not None:
                printer.log(f"SionnaDoc.build: Summarizing {doc.metadata['source']}")
                try:
                    summary = summarize_agent.invoke({"messages":
                    [("user",
                    f"""*Role:* You are a RAG Knowledge Engineer optimizing technical documentation for vector database indexing.
                    *Task:* Process the provided Markdown notebook into a clean, semantically rich format suitable for generating high-quality embeddings.
                    ...
                    *Input Notebook:*
                    
                    {cleaned_content}""")]})
                    summary = summary["messages"][-1].content
                    summary_path = target_path.with_suffix(".summary.md")
                    with open(summary_path, "w", encoding="utf-8") as f:
                        f.write(summary)
                    index_content = summary
                except Exception as e:
                    printer.log(f"SionnaDoc.build: Failed to summarize: {e}")

            tutorial_documents.append(Document(
                page_content=index_content,
                metadata={
                    "source": doc.metadata['source'],
                    "doc_type": "tutorial",
                    "original_path": str(target_path)
                }
            ))

        # ── Split and build FAISS index ──
        api_text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            separators=[
                "\n:param ",
                "\n\n",
                "\n",
                " ",
                ""
            ]
        )
        tutorial_text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            separators=[
                "\n# ",
                "\n## ",
                "\n### ",
                "\n```python\n",
                "\n```\n",
                "\n```",
                "\n\n",
                "\n",
                " ",
                ""
            ]
        )

        MIN_CHUNK_LENGTH = 50
        api_chunks = api_text_splitter.split_documents(api_documents)
        tutorial_chunks = tutorial_text_splitter.split_documents(tutorial_documents)
        all_chunks = [c for c in api_chunks + tutorial_chunks if len(c.page_content.strip()) >= MIN_CHUNK_LENGTH]
        printer.log(f"SionnaDoc.build: {len(api_chunks)} API chunks + {len(tutorial_chunks)} tutorial chunks, {len(all_chunks)} after filtering (min {MIN_CHUNK_LENGTH} chars)")

        printer.log("SionnaDoc.build: Building the vectorstore...")
        vectorstore = FaissVectorStore.from_documents(all_chunks, embeddings)
        printer.log("SionnaDoc.build: Vectorstore built")

        printer.log(f"SionnaDoc.build: Saving vectorstore to {vectorstore_path}")
        vectorstore.save_local(str(vectorstore_path))
        printer.log("SionnaDoc.build: Done")

    @classmethod
    def _build_summarize_agent(cls, sionna_doc_cfg: dict):
        """Create the optional summarisation agent from sionna_doc_config."""
        llm_cfg = sionna_doc_cfg.get("summarize_llm")
        if not llm_cfg:
            return None
        try:
            from dataclasses import asdict
            from langchain_openai import ChatOpenAI
            from config import LLMConfig
            config = LLMConfig(
                api_key=os.environ.get("MODEL_API_KEY", ""),
                base_url=llm_cfg.get("base_url", ""),
                model=llm_cfg.get("model", ""),
                temperature=llm_cfg.get("temperature", 0.0),
                top_p=llm_cfg.get("top_p", 0.95),
                model_kwargs=llm_cfg.get("model_kwargs", {}),
            )
            llm = ChatOpenAI(**asdict(config))
            return create_agent(llm, [])
        except Exception:
            return None

    def __init__(self,
                 embedding_model: str,
                 embedding_base_url: str,
                 reranker_model: str,
                 reranker_base_url: str,
                 retrieve_k: int = 12,
                 rerank_top_n: int = 4,
                 cache_dir: str = "api_doc_cache"):
        """
        Load a pre-built FAISS vectorstore using memory-mapping.

        The vectorstore must have been created by a prior call to
        :meth:`build`.  The FAISS index is memory-mapped so that
        multiple worker processes share the same physical pages.

        Args:
            embedding_model: Embedding model name served by the endpoint.
            embedding_base_url: Base URL of the embedding service
                (must expose ``/v1/embeddings``).
            reranker_model: Reranker model name served by the endpoint.
            reranker_base_url: Base URL of the reranking service
                (must expose ``/rerank``).
            retrieve_k: Number of documents to retrieve from the vector store.
            rerank_top_n: Number of top documents to keep after reranking.
            cache_dir: Directory path where the cached vectorstore lives.

        Raises:
            ValueError: If any of the required model/URL arguments are missing.
        """
        missing = []
        if not embedding_model:
            missing.append("embedding_model")
        if not embedding_base_url:
            missing.append("embedding_base_url")
        if not reranker_model:
            missing.append("reranker_model")
        if not reranker_base_url:
            missing.append("reranker_base_url")
        if missing:
            raise ValueError(
                f"SionnaDoc: required parameters not set: {', '.join(missing)}. "
                "Configure them in sionna_doc_config in your task's config.json."
            )
        self._retrieve_k = retrieve_k
        self._rerank_top_n = rerank_top_n
        self._reranker = HttpReranker(reranker_model, reranker_base_url)

        vectorstore_path = self._vectorstore_path(cache_dir)
        self._vectorstore = None

        try:
            if not vectorstore_path.exists():
                raise FileNotFoundError(
                    f"Vectorstore not found at {vectorstore_path}. "
                    "Ensure SionnaDoc.build() has been called before instantiation."
                )

            embeddings = HttpEmbeddings(embedding_model, embedding_base_url)

            index_faiss_path = str(vectorstore_path / "index.faiss")
            index_pkl_path = str(vectorstore_path / "index.pkl")

            printer.log(f"SionnaDoc: Memory-mapping FAISS index from {index_faiss_path}")
            raw_index = faiss_lib.read_index(index_faiss_path, faiss_lib.IO_FLAG_MMAP)

            with open(index_pkl_path, "rb") as f:
                docstore, index_to_id = pickle.load(f)

            self._vectorstore = FaissVectorStore(
                embeddings=embeddings,
                index=raw_index,
                docstore=docstore,
                index_to_id=index_to_id,
            )
            printer.log("SionnaDoc: Initialized (mmap)")
        except Exception as e:
            printer.log(f"SionnaDoc: Failed to load vectorstore: {e}. "
                        "Documentation search will be unavailable.")
            self._vectorstore = None

        self._workspace = None
        self._create_tools()

    @staticmethod
    def _extract_api_docs() -> list[Document]:
        """Extract docstrings from all sionna submodules."""
        documents = []

        try:
            import sionna
        except ImportError:
            printer.log("SionnaDoc: Failed to import sionna. No API docs will be extracted.")
            return documents

        modules_to_process = [("sionna", sionna)]
        for _importer, modname, _ispkg in pkgutil.walk_packages(
            sionna.__path__, prefix="sionna."
        ):
            try:
                mod = importlib.import_module(modname)
                modules_to_process.append((modname, mod))
            except Exception:
                continue

        for module_name, module in modules_to_process:
            for attr_name, obj in inspect.getmembers(module):
                if attr_name.startswith("_"):
                    continue

                if not (inspect.isclass(obj) or inspect.isfunction(obj)):
                    continue
                if getattr(obj, "__module__", None) != module_name:
                    continue

                docstring = inspect.getdoc(obj)
                if not docstring:
                    continue

                full_name = f"{module_name}.{attr_name}"
                try:
                    sig = str(inspect.signature(obj))
                except (ValueError, TypeError):
                    sig = ""
                sig_block = f"\n\n```python\n{attr_name}{sig}\n```" if sig else ""
                content = f"## {full_name}{sig_block}\n\n{docstring}"

                documents.append(Document(
                    page_content=content,
                    metadata={"source": full_name, "doc_type": "api"}
                ))

        return documents

    def set_workspace(self, workspace: Workspace):
        """Set the workspace for running help/list commands.

        Args:
            workspace: The Workspace instance to use for executing Python code.
        """
        self._workspace = workspace

    SEARCH_DESCRIPTION = """Search the sionna documentation for relevant functions, modules, and examples.

Use this tool to discover which sionna functions or modules are relevant to your task.
After finding relevant symbols, use sionna_help() to get full documentation.

Args:
    query: What you're looking for in the sionna documentation.

Returns:
    Relevant excerpts from the sionna documentation with source info.

Examples:
    """ + "\n    ".join(SEARCH_EXAMPLES)

    HELP_DESCRIPTION = """Gets the documentation and signature for a sionna function, class, or module.

Use this tool to look up how to use sionna functions and classes.

Args:
    symbol: The sionna symbol to look up.
            Can also be just 'sionna' for module overview.

Returns:
    The docstring and signature of the symbol, or an error message if not found.

Examples:
    """ + "\n    ".join(HELP_EXAMPLES)

    LIST_DESCRIPTION = """Lists available functions and classes in a sionna module.

Use this tool to explore what's available in sionna modules.

Args:
    module: The sionna module to list.

Returns:
    List of available symbols in the module.

Examples:
    """ + "\n    ".join(LIST_EXAMPLES)

    def _create_tools(self):
        """Create tools from bound methods and set their descriptions."""
        self.search = tool(self._search)
        self.help = tool(self._help)
        self.list = tool(self._list)

        self.search.description = self.SEARCH_DESCRIPTION
        self.help.description = self.HELP_DESCRIPTION
        self.list.description = self.LIST_DESCRIPTION

    def get_tools(self) -> list[BaseTool]:
        """Return all tools provided by this instance.

        Implements the ToolProvider interface.
        """
        return [self.search, self.help, self.list]

    def _search(self, query: str) -> str:
        """Search the documentation for relevant functions, modules, and examples."""
        if self._vectorstore is None:
            return ("Error: Documentation search is unavailable because the vectorstore "
                    "failed to initialize (the embedding model may be unreachable). "
                    "Use sionna_help() or sionna_list() instead.")

        try:
            results = self._vectorstore.similarity_search(query, k=self._retrieve_k)
        except Exception as e:
            return (f"Error: Documentation search failed ({e}). "
                    "The embedding model may be unreachable. "
                    "Use sionna_help() or sionna_list() instead.")

        if not results:
            return "No relevant documentation found. Try a different query or use list() to explore available modules."

        pairs = [(query, doc.page_content) for doc in results]
        scores = self._reranker.predict(pairs)

        scored_results = list(zip(results, scores))
        scored_results.sort(key=lambda x: x[1], reverse=True)
        reranked_results = [doc for doc, _ in scored_results[:self._rerank_top_n]]

        if not reranked_results:
            return "No relevant documentation found. Try a different query or use list() to explore available modules."

        formatted_results = []

        for i, doc in enumerate(reranked_results, 1):
            source = doc.metadata.get("source", "sionna docs")
            source_short = source.split("/")[-1] if "/" in source else source
            doc_type = doc.metadata.get("doc_type", "api")
            content = doc.page_content.strip()

            formatted_results.append(
                f"[{i}] From: {source_short} ({doc_type})\n"
                f"{content}"
            )

        output = "\n\n---\n\n".join(formatted_results)
        output += "\n\n---\nTip 1: Use help('symbol.name') to get the full docstring for any function or class mentioned above."
        output += "\n\n---\nTip 2: Use list('module.name') to list the functions and classes in a module."

        return output

    def _help(self, symbol: str) -> str:
        """Gets the documentation and signature for a function, class, or module."""
        if self._workspace is None:
            return "Error: Workspace not set. Cannot run help command."
        if not self._VALID_SYMBOL_RE.match(symbol):
            return f"Error: Invalid symbol name '{symbol}'. Must be a dotted Python identifier (e.g. 'sionna.phy.ofdm.ResourceGrid')."

        python_code = f'''
import sionna, inspect
try:
    obj = eval("{symbol}")
    sig = ""
    try:
        sig = str(inspect.signature(obj))
    except (ValueError, TypeError):
        pass
    doc = inspect.getdoc(obj) or "No documentation found."
    if sig:
        print(f"{symbol}{{sig}}\\n")
    print(doc)
except Exception as e:
    print(f"Error: {{e}}")
    '''
        return self._workspace._run_python_code(python_code)

    def _list(self, module: str) -> str:
        """Lists available functions and classes in a module."""
        if self._workspace is None:
            return "Error: Workspace not set. Cannot run list command."
        if not self._VALID_SYMBOL_RE.match(module):
            return f"Error: Invalid module name '{module}'. Must be a dotted Python identifier (e.g. 'sionna.phy')."

        python_code = f'''
import sionna
try:
    mod = eval("{module}")
    items = [x for x in dir(mod) if not x.startswith("_")]
    print("\\n".join(items))
except Exception as e:
    print(f"Error: {{e}}")
    '''
        return self._workspace._run_python_code(python_code)
