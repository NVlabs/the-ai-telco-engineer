# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Leaderboard module - Manages candidate solutions organized by idea/approach.

Candidates are grouped into idea clusters assigned by the manager LLM.
Each cluster corresponds to one distinct algorithmic approach explored per generation.
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class Candidate:
    """A candidate solution with code and cluster information."""
    workspace_id: str
    metric: float
    generation: int
    code: str = ""  # The main solution code
    cluster: int = 0  # Cluster number (integer id)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    success: bool = True
    error: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Candidate":
        """Create a Candidate from a dictionary."""
        required_fields = {"workspace_id", "metric", "generation"}
        missing = required_fields - data.keys()
        if missing:
            raise ValueError(f"Missing required fields for Candidate: {missing}")
        d = dict(data)
        if "cluster" in d and isinstance(d["cluster"], str):
            d["cluster"] = int(d["cluster"])
        return cls(**d)


@dataclass
class ClusteredLeaderboard:
    """
    A leaderboard that organizes candidates by cluster/approach.

    Clusters are assigned by ideas (from the manager)
    """
    clusters: dict[int, list[Candidate]] = field(default_factory=dict)
    cluster_best_metrics: dict[int, float] = field(default_factory=dict)
    cluster_descriptions: dict[int, str] = field(default_factory=dict)  # Idea description per cluster
    query: str = ""
    higher_is_better: bool = False  # If True, higher metric values are better; if False, lower is better
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    _next_cluster_id: int = 0  # Monotonically increasing; persisted in to_dict/from_dict

    def get_next_cluster_ids(self, n: int) -> list[int]:
        """
        Reserve the next n cluster ids and return them.

        Call this when generating ideas so cluster ids always increase across generations.
        """
        ids = [self._next_cluster_id + i for i in range(n)]
        self._next_cluster_id += n
        self.last_updated = datetime.now().isoformat()
        return ids

    def add_cluster(self, cluster_id: int, description: str) -> None:
        """
        Register a cluster with the given id and description.

        Call this when generating ideas (e.g. from the agent manager) to create
        clusters before adding candidates. Creates an empty candidate list and
        stores the idea description. If the cluster already exists (e.g. from a
        previous generation), only the description is updated.

        Args:
            cluster_id: Integer cluster id.
            description: Idea/approach description for this cluster.
        """
        if cluster_id not in self.clusters:
            self.clusters[cluster_id] = []
            worst = float('-inf') if self.higher_is_better else float('inf')
            self.cluster_best_metrics[cluster_id] = worst
        self.cluster_descriptions[cluster_id] = description
        self.last_updated = datetime.now().isoformat()

    def add_candidate(self, candidate: Candidate) -> None:
        """
        Add a candidate to the cluster given by candidate.cluster.

        The candidate must already have its cluster set. The cluster should
        normally exist (created via add_cluster when ideas were generated).
        """
        cluster_id = candidate.cluster

        if cluster_id not in self.clusters:
            self.clusters[cluster_id] = []

        self.clusters[cluster_id].append(candidate)
        # Sort within cluster by metric (best first), failures at end
        # For higher_is_better=True, negate metric so higher values sort first
        self.clusters[cluster_id].sort(
            key=lambda c: (not c.success, -c.metric if self.higher_is_better else c.metric)
        )
        # Update best metric only if there are successful candidates
        successful_in_cluster = [c.metric for c in self.clusters[cluster_id] if c.success]
        if successful_in_cluster:
            if self.higher_is_better:
                self.cluster_best_metrics[cluster_id] = max(successful_in_cluster)
            else:
                self.cluster_best_metrics[cluster_id] = min(successful_in_cluster)
        else:
            # Use worst possible value for failed clusters
            self.cluster_best_metrics[cluster_id] = float('-inf') if self.higher_is_better else float('inf')
        self.last_updated = datetime.now().isoformat()

    def get_all_candidates(self) -> list[Candidate]:
        """Get all candidates across all clusters."""
        all_candidates = []
        for candidates in self.clusters.values():
            all_candidates.extend(candidates)
        return all_candidates

    def get_successful_candidates(self) -> list[Candidate]:
        """Get all successful candidates across all clusters."""
        return [c for c in self.get_all_candidates() if c.success]

    def get_current_generation_candidates(self, generation: int) -> list[Candidate]:
        """Get all candidates from a specific generation."""
        return [c for c in self.get_all_candidates() if c.generation == generation]

    def get_cluster_summary(self) -> dict[int, dict]:
        """Get a summary of each cluster."""
        summary: dict[int, dict] = {}
        for cluster, candidates in self.clusters.items():
            successful = [c for c in candidates if c.success]
            if successful:
                if self.higher_is_better:
                    best_metric = max(c.metric for c in successful)
                else:
                    best_metric = min(c.metric for c in successful)
            else:
                best_metric = None
            summary[cluster] = {
                "total": len(candidates),
                "successful": len(successful),
                "best_metric": best_metric,
                "best_workspace": successful[0].workspace_id if successful else None
            }
        return summary

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "query": self.query,
            "higher_is_better": self.higher_is_better,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "next_cluster_id": self._next_cluster_id,
            "total_candidates": len(self.get_all_candidates()),
            "successful_candidates": len(self.get_successful_candidates()),
            "num_clusters": len(self.clusters),
            "cluster_descriptions": {str(k): v for k, v in self.cluster_descriptions.items()},
            "clusters": {
                str(cluster): [c.to_dict() for c in candidates]
                for cluster, candidates in self.clusters.items()
            }
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ClusteredLeaderboard":
        """Create a ClusteredLeaderboard from a dictionary."""
        if not isinstance(data, dict):
            raise ValueError(f"Expected dict, got {type(data).__name__}")

        higher_is_better = data.get("higher_is_better", False)

        clusters: dict[int, list[Candidate]] = {}
        if "clusters" in data:
            if not isinstance(data["clusters"], dict):
                raise ValueError("'clusters' field must be a dict")
            for key, candidates_data in data["clusters"].items():
                cluster_id = int(key) if isinstance(key, str) else key
                clusters[cluster_id] = [Candidate.from_dict(c) for c in candidates_data]

        cluster_best_metrics: dict[int, float] = {}
        for cluster_id, candidates in clusters.items():
            successful = [c.metric for c in candidates if c.success]
            if successful:
                if higher_is_better:
                    cluster_best_metrics[cluster_id] = max(successful)
                else:
                    cluster_best_metrics[cluster_id] = min(successful)
            else:
                cluster_best_metrics[cluster_id] = float('-inf') if higher_is_better else float('inf')

        raw_descriptions = data.get("cluster_descriptions", {})
        cluster_descriptions = {
            int(k) if isinstance(k, str) else k: v
            for k, v in raw_descriptions.items()
        }

        next_cluster_id = data.get("next_cluster_id")
        if next_cluster_id is None:
            next_cluster_id = max(clusters.keys(), default=-1) + 1

        lb = cls(
            clusters=clusters,
            cluster_best_metrics=cluster_best_metrics,
            cluster_descriptions=cluster_descriptions,
            query=data.get("query", ""),
            higher_is_better=higher_is_better,
            created_at=data.get("created_at", datetime.now().isoformat()),
            last_updated=data.get("last_updated", datetime.now().isoformat())
        )
        lb._next_cluster_id = next_cluster_id
        return lb

    def save(self, path: Path):
        """Save the leaderboard to a JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "ClusteredLeaderboard":
        """Load a leaderboard from a JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)
