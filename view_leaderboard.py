# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Leaderboard Viewer - A web application to visualize the agent optimization leaderboard.

Run with: python3 view_leaderboard.py [--workspace PATH]
Then open: http://localhost:8000

Options:
    --workspace, -w PATH    Path to the workspace folder containing leaderboard.json
                            (default: ./workspaces)
    --port, -p PORT         Port to run the server on (default: 8000)

No external dependencies required - uses Python's built-in http.server.
"""

import argparse
import json
import os
import math
from http.server import HTTPServer, SimpleHTTPRequestHandler


# Global path to leaderboard file (set by main())
LEADERBOARD_PATH = None


def load_leaderboard():
    """Load the leaderboard JSON file."""
    with open(LEADERBOARD_PATH, "r") as f:
        return json.load(f)


def sanitize_for_json(obj):
    """Recursively convert float('inf') and float('-inf') to strings for JSON."""
    if isinstance(obj, float):
        if math.isinf(obj):
            return "Infinity" if obj > 0 else "-Infinity"
        if math.isnan(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    return obj


def build_api_data(data):
    """Build all computed data for the API response."""
    higher_is_better = data.get("higher_is_better", False)

    # Build candidates dict with cluster info
    candidates = {}
    for cluster_name, cluster_candidates in data.get("clusters", {}).items():
        for candidate in cluster_candidates:
            candidates[candidate["workspace_id"]] = {
                **candidate,
                "cluster": cluster_name
            }

    # Generation stats
    from collections import defaultdict
    gen_data = defaultdict(lambda: {"total": 0, "successful": 0, "best_metric": None, "candidates": []})
    for cluster_name, cluster_candidates in data.get("clusters", {}).items():
        for c in cluster_candidates:
            gen = c.get("generation", 0)
            gen_data[gen]["total"] += 1
            gen_data[gen]["candidates"].append(c["workspace_id"])
            if c.get("success", False):
                gen_data[gen]["successful"] += 1
                m = c.get("metric")
                if m is not None and not (isinstance(m, float) and math.isinf(m)):
                    current_best = gen_data[gen]["best_metric"]
                    if current_best is None:
                        gen_data[gen]["best_metric"] = m
                    elif higher_is_better and m > current_best:
                        gen_data[gen]["best_metric"] = m
                    elif not higher_is_better and m < current_best:
                        gen_data[gen]["best_metric"] = m

    generation_stats = []
    for gen in sorted(gen_data.keys()):
        stats = gen_data[gen]
        generation_stats.append({
            "generation": gen,
            "total": stats["total"],
            "successful": stats["successful"],
            "failed": stats["total"] - stats["successful"],
            "best_metric": stats["best_metric"]
        })

    # Cluster evolution
    cluster_gen_data = defaultdict(lambda: defaultdict(lambda: {"total": 0, "successful": 0, "best_metric": None}))
    for cluster_name, cluster_candidates in data.get("clusters", {}).items():
        for c in cluster_candidates:
            gen = c.get("generation", 0)
            cluster_gen_data[gen][cluster_name]["total"] += 1
            if c.get("success", False):
                cluster_gen_data[gen][cluster_name]["successful"] += 1
                m = c.get("metric")
                if m is not None and not (isinstance(m, float) and math.isinf(m)):
                    current_best = cluster_gen_data[gen][cluster_name]["best_metric"]
                    if current_best is None:
                        cluster_gen_data[gen][cluster_name]["best_metric"] = m
                    elif higher_is_better and m > current_best:
                        cluster_gen_data[gen][cluster_name]["best_metric"] = m
                    elif not higher_is_better and m < current_best:
                        cluster_gen_data[gen][cluster_name]["best_metric"] = m

    cluster_evolution = []
    for gen in sorted(cluster_gen_data.keys()):
        clusters = []
        for cn, stats in sorted(cluster_gen_data[gen].items()):
            clusters.append({
                "name": cn,
                "total": stats["total"],
                "successful": stats["successful"],
                "best_metric": stats["best_metric"]
            })
        cluster_evolution.append({"generation": gen, "clusters": clusters})

    # Find best metric
    best_metric = None
    for cand in candidates.values():
        if cand.get("success") and cand.get("metric") is not None:
            m = cand["metric"]
            if isinstance(m, float) and math.isinf(m):
                continue
            if best_metric is None:
                best_metric = m
            elif higher_is_better and m > best_metric:
                best_metric = m
            elif not higher_is_better and m < best_metric:
                best_metric = m

    return sanitize_for_json({
        "higher_is_better": higher_is_better,
        "total_candidates": data.get("total_candidates", 0),
        "successful_candidates": data.get("successful_candidates", 0),
        "num_clusters": data.get("num_clusters", 0),
        "last_updated": data.get("last_updated", "N/A"),
        "query": data.get("query", ""),
        "cluster_descriptions": data.get("cluster_descriptions", {}),
        "best_metric": best_metric,
        "candidates": candidates,
        "generation_stats": generation_stats,
        "cluster_evolution": cluster_evolution,
    })


def generate_html():
    """Generate the static HTML shell with client-side rendering."""
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Leaderboard Viewer</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { background-color: #f5f5f5; }
        .card { margin-bottom: 1rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .card-header { font-weight: 600; }
        .success-badge { background-color: #28a745; }
        .failed-badge { background-color: #dc3545; }
        .candidate-item {
            cursor: pointer;
            padding: 8px 12px;
            border-radius: 4px;
            margin: 2px 0;
            transition: background-color 0.2s;
        }
        .candidate-item:hover { background-color: #e9ecef; }
        .candidate-success { border-left: 4px solid #28a745; }
        .candidate-failed { border-left: 4px solid #dc3545; }
        .code-container {
            max-height: 500px;
            overflow-y: auto;
            background-color: #1e1e1e;
            color: #d4d4d4;
            padding: 1rem;
            border-radius: 4px;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 0.85rem;
            white-space: pre-wrap;
        }
        .metric-value { font-weight: bold; font-size: 0.9rem; }
        .metric-good { color: #28a745; }
        .metric-bad { color: #dc3545; }
        .cluster-description-box {
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 4px;
            padding: 0.75rem 1rem;
        }
        .cluster-description-label { font-weight: 600; font-size: 0.85rem; color: #495057; margin-bottom: 0.35rem; }
        .cluster-description-text { font-size: 0.9rem; color: #212529; line-height: 1.4; white-space: pre-wrap; }
        .ideas-by-gen-content { max-height: 75vh; overflow-y: auto; }
        .ideas-by-gen-gen { margin-bottom: 1.5rem; }
        .ideas-by-gen-gen .gen-header { font-weight: 600; font-size: 1.1rem; margin-bottom: 0.5rem; color: #0d6efd; }
        .ideas-by-gen-row { padding: 0.5rem 0.75rem; margin: 0.25rem 0; background: #f8f9fa; border-radius: 4px; border-left: 3px solid #dee2e6; }
        .ideas-by-gen-row .idea-name { font-weight: 600; }
        .ideas-by-gen-row .idea-desc { font-size: 0.9rem; color: #495057; margin-top: 0.25rem; }
        .ideas-by-gen-row .idea-metric { font-weight: bold; margin-left: 0.5rem; }
        .stat-card { text-align: center; padding: 1.5rem; }
        .stat-value { font-size: 2rem; font-weight: bold; color: #0d6efd; }
        .stat-label { color: #6c757d; font-size: 0.9rem; }
        .cluster-header {
            background-color: #e9ecef;
            padding: 0.5rem 1rem;
            border-radius: 4px;
            margin-bottom: 0.5rem;
            cursor: pointer;
        }
        .cluster-header:hover { background-color: #dee2e6; }
        .nav-tabs .nav-link.active { font-weight: 600; }
        #evolutionChart { max-height: 400px; }
        .query-box {
            max-height: 400px;
            overflow-y: auto;
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 4px;
            font-size: 0.9rem;
            white-space: pre-wrap;
        }
        .cluster-name { font-family: monospace; font-size: 0.85rem; }
        .cluster-desc {
            font-size: 0.75rem;
            color: #6c757d;
            margin-top: 4px;
            line-height: 1.3;
            max-height: 2.6em;
            overflow: hidden;
            text-overflow: ellipsis;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
        }
        .collapse:not(.show) { display: none; }
        .sort-btn {
            cursor: pointer;
            border: 1px solid #dee2e6;
            background: #fff;
            border-radius: 4px;
            padding: 2px 10px;
            font-size: 0.8rem;
            margin-left: 8px;
            transition: all 0.15s;
        }
        .sort-btn:hover { background-color: #e9ecef; }
        .sort-btn.active { background-color: #0d6efd; color: #fff; border-color: #0d6efd; }
        .refresh-indicator {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background-color: #28a745;
            margin-right: 6px;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.3; }
            100% { opacity: 1; }
        }
    </style>
</head>
<body>
    <nav class="navbar navbar-dark bg-primary mb-4">
        <div class="container">
            <span class="navbar-brand mb-0 h1">Leaderboard Viewer</span>
            <span class="navbar-text text-light">
                <span class="refresh-indicator"></span>
                Auto-refresh: <span id="refreshCountdown">5</span>s
                &mdash; Last updated: <span id="lastUpdated">loading...</span>
            </span>
        </div>
    </nav>

    <div class="container-fluid">
        <!-- Global Stats -->
        <div class="row mb-4" id="statsRow">
            <div class="col-md-2"><div class="card stat-card"><div class="stat-value" id="statTotal">-</div><div class="stat-label">Total Candidates</div></div></div>
            <div class="col-md-2"><div class="card stat-card"><div class="stat-value text-success" id="statSuccess">-</div><div class="stat-label">Successful</div></div></div>
            <div class="col-md-2"><div class="card stat-card"><div class="stat-value text-danger" id="statFailed">-</div><div class="stat-label">Failed</div></div></div>
            <div class="col-md-2"><div class="card stat-card"><div class="stat-value" id="statClusters">-</div><div class="stat-label">Ideas</div></div></div>
            <div class="col-md-2"><div class="card stat-card"><div class="stat-value" id="statGens">-</div><div class="stat-label">Generations</div></div></div>
            <div class="col-md-2"><div class="card stat-card"><div class="stat-value text-primary" id="statBest">-</div><div class="stat-label">Best Metric</div></div></div>
        </div>

        <!-- Tabs -->
        <ul class="nav nav-tabs mb-3" id="mainTabs" role="tablist">
            <li class="nav-item"><button class="nav-link active" data-tab="clusters">Ideas</button></li>
            <li class="nav-item"><button class="nav-link" data-tab="ranking">Global Ranking</button></li>
            <li class="nav-item"><button class="nav-link" data-tab="evolution">Evolution Graph</button></li>
            <li class="nav-item"><button class="nav-link" data-tab="ideas-by-gen">Ideas by Generation</button></li>
            <li class="nav-item"><button class="nav-link" data-tab="query">Task Query</button></li>
        </ul>

        <div id="tabContent">
            <!-- Ideas Tab -->
            <div class="tab-pane" id="tab-clusters" style="display:block;">
                <div class="row">
                    <div class="col-md-4">
                        <div class="card">
                            <div class="card-header">
                                Ideas (<span id="clusterCount">0</span>)
                                <button class="sort-btn active" id="clusterSortAsc" onclick="setClusterSort(true)">&#9650; Asc</button>
                                <button class="sort-btn" id="clusterSortDesc" onclick="setClusterSort(false)">&#9660; Desc</button>
                            </div>
                            <div class="card-body" id="clustersContainer" style="max-height: 600px; overflow-y: auto;"></div>
                        </div>
                    </div>
                    <div class="col-md-8">
                        <div class="card">
                            <div class="card-header">Candidate Details <span id="selectedCandidate" class="badge bg-primary ms-2"></span></div>
                            <div class="card-body">
                                <div id="candidateInfo" class="mb-3"><p class="text-muted">Select a candidate to view details</p></div>
                                <div id="candidateCode" class="code-container" style="display: none;"></div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Global Ranking Tab -->
            <div class="tab-pane" id="tab-ranking" style="display:none;">
                <div class="row">
                    <div class="col-md-6">
                        <div class="card">
                            <div class="card-header">
                                Global Ranking
                                <button class="sort-btn" id="rankSortAsc" onclick="setRankSort(true)">&#9650; Asc</button>
                                <button class="sort-btn" id="rankSortDesc" onclick="setRankSort(false)">&#9660; Desc</button>
                            </div>
                            <div class="card-body" style="max-height: 600px; overflow-y: auto;">
                                <table class="table table-striped table-hover table-sm">
                                    <thead><tr><th>Rank</th><th>Workspace ID</th><th>Gen</th><th>Idea</th><th>Metric</th><th>Status</th></tr></thead>
                                    <tbody id="rankingBody"></tbody>
                                </table>
                                <p id="rankingFooter" class="text-muted text-center" style="display:none;"></p>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="card">
                            <div class="card-header">Candidate Code <span id="selectedCandidateRanking" class="badge bg-primary ms-2"></span></div>
                            <div class="card-body">
                                <div id="candidateInfoRanking" class="mb-3"><p class="text-muted">Click a candidate in the ranking to view its code</p></div>
                                <div id="clusterDescriptionRanking" class="cluster-description-box mb-3" style="display: none;">
                                    <div class="cluster-description-label">Idea description</div>
                                    <div id="clusterDescriptionRankingText" class="cluster-description-text"></div>
                                </div>
                                <div id="candidateCodeRanking" class="code-container" style="display: none;"></div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Evolution Graph Tab -->
            <div class="tab-pane" id="tab-evolution" style="display:none;">
                <div class="row mb-3">
                    <div class="col-md-8">
                        <div class="card">
                            <div class="card-header">Evolution Over Generations</div>
                            <div class="card-body"><canvas id="evolutionChart"></canvas></div>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="card">
                            <div class="card-header">Generation Summary</div>
                            <div class="card-body" style="max-height: 400px; overflow-y: auto;">
                                <table class="table table-sm">
                                    <thead><tr><th>Gen</th><th>Total</th><th>Success</th><th>Failed</th><th>Best</th></tr></thead>
                                    <tbody id="genStatsBody"></tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="row">
                    <div class="col-12">
                        <div class="card">
                            <div class="card-header">Idea Evolution Per Generation</div>
                            <div class="card-body" style="min-height: 350px;"><canvas id="clusterEvolutionChart" style="min-height: 350px; height: 350px;"></canvas></div>
                        </div>
                    </div>
                </div>
                <div class="row mt-3">
                    <div class="col-12">
                        <div class="card">
                            <div class="card-header">Idea Details Per Generation</div>
                            <div class="card-body"><div id="clusterEvolutionDetails"></div></div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Ideas by Generation Tab -->
            <div class="tab-pane" id="tab-ideas-by-gen" style="display:none;">
                <div class="card">
                    <div class="card-header">Idea descriptions and best metric per generation</div>
                    <div class="card-body"><div id="ideasByGenContent" class="ideas-by-gen-content"></div></div>
                </div>
            </div>

            <!-- Query Tab -->
            <div class="tab-pane" id="tab-query" style="display:none;">
                <div class="card">
                    <div class="card-header">Task Query / Problem Statement</div>
                    <div class="card-body"><div id="queryBox" class="query-box"></div></div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // ============ State ============
        let appData = null;
        let rankSortAsc = true;   // will be set from higher_is_better on first load
        let clusterSortAsc = true;
        let rankSortInitialized = false;
        let evolutionChart = null;
        let clusterEvolutionChart = null;
        let refreshTimer = 5;
        let activeTab = 'clusters';

        // ============ Helpers ============
        function escapeHtml(str) {
            const div = document.createElement('div');
            div.textContent = str;
            return div.innerHTML;
        }

        function formatMetric(m) {
            if (m === null || m === undefined || m === "Infinity" || m === Infinity) return '\\u221e';
            if (m === "-Infinity" || m === -Infinity) return '-\\u221e';
            return parseFloat(m).toFixed(4);
        }

        function parseMetric(m) {
            if (m === "Infinity" || m === Infinity) return Infinity;
            if (m === "-Infinity" || m === -Infinity) return -Infinity;
            if (m === null || m === undefined) return Infinity;
            return parseFloat(m);
        }

        // ============ Tab Switching ============
        document.querySelectorAll('[data-tab]').forEach(btn => {
            btn.addEventListener('click', function() {
                document.querySelectorAll('.nav-link').forEach(t => t.classList.remove('active'));
                document.querySelectorAll('.tab-pane').forEach(p => p.style.display = 'none');
                this.classList.add('active');
                activeTab = this.getAttribute('data-tab');
                document.getElementById('tab-' + activeTab).style.display = 'block';
            });
        });

        // ============ Sort Controls ============
        function setRankSort(asc) {
            rankSortAsc = asc;
            document.getElementById('rankSortAsc').classList.toggle('active', asc);
            document.getElementById('rankSortDesc').classList.toggle('active', !asc);
            if (appData) renderRanking();
        }

        function setClusterSort(asc) {
            clusterSortAsc = asc;
            document.getElementById('clusterSortAsc').classList.toggle('active', asc);
            document.getElementById('clusterSortDesc').classList.toggle('active', !asc);
            if (appData) renderClusters();
        }

        // ============ Candidate Details ============
        function showCandidate(workspaceId, targetPrefix) {
            if (!appData) return;
            const candidate = appData.candidates[workspaceId];
            if (!candidate) return;

            const prefix = targetPrefix || '';
            document.getElementById('selectedCandidate' + prefix).textContent = workspaceId;

            const metricDisplay = formatMetric(candidate.metric);
            const errorHtml = candidate.error ? `<strong>Error:</strong> <span class="text-danger">${escapeHtml(candidate.error)}</span>` : '';

            document.getElementById('candidateInfo' + prefix).innerHTML = `
                <div class="row">
                    <div class="col-md-3"><strong>Status:</strong>
                        <span class="badge ${candidate.success ? 'success-badge' : 'failed-badge'}">${candidate.success ? 'Success' : 'Failed'}</span>
                    </div>
                    <div class="col-md-3"><strong>Metric:</strong>
                        <span class="metric-value ${candidate.success ? 'metric-good' : 'metric-bad'}">${metricDisplay}</span>
                    </div>
                    <div class="col-md-3"><strong>Generation:</strong> ${candidate.generation}</div>
                    <div class="col-md-3"><strong>Idea:</strong> <span class="cluster-name">${escapeHtml(candidate.cluster)}</span></div>
                </div>
                <div class="row mt-2">
                    <div class="col-md-6"><strong>Created:</strong> ${candidate.created_at || ''}</div>
                    <div class="col-md-6">${errorHtml}</div>
                </div>
            `;
            const codeEl = document.getElementById('candidateCode' + prefix);
            codeEl.style.display = 'block';
            codeEl.textContent = candidate.code || '(no code)';

            if (prefix === 'Ranking') {
                const descEl = document.getElementById('clusterDescriptionRanking');
                const textEl = document.getElementById('clusterDescriptionRankingText');
                const descriptions = appData.cluster_descriptions || {};
                const clusterDesc = (candidate.cluster && descriptions[candidate.cluster])
                    ? descriptions[candidate.cluster]
                    : '(no description for this idea)';
                textEl.textContent = clusterDesc;
                descEl.style.display = 'block';
            }
        }

        // ============ Render Functions ============
        function renderStats() {
            const d = appData;
            const failed = d.total_candidates - d.successful_candidates;
            document.getElementById('statTotal').textContent = d.total_candidates;
            document.getElementById('statSuccess').textContent = d.successful_candidates;
            document.getElementById('statFailed').textContent = failed;
            document.getElementById('statClusters').textContent = d.num_clusters;
            document.getElementById('statGens').textContent = d.generation_stats.length;
            document.getElementById('statBest').textContent = d.best_metric !== null ? formatMetric(d.best_metric) : 'N/A';
            document.getElementById('lastUpdated').textContent = d.last_updated;
        }

        function getSortedCandidates(candidates, ascending) {
            return [...candidates].sort((a, b) => {
                const ma = parseMetric(a.metric);
                const mb = parseMetric(b.metric);
                // Put Infinity at the end always
                if (ma === Infinity && mb === Infinity) return 0;
                if (ma === Infinity) return 1;
                if (mb === Infinity) return -1;
                if (ma === -Infinity && mb === -Infinity) return 0;
                if (ma === -Infinity) return ascending ? -1 : 1;
                if (mb === -Infinity) return ascending ? 1 : -1;
                return ascending ? ma - mb : mb - ma;
            });
        }

        function renderRanking() {
            const allCandidates = Object.values(appData.candidates);
            const sorted = getSortedCandidates(allCandidates, rankSortAsc);
            const top = sorted.slice(0, 50);

            let html = '';
            top.forEach((c, i) => {
                const rowClass = c.success ? 'table-success' : 'table-danger';
                const statusBadge = c.success
                    ? '<span class="badge success-badge">Success</span>'
                    : '<span class="badge failed-badge">Failed</span>';
                html += `<tr class="${rowClass}" onclick="showCandidate('${c.workspace_id}', 'Ranking')" style="cursor:pointer;">
                    <td><strong>${i + 1}</strong></td>
                    <td>${escapeHtml(c.workspace_id)}</td>
                    <td>${c.generation || 0}</td>
                    <td><small class="cluster-name">${escapeHtml(c.cluster || '')}</small></td>
                    <td class="metric-value">${formatMetric(c.metric)}</td>
                    <td>${statusBadge}</td>
                </tr>`;
            });
            document.getElementById('rankingBody').innerHTML = html;

            const footer = document.getElementById('rankingFooter');
            if (sorted.length > 50) {
                footer.textContent = `Showing top 50 of ${sorted.length} candidates`;
                footer.style.display = 'block';
            } else {
                footer.style.display = 'none';
            }
        }

        function renderClusters() {
            const clusters = {};
            for (const [wid, c] of Object.entries(appData.candidates)) {
                const cl = c.cluster || 'default';
                if (!clusters[cl]) clusters[cl] = [];
                clusters[cl].push(c);
            }

            // Sort candidates within each cluster
            for (const cl in clusters) {
                clusters[cl] = getSortedCandidates(clusters[cl], clusterSortAsc);
            }

            // Sort cluster keys numerically when possible, else lexicographically
            const clusterKeys = Object.keys(clusters).sort((a, b) => {
                const na = parseInt(a, 10);
                const nb = parseInt(b, 10);
                if (!isNaN(na) && !isNaN(nb)) return na - nb;
                return String(a).localeCompare(String(b));
            });

            document.getElementById('clusterCount').textContent = clusterKeys.length;

            const descriptions = appData.cluster_descriptions || {};
            let html = '';
            let idx = 0;
            for (const clusterName of clusterKeys) {
                const candidates = clusters[clusterName];
                idx++;
                const desc = descriptions[clusterName];
                const descLabel = (desc != null && desc !== '')
                    ? (clusterName + ' : ' + desc)
                    : (clusterName + ' : (no description)');
                const truncatedDesc = descLabel.length > 180 ? descLabel.substring(0, 177) + '...' : descLabel;

                let items = '';
                for (const c of candidates) {
                    const statusClass = c.success ? 'candidate-success' : 'candidate-failed';
                    const metricClass = c.success ? 'metric-good' : 'metric-bad';
                    items += `<div class="candidate-item ${statusClass}" onclick="showCandidate('${c.workspace_id}', '')">
                        <div class="d-flex justify-content-between align-items-center">
                            <span><small class="text-muted">Gen ${c.generation || 0}</small> ${escapeHtml(c.workspace_id)}</span>
                            <span class="metric-value ${metricClass}">${formatMetric(c.metric)}</span>
                        </div>
                    </div>`;
                }

                html += `<div class="cluster-section mb-3">
                    <div class="cluster-header" onclick="toggleCluster('cluster-${idx}')" title="${escapeHtml(descLabel)}">
                        <div class="d-flex justify-content-between align-items-start">
                            <strong class="cluster-name">${escapeHtml(clusterName)}</strong>
                            <span class="badge bg-secondary">${candidates.length}</span>
                        </div>
                        <div class="cluster-desc">${escapeHtml(truncatedDesc)}</div>
                    </div>
                    <div id="cluster-${idx}" class="collapse show">${items}</div>
                </div>`;
            }
            document.getElementById('clustersContainer').innerHTML = html;
        }

        function toggleCluster(id) {
            const el = document.getElementById(id);
            el.classList.toggle('show');
        }

        function renderGenStats() {
            let html = '';
            for (const g of appData.generation_stats) {
                const best = g.best_metric !== null ? formatMetric(g.best_metric) : 'N/A';
                html += `<tr>
                    <td><strong>${g.generation}</strong></td>
                    <td>${g.total}</td>
                    <td class="text-success">${g.successful}</td>
                    <td class="text-danger">${g.failed}</td>
                    <td class="metric-value">${best}</td>
                </tr>`;
            }
            document.getElementById('genStatsBody').innerHTML = html;
        }

        function renderQuery() {
            document.getElementById('queryBox').textContent = appData.query || '';
        }

        function renderIdeasByGen() {
            const ce = appData.cluster_evolution || [];
            const descriptions = appData.cluster_descriptions || {};
            let html = '';
            ce.forEach(gen => {
                html += `<div class="ideas-by-gen-gen">`;
                html += `<div class="gen-header">Generation ${gen.generation}</div>`;
                gen.clusters.forEach(cl => {
                    const desc = descriptions[cl.name];
                    const descText = (desc != null && desc !== '') ? escapeHtml(desc) : '(no description)';
                    const bestStr = cl.best_metric !== null && cl.best_metric !== undefined
                        ? formatMetric(cl.best_metric)
                        : '—';
                    const metricClass = cl.best_metric !== null && cl.best_metric !== undefined ? 'metric-value' : 'text-muted';
                    html += `<div class="ideas-by-gen-row">
                        <span class="idea-name">${escapeHtml(cl.name)}</span>
                        <span class="idea-metric ${metricClass}">Best: ${bestStr}</span>
                        <div class="idea-desc">${descText}</div>
                    </div>`;
                });
                html += `</div>`;
            });
            if (ce.length === 0) html = '<p class="text-muted">No generation data.</p>';
            document.getElementById('ideasByGenContent').innerHTML = html;
        }

        const clusterColors = [
            'rgba(13, 110, 253, 0.8)', 'rgba(40, 167, 69, 0.8)', 'rgba(220, 53, 69, 0.8)',
            'rgba(255, 193, 7, 0.8)', 'rgba(111, 66, 193, 0.8)', 'rgba(23, 162, 184, 0.8)',
            'rgba(253, 126, 20, 0.8)', 'rgba(108, 117, 125, 0.8)', 'rgba(0, 123, 255, 0.8)',
            'rgba(102, 16, 242, 0.8)',
        ];

        function renderCharts() {
            const gs = appData.generation_stats;

            // Destroy old charts
            if (evolutionChart) { evolutionChart.destroy(); evolutionChart = null; }
            if (clusterEvolutionChart) { clusterEvolutionChart.destroy(); clusterEvolutionChart = null; }

            // Main evolution chart
            const ctx = document.getElementById('evolutionChart').getContext('2d');
            evolutionChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: gs.map(g => 'Gen ' + g.generation),
                    datasets: [
                        { label: 'Successful', data: gs.map(g => g.successful), backgroundColor: 'rgba(40,167,69,0.7)', borderColor: 'rgba(40,167,69,1)', borderWidth: 1 },
                        { label: 'Failed', data: gs.map(g => g.failed), backgroundColor: 'rgba(220,53,69,0.7)', borderColor: 'rgba(220,53,69,1)', borderWidth: 1 },
                        { label: 'Best Metric', data: gs.map(g => g.best_metric), type: 'line', borderColor: 'rgba(13,110,253,1)', backgroundColor: 'rgba(13,110,253,0.2)', fill: false, yAxisID: 'y1', tension: 0.1 }
                    ]
                },
                options: {
                    responsive: true,
                    scales: {
                        x: { stacked: true },
                        y: { stacked: true, beginAtZero: true, title: { display: true, text: 'Number of Candidates' } },
                        y1: { type: 'linear', display: true, position: 'right', title: { display: true, text: 'Best Metric' }, grid: { drawOnChartArea: false } }
                    },
                    plugins: { title: { display: true, text: 'Candidate Evolution Across Generations' } }
                }
            });

            // Cluster evolution chart
            const ce = appData.cluster_evolution;
            const allClusterNames = new Set();
            ce.forEach(gen => gen.clusters.forEach(c => allClusterNames.add(c.name)));
            const clusterNamesList = Array.from(allClusterNames);

            const datasets = clusterNamesList.map((cn, idx) => ({
                label: cn.length > 25 ? cn.substring(0, 25) + '...' : cn,
                data: ce.map(gen => { const cl = gen.clusters.find(c => c.name === cn); return cl ? cl.best_metric : null; }),
                borderColor: clusterColors[idx % clusterColors.length],
                backgroundColor: clusterColors[idx % clusterColors.length].replace('0.8', '0.2'),
                fill: false, tension: 0.1, pointRadius: 6, pointHoverRadius: 8
            }));

            const ctx2 = document.getElementById('clusterEvolutionChart').getContext('2d');
            clusterEvolutionChart = new Chart(ctx2, {
                type: 'line',
                data: { labels: ce.map(g => 'Gen ' + g.generation), datasets: datasets },
                options: {
                    responsive: true, maintainAspectRatio: false,
                    scales: { y: { title: { display: true, text: 'Best Metric' } } },
                    plugins: {
                        title: { display: true, text: 'Idea Best Metrics Per Generation' },
                        legend: { position: 'bottom', labels: { boxWidth: 12, font: { size: 10 } } },
                        tooltip: { callbacks: { label: function(ctx) { const fn = clusterNamesList[ctx.datasetIndex]; const v = ctx.parsed.y !== null ? ctx.parsed.y.toFixed(4) : 'N/A'; return fn + ': ' + v; } } }
                    }
                }
            });

            // Cluster details per generation
            let detailsHtml = '<div class="row">';
            ce.forEach(gen => {
                detailsHtml += `<div class="col-md-4 mb-3"><div class="card h-100">
                    <div class="card-header bg-secondary text-white py-2"><strong>Generation ${gen.generation}</strong></div>
                    <div class="card-body p-2"><table class="table table-sm table-striped mb-0">
                        <thead><tr><th>Idea</th><th>Total</th><th>OK</th><th>Best</th></tr></thead><tbody>`;
                gen.clusters.forEach(cl => {
                    const best = cl.best_metric !== null ? parseFloat(cl.best_metric).toFixed(4) : 'N/A';
                    const rc = cl.best_metric !== null ? '' : 'text-muted';
                    const sn = cl.name.length > 20 ? cl.name.substring(0, 20) + '...' : cl.name;
                    detailsHtml += `<tr class="${rc}" title="${escapeHtml(cl.name)}">
                        <td><small class="cluster-name">${escapeHtml(sn)}</small></td>
                        <td>${cl.total}</td><td class="text-success">${cl.successful}</td>
                        <td class="metric-value">${best}</td></tr>`;
                });
                detailsHtml += '</tbody></table></div></div></div>';
            });
            detailsHtml += '</div>';
            document.getElementById('clusterEvolutionDetails').innerHTML = detailsHtml;
        }

        // ============ Data Fetching ============
        function fetchData() {
            fetch('/api/data')
                .then(r => r.json())
                .then(data => {
                    // Parse Infinity strings
                    for (const k in data.candidates) {
                        const m = data.candidates[k].metric;
                        if (m === "Infinity") data.candidates[k].metric = Infinity;
                        else if (m === "-Infinity") data.candidates[k].metric = -Infinity;
                    }
                    appData = data;

                    // Set default sort direction on first load based on higher_is_better
                    if (!rankSortInitialized) {
                        rankSortAsc = !data.higher_is_better;
                        clusterSortAsc = !data.higher_is_better;
                        document.getElementById('rankSortAsc').classList.toggle('active', rankSortAsc);
                        document.getElementById('rankSortDesc').classList.toggle('active', !rankSortAsc);
                        document.getElementById('clusterSortAsc').classList.toggle('active', clusterSortAsc);
                        document.getElementById('clusterSortDesc').classList.toggle('active', !clusterSortAsc);
                        rankSortInitialized = true;
                    }

                    renderStats();
                    renderRanking();
                    renderClusters();
                    renderGenStats();
                    renderQuery();
                    renderIdeasByGen();
                    renderCharts();
                })
                .catch(err => console.error('Fetch error:', err));
        }

        // ============ Refresh Countdown ============
        setInterval(() => {
            refreshTimer--;
            if (refreshTimer <= 0) {
                refreshTimer = 5;
                fetchData();
            }
            document.getElementById('refreshCountdown').textContent = refreshTimer;
        }, 1000);

        // ============ Initial Load ============
        fetchData();
    </script>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>'''


class LeaderboardHandler(SimpleHTTPRequestHandler):
    """Custom HTTP handler for serving the leaderboard."""

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            try:
                html_content = generate_html()
                self.send_response(200)
                self.send_header("Content-type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(html_content.encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-type", "text/plain")
                self.end_headers()
                self.wfile.write(f"Error: {str(e)}".encode("utf-8"))

        elif self.path == "/api/data":
            try:
                data = load_leaderboard()
                api_data = build_api_data(data)
                payload = json.dumps(api_data)
                self.send_response(200)
                self.send_header("Content-type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                self.end_headers()
                self.wfile.write(payload.encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))

        else:
            super().do_GET()

    def log_message(self, format, *args):
        """Suppress default logging for cleaner output."""
        pass


def main():
    global LEADERBOARD_PATH

    parser = argparse.ArgumentParser(
        description="Leaderboard Viewer - Visualize agent optimization results",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--workspace", "-w",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "workspaces"),
        help="Path to the workspace folder containing leaderboard.json (default: ./workspaces)"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8000,
        help="Port to run the server on (default: 8000)"
    )
    args = parser.parse_args()

    workspace_path = os.path.abspath(args.workspace)
    LEADERBOARD_PATH = os.path.join(workspace_path, "leaderboard.json")

    port = args.port
    server_address = ("", port)

    print(f"Loading leaderboard from: {LEADERBOARD_PATH}")

    if not os.path.exists(LEADERBOARD_PATH):
        print(f"Error: Leaderboard file not found at {LEADERBOARD_PATH}")
        return

    data = load_leaderboard()
    print(f"  - Total candidates: {data.get('total_candidates', 0)}")
    print(f"  - Successful: {data.get('successful_candidates', 0)}")
    print(f"  - Ideas: {data.get('num_clusters', 0)}")
    print()
    print(f"Starting server at http://localhost:{port}")
    print("Press Ctrl+C to stop")

    httpd = HTTPServer(server_address, LeaderboardHandler)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        httpd.server_close()


if __name__ == "__main__":
    main()
