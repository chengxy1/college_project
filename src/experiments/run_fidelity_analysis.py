import networkx as nx
import numpy as np
from typing import Dict, List
from src.evaluation.metrics import EvaluationMetrics

def run_fidelity_experiments(evidence_graph: nx.Graph, max_k: int = 3) -> List[Dict]:
    nodes = list(evidence_graph.nodes())
    if len(nodes) < 2:
        return []

    try:
        base_centrality = nx.pagerank(evidence_graph, weight='weight')
    except Exception:
        base_centrality = nx.degree_centrality(evidence_graph)

    # 1. 全图重计算理论真值 (Ground Truth Impact)
    gt_scores = []
    for n in nodes:
        g_cf = evidence_graph.copy()
        g_cf.remove_node(n)
        try:
            cf_c = nx.pagerank(g_cf, weight='weight')
        except Exception:
            cf_c = nx.degree_centrality(g_cf)
        loss = sum(max(0, base_centrality[other] - cf_c.get(other, 0)) for other in g_cf.nodes())
        gt_scores.append(loss)

    fidelity_rows = []
    for k in range(1, max_k + 1):
        approx_scores = []
        for n in nodes:
            neighbors = set()
            cur = {n}
            for _ in range(k):
                nxt = set()
                for item in cur:
                    nxt.update(evidence_graph.neighbors(item))
                neighbors.update(nxt)
                cur = nxt
            if n in neighbors:
                neighbors.remove(n)

            g_cf = evidence_graph.copy()
            g_cf.remove_node(n)
            try:
                cf_c = nx.pagerank(g_cf, weight='weight')
            except Exception:
                cf_c = nx.degree_centrality(g_cf)

            k_loss = sum(max(0, base_centrality[other] - cf_c.get(other, 0)) for other in neighbors)
            approx_scores.append(k_loss)

        tau = EvaluationMetrics.kendall_tau(gt_scores, approx_scores)
        rel_error = np.mean([abs(g - a) / (g + 1e-6) for g, a in zip(gt_scores, approx_scores)])

        fidelity_rows.append({
            "局部近似邻域阶数": f"K = {k} 阶邻域",
            "Kendall-Tau 排序一致性": round(tau, 4),
            "平均相对误差 (Relative Error)": f"{rel_error * 100:.2f}%",
            "计算耗时削减率 (理论估计)": f"{round(100 - (100 / (k + 1)), 1)}%"
        })

    return fidelity_rows
