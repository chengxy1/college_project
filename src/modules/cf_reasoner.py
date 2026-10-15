import networkx as nx
from typing import Set, Dict
from src.models.schemas import Document

class CounterfactualReasoner:
    def __init__(self, config: dict):
        self.k_hop = config.get("k_hop", 2)
        self.alpha = config.get("damping_factor", 0.85)

    def _get_k_hop_neighbors(self, G: nx.Graph, node: str, k: int) -> Set[str]:
        neighbors = set()
        current_layer = {node}
        for _ in range(k):
            next_layer = set()
            for n in current_layer:
                if n in G:
                    next_layer.update(G.neighbors(n))
            neighbors.update(next_layer)
            current_layer = next_layer
        if node in neighbors:
            neighbors.remove(node)
        return neighbors

    def compute_counterfactual_impact(self, G: nx.Graph) -> Dict[str, float]:
        impact_scores = {}
        if G.number_of_nodes() == 0:
            return impact_scores

        try:
            base_centrality = nx.pagerank(G, weight='weight', alpha=self.alpha)
        except Exception:
            base_centrality = nx.degree_centrality(G)

        for node in G.nodes():
            doc_obj: Document = G.nodes[node]['doc_obj']
            k_neighbors = self._get_k_hop_neighbors(G, node, self.k_hop)

            if not k_neighbors:
                impact_scores[node] = doc_obj.base_confidence * 0.5
                continue

            # 反事实干预：模拟物理剔除节点
            G_cf = G.copy()
            G_cf.remove_node(node)

            try:
                cf_centrality = nx.pagerank(G_cf, weight='weight', alpha=self.alpha)
            except Exception:
                cf_centrality = nx.degree_centrality(G_cf)

            marginal_loss = 0.0
            for nb in k_neighbors:
                orig_c = base_centrality.get(nb, 0.0)
                new_c = cf_centrality.get(nb, 0.0)
                marginal_loss += max(0.0, orig_c - new_c)

            impact_scores[node] = doc_obj.base_confidence * (1.0 + marginal_loss)

        return impact_scores
