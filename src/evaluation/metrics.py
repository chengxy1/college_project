import numpy as np
import scipy.stats as stats
from typing import List, Dict

class EvaluationMetrics:
    """定量评测指标计算引擎：计算 NDCG@K, MRR@K, Factuality@K 与 Kendall-Tau"""
    @staticmethod
    def dcg_at_k(relevance_scores: List[float], k: int) -> float:
        scores = np.asfarray(relevance_scores)[:k]
        if scores.size == 0:
            return 0.0
        discounts = np.log2(np.arange(len(scores)) + 2)
        return float(np.sum((2 ** scores - 1) / discounts))

    @classmethod
    def ndcg_at_k(cls, ranked_ids: List[str], ground_truth_relevance: Dict[str, float], k: int = 10) -> float:
        relevance_scores = [ground_truth_relevance.get(doc_id, 0.0) for doc_id in ranked_ids]
        actual_dcg = cls.dcg_at_k(relevance_scores, k)
        ideal_scores = sorted(list(ground_truth_relevance.values()), reverse=True)
        ideal_dcg = cls.dcg_at_k(ideal_scores, k)
        return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0

    @staticmethod
    def mrr_at_k(ranked_ids: List[str], ground_truth_relevance: Dict[str, float], k: int = 10) -> float:
        for i, doc_id in enumerate(ranked_ids[:k], start=1):
            if ground_truth_relevance.get(doc_id, 0.0) >= 1.0:
                return 1.0 / i
        return 0.0

    @staticmethod
    def factuality_at_k(ranked_ids: List[str], conflict_doc_ids: List[str], k: int = 1) -> float:
        top_k = ranked_ids[:k]
        if not top_k:
            return 0.0
        conflict_count = sum(1 for doc_id in top_k if doc_id in conflict_doc_ids)
        return 1.0 - (conflict_count / len(top_k))

    @staticmethod
    def kendall_tau(scores_a: List[float], scores_b: List[float]) -> float:
        tau, _ = stats.kendalltau(scores_a, scores_b)
        return float(tau) if not np.isnan(tau) else 1.0
