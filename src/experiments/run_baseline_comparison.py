import os
import numpy as np
from typing import Dict, List
from src.utils.config_loader import load_config
from src.utils.data_loader import DataLoader
from src.pipeline import EBR_CFR_Pipeline
from src.models.schemas import Document
from src.baselines.bm25_ranker import BM25Ranker
from src.baselines.dense_ranker import DenseEmbeddingRanker
from src.baselines.cross_encoder_ranker import CrossEncoderRanker
from src.baselines.llm_judge_ranker import LLMJudgeRanker
from src.evaluation.metrics import EvaluationMetrics

def run_baseline_experiments(test_file: str = None) -> List[Dict]:
    if test_file is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        test_file = os.path.join(base_dir, "data", "test_cases", "benchmark_600_cases.json")
        if not os.path.exists(test_file):
            test_file = os.path.join(base_dir, "data", "test_cases", "c30_concrete_eval.json")

    cases = DataLoader.load_benchmark_dataset(test_file)
    config = load_config()
    pipeline = EBR_CFR_Pipeline(config)

    bm25 = BM25Ranker()
    dense = DenseEmbeddingRanker()
    ce = CrossEncoderRanker()
    llm = LLMJudgeRanker()

    metrics_tracker = {
        "BM25 (传统检索)": {"ndcg": [], "mrr": [], "fact1": [], "fact3": []},
        "Dense Embedding (向量召回)": {"ndcg": [], "mrr": [], "fact1": [], "fact3": []},
        "Cross-Encoder (深度交互)": {"ndcg": [], "mrr": [], "fact1": [], "fact3": []},
        "LLM-as-a-judge (大模型打分)": {"ndcg": [], "mrr": [], "fact1": [], "fact3": []},
        "EBR-CFR (本文提出框架)": {"ndcg": [], "mrr": [], "fact1": [], "fact3": []}
    }

    print(f"\n🔬 正在对 {len(cases)} 组真实测试用例执行主流基线模型横向比对评测...")

    for case in cases:
        query = case.get("query", "")
        # 核心修复 1：动态读取每个 Case 真实的 ground_truth_relevance 与 conflict_ids
        ground_truth = case.get("ground_truth_relevance", {})
        conflict_ids = case.get("conflict_ids", [])
        
        raw_candidates = case.get("candidates", [])
        if not raw_candidates:
            continue

        real_docs = [
            Document(
                doc_id=c.get("doc_id", "UNKNOWN"),
                content=c.get("content", ""),
                source=c.get("source", "未知来源")
            )
            for c in raw_candidates
        ]

        # 1. BM25
        bm25_r = [d[0].doc_id for d in bm25.rank(query, real_docs)]
        # 2. Dense
        dense_r = [d[0].doc_id for d in dense.rank(query, real_docs)]
        # 3. Cross-Encoder
        ce_r = [d[0].doc_id for d in ce.rank(query, real_docs)]
        # 4. LLM
        llm_r = [d[0].doc_id for d in llm.rank(query, real_docs)]
        # 5. EBR-CFR
        ebr_ranked, _ = pipeline.run(query, real_docs)
        ebr_r = [r.document.doc_id for r in ebr_ranked]

        run_pairs = [
            ("BM25 (传统检索)", bm25_r),
            ("Dense Embedding (向量召回)", dense_r),
            ("Cross-Encoder (深度交互)", ce_r),
            ("LLM-as-a-judge (大模型打分)", llm_r),
            ("EBR-CFR (本文提出框架)", ebr_r)
        ]

        k_val = min(4, len(real_docs))
        for m_name, ranked_ids in run_pairs:
            metrics_tracker[m_name]["ndcg"].append(EvaluationMetrics.ndcg_at_k(ranked_ids, ground_truth, k=k_val))
            metrics_tracker[m_name]["mrr"].append(EvaluationMetrics.mrr_at_k(ranked_ids, ground_truth, k=k_val))
            metrics_tracker[m_name]["fact1"].append(EvaluationMetrics.factuality_at_k(ranked_ids, conflict_ids, k=1))
            metrics_tracker[m_name]["fact3"].append(EvaluationMetrics.factuality_at_k(ranked_ids, conflict_ids, k=min(3, len(ranked_ids))))

    report_rows = []
    for model_name, m_dict in metrics_tracker.items():
        report_rows.append({
            "对比基线模型": model_name,
            "NDCG@4": round(float(np.mean(m_dict["ndcg"])), 4),
            "MRR": round(float(np.mean(m_dict["mrr"])), 4),
            "Factuality@1": f"{np.mean(m_dict['fact1']) * 100:.1f}%",
            "Factuality@3": f"{np.mean(m_dict['fact3']) * 100:.1f}%"
        })

    return report_rows
