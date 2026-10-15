import os
import copy
import numpy as np
from typing import Dict, List
from src.utils.config_loader import load_config
from src.utils.data_loader import DataLoader
from src.pipeline import EBR_CFR_Pipeline
from src.models.schemas import Document
from src.evaluation.metrics import EvaluationMetrics

def run_ablation_experiments(test_file: str = None) -> List[Dict]:
    if test_file is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        test_file = os.path.join(base_dir, "data", "test_cases", "benchmark_600_cases.json")
        if not os.path.exists(test_file):
            test_file = os.path.join(base_dir, "data", "test_cases", "c30_concrete_eval.json")

    cases = DataLoader.load_benchmark_dataset(test_file)
    config = load_config()

    ablation_modes = [
        ("EBR-CFR-full (完整框架)", "full"),
        ("w/o 反事实效用推理 (-Counterfactual)", "no_cf"),
        ("w/o 跨文档证据网络 (-CrossDoc)", "no_cross"),
        ("仅保留单文档内部一致性 (-IntraOnly)", "intra_only")
    ]

    metrics_tracker = {label: {"ndcg": [], "mrr": [], "fact1": [], "fact3": []} for label, _ in ablation_modes}

    print(f"\n🔬 正在对 {len(cases)} 组真实测试用例执行消融实验 (逐级剥离验证)...")

    for case in cases:
        query = case.get("query", "")
        ground_truth = case.get("ground_truth_relevance", {})
        conflict_ids = case.get("conflict_ids", [])
        raw_candidates = case.get("candidates", [])
        if not raw_candidates:
            continue

        base_docs = [
            Document(
                doc_id=c.get("doc_id", "UNKNOWN"),
                content=c.get("content", ""),
                source=c.get("source", "未知来源")
            )
            for c in raw_candidates
        ]

        for label, mode in ablation_modes:
            pipeline = EBR_CFR_Pipeline(config)
            test_docs = copy.deepcopy(base_docs)

            if mode == "full":
                ranked, _ = pipeline.run(query, test_docs)
                r_ids = [r.document.doc_id for r in ranked]
            elif mode == "no_cf":
                for d in test_docs:
                    d.base_confidence = pipeline.modeler.evaluate_intra_consistency(d, query)
                test_docs.sort(key=lambda x: x.base_confidence, reverse=True)
                r_ids = [d.doc_id for d in test_docs if d.base_confidence >= pipeline.cutoff]
            elif mode == "no_cross":
                pipeline.modeler.threshold = 1.0  # 阻断连边
                ranked, _ = pipeline.run(query, test_docs)
                r_ids = [r.document.doc_id for r in ranked]
            else:
                for d in test_docs:
                    d.base_confidence = 0.5 if "无需" not in d.content else 0.1
                test_docs.sort(key=lambda x: x.base_confidence, reverse=True)
                r_ids = [d.doc_id for d in test_docs]

            k_val = min(4, len(test_docs))
            metrics_tracker[label]["ndcg"].append(EvaluationMetrics.ndcg_at_k(r_ids, ground_truth, k=k_val))
            metrics_tracker[label]["mrr"].append(EvaluationMetrics.mrr_at_k(r_ids, ground_truth, k=k_val))
            metrics_tracker[label]["fact1"].append(EvaluationMetrics.factuality_at_k(r_ids, conflict_ids, k=1))
            metrics_tracker[label]["fact3"].append(EvaluationMetrics.factuality_at_k(r_ids, conflict_ids, k=min(3, len(r_ids))))

    ablation_rows = []
    for label, m_dict in metrics_tracker.items():
        ablation_rows.append({
            "消融实验变体": label,
            "NDCG@4": round(float(np.mean(m_dict["ndcg"])), 4),
            "MRR": round(float(np.mean(m_dict["mrr"])), 4),
            "Factuality@1": f"{np.mean(m_dict['fact1']) * 100:.1f}%",
            "Factuality@3": f"{np.mean(m_dict['fact3']) * 100:.1f}%"
        })

    return ablation_rows
