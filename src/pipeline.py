from typing import List, Tuple
import networkx as nx
from src.models.schemas import Document, RankedResult, ExplanatoryMetadata
from src.modules.structure_modeler import EvidenceStructureModeler
from src.modules.cf_reasoner import CounterfactualReasoner


class EBR_CFR_Pipeline:
    def __init__(self, config: dict):
        self.cutoff = config.get("evidence_modeling", {}).get("confidence_cutoff", 0.3)
        self.modeler = EvidenceStructureModeler(config.get("evidence_modeling", {}))
        self.reasoner = CounterfactualReasoner(config.get("counterfactual_reasoning", {}))

    def run(self, query: str, docs: List[Document]) -> Tuple[List[RankedResult], nx.Graph]:
        # 1. 证据结构建模
        for doc in docs:
            doc.claims = self.modeler.extract_claims(doc.content)
            doc.base_confidence = self.modeler.evaluate_intra_consistency(doc, query)

        graph = self.modeler.build_evidence_graph(docs)

        # 2. 反事实效用推理
        cf_scores = self.reasoner.compute_counterfactual_impact(graph)

        # 3. 契约装配与准入过滤
        results = []
        for doc in docs:
            if doc.base_confidence < self.cutoff:
                continue

            score = cf_scores.get(doc.doc_id, 0.0)
            neighbors = list(graph.neighbors(doc.doc_id)) if doc.doc_id in graph else []

            conflict_tags = ["存在事实/指标冲突"] if "不合格" in doc.content or "废弃" in doc.content else [
                "内部校验一致"]
            cf_summary = f"剔除该节点后将破坏 {len(neighbors)} 条证据支撑链，边际效用损失为 {score - doc.base_confidence:.4f}"

            meta = ExplanatoryMetadata(
                conflict_tags=conflict_tags,
                verification_chain=neighbors,
                cf_summary=cf_summary
            )
            results.append(RankedResult(document=doc, cf_score=score, metadata=meta))

        # 4. 排序输出
        results.sort(key=lambda x: x.cf_score, reverse=True)

        # 同时返回排序列表和图对象供可视化使用
        return results, graph