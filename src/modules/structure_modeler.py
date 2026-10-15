import re
import networkx as nx
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.models.schemas import Document

class EvidenceStructureModeler:
    def __init__(self, config: dict):
        self.threshold = config.get("similarity_threshold", 0.12)
        self.penalty = config.get("intra_conflict_penalty", 0.35)
        self.vectorizer = TfidfVectorizer()

    def extract_claims(self, text: str) -> List[str]:
        sentences = re.split(r'[。！；\n.?;]', text)
        return [s.strip() for s in sentences if len(s.strip()) > 5]

    def evaluate_intra_consistency(self, doc: Document, query: str) -> float:
        penalty = 0.0
        if ("必须" in doc.content and "无需" in doc.content) or ("严禁" in doc.content and "允许" in doc.content):
            penalty += self.penalty

        query_tokens = set(re.findall(r'\w+', query.lower()))
        doc_tokens = set(re.findall(r'\w+', doc.content.lower()))
        overlap = len(query_tokens.intersection(doc_tokens)) / (len(query_tokens) + 1e-9)

        base_score = 0.5 + (0.5 * overlap) - penalty
        return max(0.05, min(1.0, base_score))

    def build_evidence_graph(self, docs: List[Document]) -> nx.Graph:
        G = nx.Graph()
        if not docs:
            return G

        for doc in docs:
            G.add_node(doc.doc_id, doc_obj=doc)

        corpus = [doc.content for doc in docs]
        try:
            tfidf_matrix = self.vectorizer.fit_transform(corpus)
            sim_matrix = cosine_similarity(tfidf_matrix)

            for i in range(len(docs)):
                for j in range(i + 1, len(docs)):
                    sim = float(sim_matrix[i, j])
                    if sim >= self.threshold:
                        G.add_edge(docs[i].doc_id, docs[j].doc_id, weight=sim)
        except Exception:
            pass
        return G
