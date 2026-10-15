from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.models.schemas import Document

class DenseEmbeddingRanker:
    """基线 2：向量语义相似度直接排序 (Dense Retrieval Baseline)"""
    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def rank(self, query: str, docs: List[Document]) -> List[Tuple[Document, float]]:
        if not docs:
            return []
        corpus = [query] + [d.content for d in docs]
        matrix = self.vectorizer.fit_transform(corpus)
        sim_scores = cosine_similarity(matrix[0:1], matrix[1:])[0]

        ranked = sorted(zip(docs, sim_scores), key=lambda x: x[1], reverse=True)
        return [(doc, float(score)) for doc, score in ranked]
