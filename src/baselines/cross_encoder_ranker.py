from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.models.schemas import Document

class CrossEncoderRanker:
    """基线 3：模拟深度交叉编码器 (Cross-Encoder / MiniLM / bge-reranker)"""
    def __init__(self):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))

    def rank(self, query: str, docs: List[Document]) -> List[Tuple[Document, float]]:
        if not docs:
            return []
        corpus = [query] + [d.content for d in docs]
        matrix = self.vectorizer.fit_transform(corpus)
        sims = cosine_similarity(matrix[0:1], matrix[1:])[0]

        scores = []
        for i, doc in enumerate(docs):
            deep_score = float(sims[i])
            if "规范" in doc.content or "标准" in doc.content:
                deep_score *= 1.15
            scores.append(deep_score)

        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return ranked
