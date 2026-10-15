import math
import re
from typing import List, Tuple
from src.models.schemas import Document

class BM25Ranker:
    """基线 1：经典稀疏检索 BM25 算法"""
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\w+', text.lower())

    def rank(self, query: str, docs: List[Document]) -> List[Tuple[Document, float]]:
        query_tokens = self._tokenize(query)
        corpus = [self._tokenize(d.content) for d in docs]
        num_docs = len(docs)
        if num_docs == 0:
            return []

        avg_dl = sum(len(c) for c in corpus) / (num_docs + 1e-9)
        scores = []
        for doc_tokens in corpus:
            doc_len = len(doc_tokens)
            score = 0.0
            for qt in query_tokens:
                n_q = sum(1 for d in corpus if qt in d)
                idf = math.log((num_docs - n_q + 0.5) / (n_q + 0.5) + 1.0)
                tf = doc_tokens.count(qt)
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / (avg_dl + 1e-9)))
                score += idf * (numerator / (denominator + 1e-9))
            scores.append(score)

        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return ranked
