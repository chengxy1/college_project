import random
from typing import List, Tuple
from src.models.schemas import Document

class LLMJudgeRanker:
    """基线 4：LLM-as-a-judge 成对比较打分基线 (模拟 Qwen / LLaMA-3 打分器)"""
    def __init__(self, temperature: float = 0.2):
        self.temperature = temperature

    def rank(self, query: str, docs: List[Document]) -> List[Tuple[Document, float]]:
        ranked_scores = []
        for doc in docs:
            score = 0.5
            if "GB" in doc.content or "规范" in doc.content:
                score += 0.35
            if "实测" in doc.content or "检测" in doc.content:
                score += 0.25
            if "招工" in doc.content:
                score -= 0.4
            score += (random.random() - 0.5) * 0.05
            ranked_scores.append((doc, float(score)))

        ranked_scores.sort(key=lambda x: x[1], reverse=True)
        return ranked_scores
