import numpy as np
from scipy import stats
from typing import List, Tuple

class StatisticalSignificanceTest:
    """
    学术论文必备：统计显著性检验 (Paired t-test 与 Wilcoxon 符号秩检验)
    用于证明 EBR-CFR 相较于基线模型的提升具备统计学显著性 (p < 0.05 或 p < 0.01)
    """
    @staticmethod
    def paired_t_test(baseline_scores: List[float], proposed_scores: List[float]) -> Tuple[float, float]:
        """配对样本 t 检验"""
        if len(baseline_scores) != len(proposed_scores) or len(baseline_scores) < 2:
            return 0.0, 1.0
        t_stat, p_val = stats.ttest_rel(proposed_scores, baseline_scores)
        return float(t_stat), float(p_val)

    @staticmethod
    def wilcoxon_test(baseline_scores: List[float], proposed_scores: List[float]) -> Tuple[float, float]:
        """Wilcoxon 符号秩检验 (非参数检验，适合非正态分布的排序指标)"""
        diff = np.array(proposed_scores) - np.array(baseline_scores)
        if np.all(diff == 0) or len(diff) < 5:
            return 0.0, 1.0
        try:
            stat, p_val = stats.wilcoxon(proposed_scores, baseline_scores)
            return float(stat), float(p_val)
        except Exception:
            return 0.0, 1.0
