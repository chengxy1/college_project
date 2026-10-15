import os
import pytest
from src.utils.config_loader import load_config
from src.utils.data_loader import DataLoader
from src.pipeline import EBR_CFR_Pipeline

def test_pipeline_ranking_and_filtering():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, "config", "settings.yaml")
    test_json = os.path.join(base_dir, "data", "test_cases", "c30_concrete_eval.json")

    config = load_config(config_path)
    pipeline = EBR_CFR_Pipeline(config)
    query, docs = DataLoader.load_test_case(test_json)

    ranked_results = pipeline.run(query, docs)

    # 1. 验证过滤能力：存在自相矛盾的噪音文档是否被拦截
    ranked_ids = [r.document.doc_id for r in ranked_results]
    assert "DOC_005_NOISE_RECRUIT" not in ranked_ids, "低置信度噪音文档未被滤除"

    # 2. 验证排序有效性
    assert ranked_results[0].cf_score >= ranked_results[-1].cf_score

    # 3. 验证元数据契约生成
    top_meta = ranked_results[0].metadata
    assert hasattr(top_meta, "verification_chain")
    assert hasattr(top_meta, "cf_summary")
