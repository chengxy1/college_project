# test_analyze.py
from select_top10_and_build import analyze_standards_in_dataset

# 扫描并输出前 10 篇最优质的规范清单
top10_list = analyze_standards_in_dataset(
    raw_jsonl_path="data/test_cases/3.5_clean_qa_dataset.jsonl",
    top_n=10
)