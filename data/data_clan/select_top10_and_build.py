import os
import sys
import json
import random
import re
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any

# 自动解析项目根路径 (D:\college_project)
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent if CURRENT_DIR.name == "data_clan" else CURRENT_DIR

DEFAULT_RAW_JSONL = PROJECT_ROOT / "data" / "test_cases" / "3.5_clean_qa_dataset.jsonl"
DEFAULT_OUTPUT_BENCHMARK = PROJECT_ROOT / "data" / "test_cases" / "benchmark_600_cases.json"

def clean_text(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()

def is_meaningless_qa(question: str, context: str, answer: str) -> bool:
    """过滤目录、纯页码、无实质工程意义的废题"""
    bad_keywords = ["目录", "目次", "编制说明", "起草单位", "起草人", "…………………………………………", "……"]
    for kw in bad_keywords:
        if kw in question or kw in context:
            return True
    if re.search(r'编号\s*[\d\.]+\s*对应', question):
        return True
    if len(question) < 8 or len(context) < 50 or len(answer) < 2:
        return True
    return False

def analyze_dataset_standards(raw_jsonl_path: str, top_n: int = 10) -> List[str]:
    std_stats = defaultdict(lambda: {"count": 0, "task_types": defaultdict(int)})
    with open(raw_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            try:
                item = json.loads(line)
            except Exception:
                continue

            q = item.get("question", "").strip()
            ctx = item.get("context", "").strip()
            ans = item.get("gold_answer", "").strip()
            src = item.get("source_standard", "").strip()

            if is_meaningless_qa(q, ctx, ans):
                continue

            std_stats[src]["count"] += 1
            std_stats[src]["task_types"][item.get("task_type", "未分类")] += 1

    sorted_stds = sorted(std_stats.items(), key=lambda x: x[1]["count"], reverse=True)
    print("\n" + "=" * 80)
    print(f"🏆 过滤目录与无效废题后，优质题量 Top-{top_n} 规范清单：")
    print("=" * 80)
    top_list = []
    for rank, (name, stat) in enumerate(sorted_stds[:top_n], start=1):
        print(f"[{rank:02d}] 规范名称: {name} (高质量问答: {stat['count']} 条)")
        top_list.append(name)
    return top_list


def build_benchmark_for_standards(
    raw_jsonl_path: str,
    selected_standards: List[str],
    output_path: str,
    target_count: int = 600,
    conflict_ratio: float = 0.25,
    seed: int = 42
):
    random.seed(seed)
    selected_set = set(selected_standards)
    per_std_quota = target_count // len(selected_standards)

    print(f"\n{'=' * 80}")
    print(f"🎯 正在构建高质量 600 条 Benchmark（强制剔除目录废题，每组至少 4~5 篇候选文档）...")
    print(f"{'=' * 80}")

    std_buckets = defaultdict(lambda: defaultdict(list))
    with open(raw_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            try:
                item = json.loads(line)
            except Exception:
                continue

            src = item.get("source_standard", "").strip()
            if src not in selected_set:
                continue

            q = item.get("question", "").strip()
            ctx = item.get("context", "").strip()
            ans = item.get("gold_answer", "").strip()
            task_type = item.get("task_type", "工程规范审核").strip()

            if is_meaningless_qa(q, ctx, ans):
                continue

            std_buckets[src][task_type].append(item)

    sampled_records = []
    for src in selected_standards:
        task_dict = std_buckets[src]
        if not task_dict: continue

        std_items = []
        per_task = max(1, per_std_quota // max(1, len(task_dict)))
        for t_type, items in task_dict.items():
            std_items.extend(random.sample(items, min(len(items), per_task)))

        if len(std_items) < per_std_quota:
            remaining = [it for sub in task_dict.values() for it in sub if it not in std_items]
            std_items.extend(random.sample(remaining, min(len(remaining), per_std_quota - len(std_items))))

        sampled_records.extend(std_items)

    if len(sampled_records) > target_count:
        sampled_records = random.sample(sampled_records, target_count)
    elif len(sampled_records) < target_count:
        all_leftovers = [it for t_dict in std_buckets.values() for it_list in t_dict.values() for it in it_list if it not in sampled_records]
        sampled_records.extend(random.sample(all_leftovers, min(len(all_leftovers), target_count - len(sampled_records))))

    # 全局语料池用于采样负样本
    global_contexts = [r.get("context", "") for r in sampled_records]

    benchmark_dataset = []
    for idx, item in enumerate(sampled_records, start=1):
        case_id = f"CASE_{idx:04d}"
        query = clean_text(item.get("question", ""))
        gold_context = clean_text(item.get("context", ""))
        source_std = item.get("source_standard", "")
        clause = item.get("clause_number", "条文")
        task_type = item.get("task_type", "工程技术审核")

        candidates = []
        gt_relevance = {}
        conflict_ids = []

        # 1. 黄金正样本 (权重 3.0)
        pos_id = f"DOC_{idx:04d}_POS"
        candidates.append({
            "doc_id": pos_id,
            "source": f"{source_std} [{clause}]",
            "content": gold_context
        })
        gt_relevance[pos_id] = 3.0

        # 2. 强干扰项 (切分子句或同规范条文，权重 1.0)
        sub_paras = [p.strip() for p in re.split(r'[。；\n]+', gold_context) if len(p.strip()) > 15]
        if len(sub_paras) > 1:
            for p_i, p_text in enumerate(sub_paras[1:3], start=1):
                hard_id = f"DOC_{idx:04d}_HARD_{p_i}"
                candidates.append({
                    "doc_id": hard_id,
                    "source": f"{source_std} 关联规定",
                    "content": p_text + "。"
                })
                gt_relevance[hard_id] = 1.0
        else:
            # 若无法切分，从同规范其他上下文借一条
            other_ctx = random.choice(std_buckets[source_std][task_type])["context"]
            hard_id = f"DOC_{idx:04d}_HARD_1"
            candidates.append({
                "doc_id": hard_id,
                "source": f"{source_std} 临近条款",
                "content": clean_text(other_ctx[:120]) + "..."
            })
            gt_relevance[hard_id] = 1.0

        # 3. 跨规范弱相关负样本 2 篇 (确保文档池丰富，权重 0.0)
        distractor_samples = random.sample(global_contexts, 2)
        for d_i, d_ctx in enumerate(distractor_samples, start=1):
            dist_id = f"DOC_{idx:04d}_DIST_{d_i}"
            candidates.append({
                "doc_id": dist_id,
                "source": "其他建筑类标准参考",
                "content": clean_text(d_ctx[:110]) + "..."
            })
            gt_relevance[dist_id] = 0.0

        # 4. 注入冲突样本 (约 25% 注入矛盾或废止版本)
        if random.random() < conflict_ratio:
            conflict_id = f"DOC_{idx:04d}_CONFLICT"
            precond = item.get("precondition", "该工程参数指标")
            num_th = item.get("num_threshold", "标准限值")
            if num_th and num_th != "无":
                conflict_content = f"【历史废止/地方争议标准】关于{precond}，原旧版或地方细则允许按指标 {num_th} 浮动下调执行（注：与本规范最新现行强制性标准存在指标冲突）。"
            else:
                conflict_content = f"【历史废止规范】关于{precond}的检验与设计要求，按照已作废旧版规程执行，无需执行全项复核（注：已被住建部新规强制替换）。"

            candidates.append({
                "doc_id": conflict_id,
                "source": "历史废止/争议条文",
                "content": conflict_content
            })
            gt_relevance[conflict_id] = 0.5
            conflict_ids.append(conflict_id)

        random.shuffle(candidates)

        benchmark_dataset.append({
            "case_id": case_id,
            "task_type": task_type,
            "source_standard": source_std,
            "query": query,
            "golden_answer": item.get("gold_answer", ""),
            "ground_truth_relevance": gt_relevance,
            "conflict_ids": conflict_ids,
            "candidates": candidates
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_dataset, f, ensure_ascii=False, indent=2)

    print(f"\n🎉 高质量评测集重新生成完毕 -> {os.path.abspath(output_path)}")
    print(f"📌 总用例数: {len(benchmark_dataset)} 组")
    print(f"📌 平均每条用例候选文档数: {sum(len(c['candidates']) for c in benchmark_dataset) / len(benchmark_dataset):.1f} 篇（完美支持拓扑图）")
    print(f"📌 注入冲突样本: {sum(1 for c in benchmark_dataset if c['conflict_ids'])} 组")

def main():
    raw_jsonl = str(DEFAULT_RAW_JSONL)
    output_benchmark = str(DEFAULT_OUTPUT_BENCHMARK)

    selected_standards = analyze_dataset_standards(raw_jsonl, top_n=10)
    build_benchmark_for_standards(
        raw_jsonl_path=raw_jsonl,
        selected_standards=selected_standards,
        output_path=output_benchmark,
        target_count=600,
        conflict_ratio=0.25
    )

if __name__ == "__main__":
    main()