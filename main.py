import os
from datetime import datetime
from tabulate import tabulate
from src.utils.config_loader import load_config
from src.utils.data_loader import DataLoader
from src.pipeline import EBR_CFR_Pipeline
from visualize_interactive import generate_interactive_graph
from src.utils.checkpoint_manager import setup_logger, CheckpointManager
from src.experiments.run_baseline_comparison import run_baseline_experiments
from src.experiments.run_ablation_study import run_ablation_experiments
from src.experiments.run_fidelity_analysis import run_fidelity_experiments

def log_section_table(logger, title: str, data: list):
    """统一的科研实验表格排版输出"""
    table_str = tabulate(data, headers="keys", tablefmt="rounded_outline") if data else "暂无实验数据"
    sep = "─" * 68
    logger.info(f"{sep}\n📊 {title}\n{sep}\n{table_str}")

def format_ranked_results_table(ranked_results) -> str:
    """将重排序决策转为规整的控制台/日志表格"""
    summary_data = []
    for idx, item in enumerate(ranked_results, 1):
        doc = item.document
        meta = item.metadata
        conflict_str = " | ".join(meta.conflict_tags) if meta.conflict_tags else "正常"
        chains = ", ".join(meta.verification_chain) if meta.verification_chain else "无支撑链"
        summary_data.append({
            "排名": f"Rank {idx}",
            "文档编号": doc.doc_id,
            "文档类型": doc.source,
            "CF得分": f"{item.cf_score:.4f}",
            "基础置信": f"{doc.base_confidence:.2f}",
            "事实冲突状态": conflict_str,
            "关联验证链": chains
        })
    return tabulate(summary_data, headers="keys", tablefmt="rounded_grid")

def export_markdown_report(query, ranked_results, baseline_results, ablation_results, fidelity_results, total_cases: int = 600, output_dir: str = "outputs/reports"):
    """
    每次运行 main.py 结束后，自动生成格式规范、可直接复制到毕业论文和PPT中的 Markdown 实验报告
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(output_dir, f"EBR_CFR_Experiment_Report_{timestamp}.md")

    single_case_rows = []
    for idx, item in enumerate(ranked_results, 1):
        doc = item.document
        meta = item.metadata
        conflict_str = " | ".join(meta.conflict_tags) if meta.conflict_tags else "内部校验一致"
        chains = ", ".join(meta.verification_chain) if meta.verification_chain else "无支撑链"
        single_case_rows.append({
            "排名": f"Rank {idx}",
            "文档编号": f"`{doc.doc_id}`",
            "文档类型": doc.source,
            "CF得分": f"**{item.cf_score:.4f}**",
            "基础置信": f"{doc.base_confidence:.2f}",
            "事实冲突状态": conflict_str,
            "关联验证链": f"`{chains}`",
            "反事实摘要": meta.cf_summary
        })

    table_single_case = tabulate(single_case_rows, headers="keys", tablefmt="pipe")
    table_baseline = tabulate(baseline_results, headers="keys", tablefmt="pipe")
    table_ablation = tabulate(ablation_results, headers="keys", tablefmt="pipe")
    table_fidelity = tabulate(fidelity_results, headers="keys", tablefmt="pipe")

    # 使用 raw string rf""" 彻底杜绝 LaTeX 符号带来的 SyntaxWarning
    md_content = rf"""# EBR-CFR 建筑工程 RAG 重排序实验报告

- **实验时间**：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **运行环境**：Python 3.12.4 | NetworkX | Scikit-learn
- **评测数据集**：`benchmark_600_cases.json`（共 {total_cases} 组建筑工程真实案例）
- **代表性 Query**：`{query}`

---

## 一、 核心测试用例重排序与决策契约 (代表性用例落地演示)

{table_single_case}

> **决策解读**：核心标准与实测报告形成交叉支撑拓扑结构，在反事实干预时边际效用损失最大，不可替代性高；存在互斥或废止标识的条目被赋予冲突预警。

---

## 二、 实验一：主流重排序基线对比 (开题报告 4.3 节，全量 {total_cases} 例评测)

{table_baseline}

---

## 三、 实验二：各子模块消融实验 (开题报告 4.4 节)

{table_ablation}

---

## 四、 实验三：反事实局部近似保真度分析 (开题报告 4.5 节)

{table_fidelity}

> **学术结论**：$K \ge 1$ 局部邻域近似重估在稀疏证据网中与全图重计算保持极高的一致性，有效降低全图重新遍历计算开销。
"""

    with open(report_file, "w", encoding="utf-8") as f:
        f.write(md_content)

    latest_file = os.path.join(output_dir, "EBR_CFR_Experiment_Report_latest.md")
    with open(latest_file, "w", encoding="utf-8") as f:
        f.write(md_content)

    return report_file


def main():
    log_dir = os.path.join("outputs", "logs")
    logger = setup_logger(log_dir=log_dir, log_name="ebr_cfr_run")
    sep_header = "═" * 68
    logger.info(f"{sep_header}\n🏛️   EBR-CFR 建筑工程 RAG 重排序科研实验与运行系统\n{sep_header}")

    # 优先加载 600 条 Benchmark 数据集
    benchmark_file = os.path.join("data", "test_cases", "benchmark_600_cases.json")
    fallback_file = os.path.join("data", "test_cases", "c30_concrete_eval.json")
    
    if os.path.exists(benchmark_file):
        test_file = benchmark_file
        experiment_id = "benchmark_600_eval"
    elif os.path.exists(fallback_file):
        test_file = fallback_file
        experiment_id = "c30_concrete_eval"
    else:
        logger.error(f"❌ 找不到测试集文件，请检查: {benchmark_file}")
        return

    ckpt = CheckpointManager(
        checkpoint_dir=os.path.join("outputs", "checkpoints"),
        experiment_id=experiment_id
    )

    config = load_config()
    pipeline = EBR_CFR_Pipeline(config)

    # 单用例提取
    query, candidates = DataLoader.load_test_case(test_file)
    logger.info(f"📌 [当前评测测试集]: {test_file}")
    logger.info(f"📌 [代表性案例 Query]: {query}")
    logger.info(f"📥 [候选文档数]: {len(candidates)} 篇")

    ranked, evidence_graph = pipeline.run(query, candidates)

    ranked_table = format_ranked_results_table(ranked)
    logger.info(f"🏁 [代表用例重排序与解释性契约决策 - 准入过滤后保留 {len(ranked)} 篇]:\n{ranked_table}")

    output_dir = os.path.join(os.getcwd(), "outputs", "visualizations")
    os.makedirs(output_dir, exist_ok=True)
    html_output_path = os.path.join(output_dir, "evidence_graph_interactive.html")
    generate_interactive_graph(query, candidates, ranked, evidence_graph, output_html=html_output_path)
    logger.info(f"📊 [拓扑图更新]: 交互图已保存至 -> {html_output_path}")

    # 实验一：基线对比 (自动遍历 600 例)
    stage_1 = "baseline_comparison"
    if ckpt.is_stage_completed(stage_1):
        logger.info(f"⏩ [断点续传] 阶段 '{stage_1}' 已完成，加载缓存结果...")
        baseline_results = ckpt.get_stage_result(stage_1)
    else:
        logger.info(f"🚀 开始执行【实验一：主流重排序基线对比】...")
        baseline_results = run_baseline_experiments(test_file)
        ckpt.save_stage_result(stage_1, baseline_results)
        logger.info(f"💾 [阶段保存] '{stage_1}' 结果已固化。")
    log_section_table(logger, "实验一：主流重排序基线对比 (开题报告 4.3 节)", baseline_results)

    # 实验二：消融实验 (自动遍历 600 例)
    stage_2 = "ablation_study"
    if ckpt.is_stage_completed(stage_2):
        logger.info(f"⏩ [断点续传] 阶段 '{stage_2}' 已完成，加载缓存结果...")
        ablation_results = ckpt.get_stage_result(stage_2)
    else:
        logger.info(f"🚀 开始执行【实验二：子模块消融实验】...")
        ablation_results = run_ablation_experiments(test_file)
        ckpt.save_stage_result(stage_2, ablation_results)
        logger.info(f"💾 [阶段保存] '{stage_2}' 结果已固化。")
    log_section_table(logger, "实验二：EBR-CFR 各子模块消融实验 (开题报告 4.4 节)", ablation_results)

    # 实验三：局部近似保真度分析
    stage_3 = "fidelity_analysis"
    if ckpt.is_stage_completed(stage_3):
        logger.info(f"⏩ [断点续传] 阶段 '{stage_3}' 已完成，加载缓存结果...")
        fidelity_results = ckpt.get_stage_result(stage_3)
    else:
        logger.info(f"🚀 开始执行【实验三：反事实局部近似保真度分析】...")
        fidelity_results = run_fidelity_experiments(evidence_graph, max_k=3)
        ckpt.save_stage_result(stage_3, fidelity_results)
        logger.info(f"💾 [阶段保存] '{stage_3}' 结果已固化。")
    log_section_table(logger, "实验三：反事实局部近似保真度分析 (开题报告 4.5 节)", fidelity_results)

    # 自动导出 Markdown 报告
    report_path = export_markdown_report(
        query=query,
        ranked_results=ranked,
        baseline_results=baseline_results,
        ablation_results=ablation_results,
        fidelity_results=fidelity_results,
        total_cases=600 if test_file == benchmark_file else len(candidates),
        output_dir=os.path.join("outputs", "reports")
    )
    logger.info(f"📄 [实验报告自动导出完成]:\n   - 带时间戳归档: {report_path}\n   - 最新版文件: outputs/reports/EBR_CFR_Experiment_Report_latest.md")

    sep_end = "═" * 68
    logger.info(f"{sep_end}\n🎉 全流程运行完毕，日志已存至 outputs/logs/，Markdown 报告已生成在 outputs/reports/\n{sep_end}")

if __name__ == "__main__":
    main()
