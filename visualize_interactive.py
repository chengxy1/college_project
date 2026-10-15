import os
from pyvis.network import Network


def generate_interactive_graph(query, candidates, ranked_results, evidence_graph,
                               output_html="evidence_graph_interactive.html"):
    """
    接收流水线的排序结果与证据拓扑图，生成全中文交互式 HTML 关系网络图
    """
    # 建立映射字典：以 doc_id 索引排序打分、元数据和原始文档对象
    score_dict = {r.document.doc_id: (r.cf_score, r.metadata, r.document) for r in ranked_results}

    # 初始化交互式图谱
    net = Network(height="800px", width="100%", bgcolor="#F8FAFC", font_color="#0F172A", directed=False)

    # 遍历节点并注入全中文属性
    for node in evidence_graph.nodes():
        if node in score_dict:
            score, meta, doc = score_dict[node]
            is_conflict = "存在事实/指标冲突" in meta.conflict_tags

            # 节点中文主显示标签 (例如: 国家标准: DOC_001)
            display_label = f"【{doc.source}】\n{node}"

            # 状态着色：红色代表冲突/异常，蓝色代表高分核心依据，灰色代表普通/边缘依据
            if is_conflict:
                color = "#EF4444"
                status_text = "⚠️ 存在指标/规范冲突"
            elif score > 0.6:
                color = "#2563EB"
                status_text = "⭐ 核心支撑依据 (权威)"
            else:
                color = "#64748B"
                status_text = "📄 基础辅助证据"

            # 中文悬停气泡提示 (HTML 格式)
            tooltip = f"""
            <div style="font-family: Microsoft YaHei, sans-serif; font-size: 13px; line-height: 1.6; padding: 5px;">
                <b style="color: {color}; font-size: 14px;">{status_text}</b><br>
                <b>文档编号：</b>{node}<br>
                <b>文档来源：</b>{doc.source}<br>
                <b>反事实边际得分：</b><span style="color:#2563EB; font-weight:bold;">{score:.4f}</span><br>
                <b>基础置信度：</b>{doc.base_confidence:.2f}<br>
                <b>冲突排查标记：</b>{', '.join(meta.conflict_tags)}<br>
                <hr style="margin: 5px 0; border: none; border-top: 1px dashed #CBD5E1;">
                <b>反事实摘要：</b>{meta.cf_summary}<br>
                <b>文档内容摘要：</b>{doc.content[:60]}...
            </div>
            """

            node_size = score * 35 + 15
        else:
            # 未进入排序列表的节点（如被初筛过滤）
            display_label = f"【未收录】\n{node}"
            color = "#CBD5E1"
            tooltip = f"文档编号：{node}（因置信度过低未参与最终重排序）"
            node_size = 10

        net.add_node(
            node,
            label=display_label,
            size=node_size,
            color=color,
            title=tooltip,
            font={"size": 12, "face": "Microsoft YaHei", "color": "#0F172A"}
        )

    # 遍历边并添加中文关联描述
    for u, v, data in evidence_graph.edges(data=True):
        weight = data.get('weight', 0.1)
        edge_title = f"跨文档证据支撑度 (相似权重): {weight:.2f}"
        net.add_edge(
            u, v,
            value=weight,
            title=edge_title,
            color="#60A5FA",
            label=f"关联度: {weight:.2f}",
            font={"size": 10, "face": "Microsoft YaHei", "color": "#475569"}
        )

    # 力导向图物理布局配置
    net.set_options("""
    {
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -3500,
          "centralGravity": 0.3,
          "springLength": 130,
          "springConstant": 0.04
        },
        "minVelocity": 0.75
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 150
      }
    }
    """)

    net.save_graph(output_html)
    print(f"📊 [拓扑图构建完成]: 交互式关系图已保存至 -> {os.path.abspath(output_html)}")