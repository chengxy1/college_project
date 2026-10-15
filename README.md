# EBR-CFR 工程实现项目 (Evidence-Based Reranking with Counterfactual Reasoning)

硕士学位论文核心重排序算法工程原型。

## 目录结构
- `config/`: 系统及算法参数配置 (`settings.yaml`)
- `data/raw_documents/`: 原始非结构化工程规范与报告文本
- `data/test_cases/`: 结构化评估与测试样本 (`c30_concrete_eval.json`)
- `src/models/`: 数据模型与元数据契约 (`schemas.py`)
- `src/modules/`: 
  - `structure_modeler.py`: 层级一 证据结构建模
  - `cf_reasoner.py`: 层级二 反事实效用推理
- `src/pipeline.py`: 双层流水线调度与过滤决策
- `tests/`: 自动化单元测试与指标验证
- `main.py`: 快速运行入口

## 快速上手
1. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
2. 运行自动化测试：
   ```bash
   pytest tests/ -v -s
   ```
3. 运行完整重排序流水线：
   ```bash
   python main.py
   ```
