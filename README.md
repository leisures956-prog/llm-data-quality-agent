# LLM Web Data Quality Pipeline

面向大语言模型预训练数据建设的数据质量处理项目。项目把原始网页数据转换为可审计、可评估、可采样的数据资产，重点覆盖网页正文抽取、特征计算、重复检测、质量评分、人工审核校准、数据分布分析和采样权重分配。

这个项目对应的不是一个简单的“网页爬虫”，而是一条可解释的数据治理 pipeline：每条数据都会保留处理结果、质量分数、决策原因和必要的审计信息，便于回答“为什么保留、为什么拒绝、规则改动带来了什么影响”。

## 当前项目状态

当前已完成 `benchmark_v1` 和 HTML Benchmark v2 的端到端验证。

Benchmark v1 的正式决策流程为 `calibrated_scoring_v1`；HTML Benchmark v2 当前使用的规则版本为 `quality_rules_v3_content_density_aggregation`，新增了内容密度、导航页和文章聚合页识别。

已实现：

- 原始网页数据 schema 检查与数据集清单生成
- HTML 正文抽取与清洗
- 文本质量特征计算
- 精确重复检测
- 近似重复检测
- 规则化质量评分与 `keep / review / reject` 决策
- 短但有信息量文本的校准处理
- 人工审核集评估、误杀/误保统计
- 修复、恢复和规则消融实验
- 语言、主题、长度和质量分布统计
- 采样资格判断与采样权重分配
- Benchmark 汇总报告生成
- 真实网页 HTML Benchmark v2 验证
- 内容密度优先评分
- 导航页和目录页识别
- Blog、News、Insider 等文章聚合页识别
- 不覆盖旧结果的版本化 Benchmark 运行
- Benchmark 版本差异对比报告
- review 数量预算控制

当前仍需加强：

- 更大规模、更多网页类型的人工审核集
- 真实网页中的导航、广告、页脚和 Cookie 模板验证
- 更可靠的领域/主题分类器
- 自动化回归测试和 CI 运行
- 生产环境下的数据来源、许可证和溯源管理

## Pipeline 流程

```text
原始 JSONL 网页数据
        |
        v
数据集 schema 检查
        |
        v
HTML 正文抽取与清洗
        |
        v
质量特征计算
        |
        v
精确去重
        |
        v
近似去重（字符 3-5 gram）
        |
        v
基础质量评分
        |
        v
校准评分与正式决策
        |
        +--> 人工审核评估
        +--> 规则消融实验
        +--> 语言/主题/长度分布
        +--> 采样计划与权重
        |
        v
Benchmark 汇总报告
```

正式决策含义：

- `keep`：通过质量筛选，可以进入候选训练数据
- `review`：存在重复、长度或质量不确定性，需要人工复核
- `reject`：质量不足、模板内容过多或重复风险较高，不进入候选训练数据

## Benchmark v1 结果

### 数据集规模

- 原始数据：48 条
- 语言：中文 27 条，英文 21 条
- 数据集检查：48 条全部有效，0 条错误
- 精确去重后：47 条
- 近似重复：1 组，涉及 2 条文档
- 近似重复报告：`doc_001 <-> doc_007`，相似度 `0.7846`

### 正式决策分布

在 47 条精确去重后的数据上：

| 决策 | 数量 | 含义 |
| --- | ---: | --- |
| keep | 41 | 可进入候选训练数据 |
| review | 3 | 需要人工复核 |
| reject | 3 | 暂不进入候选训练数据 |

当前 `review` 文档为：`doc_001`、`doc_007`、`doc_008`。

### 人工审核集评估

人工审核集共 16 条，其中人工标签为 `keep` 的 12 条、`reject` 的 4 条。

校准后的结果：

| 指标 | 结果 |
| --- | ---: |
| Accuracy | 0.8750 |
| Precision | 1.0000 |
| Recall | 0.8333 |
| F1 | 0.9091 |
| 硬误杀数量 | 0 |
| 误保数量 | 0 |

其中，`doc_008` 虽然正文较短，但人工判断其内容完整且有明确的数据质量知识。校准规则将其从直接拒绝调整为 `review`，避免把短但有价值的内容直接误杀。

### 规则消融结论

当前对比了 4 个策略：

| 策略 | review | reject | keep | 误杀 | 误保 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `baseline_v1` | 3 | 3 | 10 | 0 | 0 |
| `tolerant_review_v2` | 4 | 2 | 10 | 0 | 0 |
| `strict_review_v2` | 2 | 4 | 10 | 1 | 0 |
| `without_duplicate_review` | 1 | 3 | 12 | 0 | 1 |

在 Benchmark v1 人工审核集上，正式流程保留近似重复复核，并采用 `quality_rules_v1`。选择规则时不只看 F1，还重点关注误杀和误保风险；HTML Benchmark v2 则使用升级后的 `quality_rules_v3_content_density_aggregation`。

## HTML Benchmark v2：聚合页规则回归结果

本实验使用 34 条真实网页 HTML 数据，对比聚合页规则加入前后的结果。

| 指标 | `quality_rules_v2` | `quality_rules_v3` | 变化 |
| --- | ---: | ---: | ---: |
| 总文档数 | 34 | 34 | 0 |
| keep | 27 | 26 | -1 |
| reject | 7 | 8 | +1 |
| review | 0 | 0 | 0 |
| 决策变化 | - | 1 | +1 |
| 分数变化 | - | 2 | +2 |
| 新增文档 | - | 0 | 0 |
| 删除文档 | - | 0 | 0 |

### 规则触发结果

- `html_v2_031 / Python Insider`
  - 原分数：45
  - 新分数：20
  - 原决策：`reject`
  - 新决策：`reject`
  - 新增原因：疑似博客或文章聚合页

- `html_v2_034 / MDN Blog`
  - 原分数：90
  - 新分数：65
  - 原决策：`keep`
  - 新决策：`reject`
  - 新增原因：疑似博客或文章聚合页

### 实验结论

聚合页规则只改变了 1 条正式决策，没有改变文档数量，也没有影响其他文档的决策结果。该规则成功识别了由多个文章摘要组成的 Blog 聚合页面，降低了这类页面进入候选训练数据的风险。

版本对比报告：

`reports/benchmark_compare_density_v2_vs_aggregation_v3.json`

新版本结果：

- `data/processed/html_benchmark_v2_aggregation_v3`
- `reports/html_benchmark_v2_aggregation_v3`

## 输入数据格式

原始数据使用 JSONL，每行一个网页文档。必需字段：

```json
{
  "id": "doc_001",
  "url": "https://example.com/article",
  "domain": "technology",
  "language": "zh",
  "html": "<html>...</html>"
}
```

推荐同时提供：

```json
{
  "source": "source_name",
  "collected_at": "2026-01-01T00:00:00Z",
  "license": "明确的数据许可信息"
}
```

当前 schema 名称为 `raw_web_v1`，具体校验逻辑位于 `app/pipeline/dataset_registry.py`。

## 目录说明

```text
app/pipeline/                 核心处理逻辑和基础运行脚本
config/quality_rules.json     quality_rules_v3_content_density_aggregation 规则配置
data/raw/                     原始 JSONL 数据
data/processed/               各阶段处理结果
data/labels/                  人工审核标签
reports/                      统计、评估和审计报告
tools/                        Benchmark、评估、消融和报告工具
tests/                        自动化测试
```

重要产物：

- `data/raw/benchmark_v1.jsonl`：Benchmark v1 原始数据
- `data/labels/benchmark_audit_v1.jsonl`：16 条人工审核标签
- `data/processed/production_decisions_benchmark_v1.jsonl`：正式决策结果
- `reports/benchmark_v1_run_report.json`：Benchmark 汇总报告
- `reports/benchmark_audit_calibrated_summary.json`：校准后人工审核评估
- `reports/ablation_benchmark_audit_summary.json`：规则消融实验结果
- `reports/distribution_benchmark_summary.json`：数据分布统计
- `reports/sampling_benchmark_summary.json`：采样计划统计
- `reports/benchmark_compare_density_v2_vs_aggregation_v3.json`：规则版本对比报告

## 运行环境

项目当前在 Python 3.11 环境 `llm-data-quality311` 中开发和验证。项目不需要 GPU，也不依赖大模型即可运行当前 pipeline。

请先进入项目根目录并激活自己的 Python 环境，再执行下面的命令。

## 测试

运行全部自动化测试：

```powershell
python -m pytest -q
```

当前已验证：`50 passed`。

Windows 环境下如果默认临时目录没有权限，可以使用项目目录下的临时目录：

```powershell
python -m pytest -q `
    --basetemp D:\Codex\llm-data-quality-agent\.pytest_tmp_local
```

## Benchmark v1 运行顺序

以下命令应在项目根目录执行。Benchmark 处理链使用 `data/raw/benchmark_v1.jsonl`，而不是默认的 `sample.jsonl`。

```powershell
$env:DATASET_PATH = "data/raw/benchmark_v1.jsonl"

python -m app.pipeline.run_dataset_check
python -m app.pipeline.run_features
python -m app.pipeline.run_deduplicate
python -m app.pipeline.run_near_deduplicate
python -m app.pipeline.run_scoring

python -m tools.evaluate_benchmark_audit
python -m tools.run_calibrated_benchmark
python -m tools.evaluate_calibrated_benchmark_audit
python -m tools.run_finalize_benchmark
python -m tools.run_distribution_benchmark
python -m tools.run_sampling_benchmark
python -m tools.build_benchmark_report
```

规则消融实验单独运行：

```powershell
python -m tools.run_ablation_benchmark_audit
```

如果后续切换回默认示例数据，可以清除当前 PowerShell 会话中的数据集路径：

```powershell
Remove-Item Env:DATASET_PATH
```

## 注意事项

Benchmark 专用脚本会生成带有 `_benchmark_v1` 后缀的结果，便于与早期通用结果区分。当前部分基础 pipeline 脚本仍会写入 `data/processed/` 下的通用文件，例如 `features.jsonl`、`quality_scored.jsonl` 和 `near_deduplicated.jsonl`。运行前应确认这些文件可以被重新生成，重要结果以 Benchmark 专用文件和 `reports/benchmark_v1_run_report.json` 为准。

不要只依据 `keep` 数量判断规则好坏。数据质量项目需要同时观察：

- 误杀（false rejection）
- 误保（false acceptance）
- 人工复核量
- 不同语言和主题的表现
- 去重和采样策略对数据覆盖的影响

## 下一阶段计划

1. 将 Benchmark 版本对比接入自动化回归测试。
2. 继续增加不同类型的真实网页 HTML 数据，但采用批量收集和分层抽样，不逐条人工处理。
3. 补充数据来源、许可证、采集时间和处理版本等溯源信息。
4. 增加 CI 自动运行测试和规则回归检查。
5. 规则稳定后，再评估是否加入轻量分类器或 LLM scorer，并与当前规则进行可重复的消融对比。
