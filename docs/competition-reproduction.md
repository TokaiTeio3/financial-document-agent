# 竞赛提交版复现说明

本文件保留 AFAC 赛题四“金融长文本 Agent 的动态记忆压缩与高效问答挑战”的竞赛提交版运行说明。命令在仓库根目录执行。公开仓库只有源码，不附带官方数据、完整标注、历史答案、报告 PDF 或验收产物；需自行准备有授权的输入。赛后 Demo 请看 [首页](../README.md)。

- 团队名称：北宇治吹奏部
- B 榜封榜排名：第 18 名
- B 榜总分：93.11（最高分 93.50）
- 原始方法报告未再分发；主要设计与复盘整理在 [设计笔记](design-notes.md) 与 [真实失败分析](bad-cases.md)。

系统从官方原始数据冷启动，完成 PDF/TXT/HTML 解析、文本标准化、文档定位、动态检索、证据胶囊整理、Qwen 推理、受限计算、答案格式化和调用级审计，最终生成 `answer.csv`。正式流程不使用 embedding、向量数据库、深度学习 rerank、非 Qwen 推理模型、历史答案记忆或榜单反馈。

## 1. 一次性答案生成

赛事验收使用 `generate_answer.sh`。该脚本从指定官方输入目录或 ZIP 开始，依次执行预处理和完整 B 榜推理，不读取或拼接仓库中已有的 `answer.csv`。

```bash
export DASHSCOPE_API_KEY="your-api-key"

bash generate_answer.sh \
  --input /path/to/input \
  --output /path/to/output \
  --workers 4
```

运行完成后，最终提交文件位于：

```text
<output>/answer.csv
```

同一输出目录还包含 `evidence.json`、`api_calls.jsonl`、`evidence_tables.json`、`run_metrics.json`、`run/` 和完整预处理工作目录 `work/preprocess/`，可用于复核答案来源与完整解题过程。输出目录必须不存在或为空。

Windows PowerShell 可直接调用等价入口：

```powershell
$env:DASHSCOPE_API_KEY="your-api-key"

python generate_answer.py `
  --input D:\path\to\input `
  --output D:\path\to\output `
  --workers 4
```

## 2. 两个独立复现流程

预处理复现与 B 榜性能复现使用不同入口，二者不会隐式互相调用：

| 流程 | 入口 | 输入 | 输出 | 调用 Qwen API |
|---|---|---|---|---|
| 预处理复现 | `run_preprocess.py` / `preprocess.sh` | 官方 ZIP 或原始 `raw_dataset` | `processed_data`、分页文件和报告 | 否 |
| B 榜性能复现 | `run_reproduce.py` / `reproduce.sh` | B 榜题目和既有 `processed_data` | `answer.csv`、日志和证据链 | 是 |

### 2.1 环境要求

- Python 3.11 或更高版本
- Windows 或 Linux
- 推荐 8 核 CPU、16 GB 内存、20 GB 可用磁盘
- B 榜性能复现需要稳定访问 DashScope，并设置 `DASHSCOPE_API_KEY`
- 预处理复现不需要 API Key
- 不需要本地 GPU

### 2.2 预处理复现

该流程只验证 PDF/TXT/HTML 解析、文本标准化和合并文档生成，不生成答案。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

python run_preprocess.py `
  --input D:\path\to\raw_dataset `
  --output D:\path\to\preprocess_output `
  --workers 4
```

`--input` 可以指向官方 ZIP，也可以指向含有 `questions/` 和 `raw/` 的目录，或其上层包含 `raw_dataset/` 的目录。输出为：

```text
<preprocess_output>/
  parsed_pages/
  normalized_pages/
  processed_data/
  preprocess_report.json
  raw_dataset/              # 仅当输入为 ZIP 时生成
```

### 2.3 B 榜性能复现

该流程不执行原始文档解析，只读取自行准备的 `processed_data/` 和题目。其时间口径对应“索引就绪后的全量推理阶段”。只有按下述结构装配了本地输入包时，才使用 `--input .`：

```powershell
$env:DASHSCOPE_API_KEY="your-api-key"

python run_reproduce.py `
  --input . `
  --output D:\path\to\performance_output `
  --workers 4
```

也可以显式指定两个输入目录：

```powershell
python run_reproduce.py `
  --questions-dir D:\path\to\questions `
  --processed-data D:\path\to\processed_data `
  --output D:\path\to\performance_output `
  --workers 4
```

`--input` 指向的性能复现包根目录必须包含：

```text
<input>/
  data/raw_dataset/questions/
  processed_data/
```

### 2.4 Linux

```bash
chmod +x preprocess.sh reproduce.sh

# 预处理复现：不需要 DASHSCOPE_API_KEY
./preprocess.sh \
  --input /path/to/raw_dataset \
  --output /path/to/preprocess_output \
  --workers 4

# B 榜性能复现：在最终提交包根目录执行
export DASHSCOPE_API_KEY="your-api-key"
./reproduce.sh \
  --input . \
  --output /path/to/performance_output \
  --workers 4
```

Shell 脚本会创建或复用 `.venv`，支持 `--python` 和 `--skip-install`；性能复现还支持 `--config`。

## 3. 性能复现输出

输出目录必须不存在或为空，避免旧结果与新结果混合。一次 B 榜性能复现生成：

```text
<performance_output>/
  answer.csv
  evidence.json
  api_calls.jsonl
  evidence_tables.json
  run_metrics.json
  runtime_index/
  run/
    answer.csv
    results_b_reproduction.json
    run.log.jsonl
    manifest.json
```

主要产物说明：

| 产物 | 用途 |
|---|---|
| `answer.csv` | 一次全量运行直接生成的九列提交文件 |
| `evidence.json` | 按 qid 聚合的证据链 |
| `api_calls.jsonl` | 完整 API 请求、响应、参数、错误和原始 usage |
| `evidence_tables.json` | 调用级证据编号到文档、页码、摘录和哈希的映射 |
| `run_metrics.json` | 路由、调用与 Token 聚合指标 |
| `manifest.json` | 本次运行的配置、状态和结果清单 |

## 4. 方法概述

完整方法包含预处理与性能复现两个阶段；提交脚本将两个阶段显式分开：

```text
预处理复现：
  官方原始输入
  -> PDF/TXT/HTML 按页解析与标准化
  -> processed_data + preprocess_report.json

B 榜性能复现：
  B 榜题目 + processed_data
  -> 文档别名、实体、年份和正文词面定位
  -> 文档级定位、页级评分、页内证据截取
  -> 证据胶囊与证据充足度检查
  -> 联合裁决、逐选项核验或受限计算
  -> 一致性检查与本地格式化
  -> answer.csv、证据链和审计日志
```

### 4.1 数据处理

- PDF 使用 PyMuPDF4LLM 按物理页转换为 Markdown。
- TXT 依次尝试 UTF-8-SIG、UTF-8 和 GB18030 编码。
- HTML 清理脚本与样式节点，并按标题、段落、列表和表格恢复换行。
- 标准化阶段处理全角空格、异常字符间距、跨行断句、重复页眉页脚和目录行。
- 合并文档保留页码与源文件标记，保证证据能够回到原始页面复核。

### 4.2 原子约束与文档定位

题目和选项被拆解为主体、期间、指标、单位、数值、条件、例外和量词等可核验约束。系统为每份文档建立由文档编号、标题、公司简称、产品名等组成的别名集合，并综合标题匹配、别名匹配、实体一致性、正文词面命中、文档编号和年份关系完成定位。所有实体均从当前题目和当前文档动态获得。

### 4.3 动态检索与证据胶囊

检索采用“文档级定位 - 页级评分 - 页内证据窗”三级粒度。页级评分以 BM25 为基础，并叠加实体精确匹配、低频约束锚定和相邻页扩展；不使用 embedding、向量相似度或隐藏 reranker。

高分页面在进入模型前被整理为证据胶囊，每条胶囊保留材料类型、`doc_id`、`page_id`、文档别名、检索分数和局部原文。表格证据同时保留表头、单位、指标行和比较期；条款证据保留条件、例外和相邻上下文。

系统以证据数量覆盖率、文档覆盖率、数值检查点覆盖率和条件约束覆盖率计算证据充足度，阈值为 `0.62`。证据不足时，只针对缺失约束生成 Query Repair 或 Bridge 查询，已经确定的选项不重复检索。

### 4.4 自适应推理与受限计算

路由器根据证据共享关系、是否跨主体或跨年度、是否含否定或免责条件，以及是否需要确定性计算，选择：

- 联合裁决：多个选项共享主体、期间和证据页面；
- 逐选项核验：选项涉及独立主体、独立期间或独立条件；
- 受限计算：问题涉及公式、换算、比较或排序。

模型裁决输出 `verdict`、`reasoning` 和 `evidence_ids`；计算路线另返回 `program_result`。Qwen 负责识别业务口径、变量和计算方式，本地 AST 白名单执行器负责确定性运算，实现业务语义判断与数值执行分离。

### 4.5 答案与审计

Qwen 不直接修改 `answer.csv`。本地格式器负责选项顺序、百分号、日期、小数位和空值等输出契约。每次调用的完整 messages、原始响应、usage、检索结果、路由、计算结果和错误信息均写入日志。

`EvidenceTable` 通过 `call_id`、`route` 和 `option` 把模型调用与 `doc_id`、`page_id`、证据摘录、分数和 SHA-256 内容哈希连接起来，使答案、证据和 Token 统计能够逐题追溯。

## 5. 关键配置

正式入口读取 `config/config.ultra.yaml` 的 `reproduction` profile：

- 服务：阿里云百炼 DashScope OpenAI 兼容接口
- API 地址：`https://dashscope.aliyuncs.com/compatible-mode/v1`
- 模型：`qwen3.7-max-2026-06-08`
- `temperature=0.0`
- `top_p=1.0`
- `seed=20260726`
- `enable_thinking=false`
- 单次请求超时 180 秒，最多重试 2 次
- 默认 4 个 workers
- 固定模型不可用时禁止静默切换

API 返回的 `prompt_tokens`、`completion_tokens` 和 `total_tokens` 是唯一 Token 统计来源。查询规划、裁决、格式修复、失败与重试调用均进入日志；代码不会估算、缩放或人工补齐 Token。

固定模型、采样参数、Prompt、检索排序和运行配置用于降低随机波动，但 `seed` 不代表服务端绝对确定性保证。

## 6. 参考运行

参考运行对应 2026-07-26 的一次 100 题冷启动全量推理，时间口径为索引就绪后的推理阶段。

- 开始：2026-07-26 22:24:54.701（UTC+08:00）
- 结束：2026-07-26 22:49:08.997（UTC+08:00）
- 推理耗时：24 分 14.296 秒
- workers：4
- API 调用：248 次成功，0 次错误
- Prompt Token：540,656
- Completion Token：32,149
- Total Token：572,805

| 领域 | 题数 | Prompt | Completion | Total | 题均 Token |
|---|---:|---:|---:|---:|---:|
| 金融合同 | 20 | 80,075 | 4,878 | 84,953 | 4,247.7 |
| 上市公司报告 | 20 | 143,759 | 9,109 | 152,868 | 7,643.4 |
| 保险条款 | 20 | 91,305 | 6,199 | 97,504 | 4,875.2 |
| 监管法规 | 20 | 144,616 | 6,208 | 150,824 | 7,541.2 |
| 行业研究 | 20 | 80,901 | 5,755 | 86,656 | 4,332.8 |
| 合计 | 100 | 540,656 | 32,149 | 572,805 | 5,728.1 |

文档解析耗时取决于 PDF 数量、页数、版式复杂度和磁盘速度，不将其写成固定值。API 阶段耗时还可能受网络、服务端排队和重试影响。

## 7. 代码结构

| 目录 | 职责 |
|---|---|
| `src/preprocess/` | PDF/TXT/HTML 解析、页面标准化与合并文档生成 |
| `src/finqa_agent/core/` | 数据结构、原子约束、领域口径和 Prompt 策略 |
| `src/finqa_agent/retrieval/` | 文档定位、查询计划、BM25、证据预算和证据追溯 |
| `src/finqa_agent/reasoning/` | 联合裁决、逐选项核验、补证和答案一致性 |
| `src/finqa_agent/calculation/` | 程序生成、受限 Python 执行和计算结果校准 |
| `src/finqa_agent/runtime/` | 运行流水线、API、输入输出、日志和运行契约 |
| `scripts/` | 验收、证据构建、参数控制台和提交包装配 |

一次性答案生成入口为 `generate_answer.sh` / `generate_answer.py`，预处理入口为 `run_preprocess.py`，B 榜性能复现入口为 `run_reproduce.py`；Agent 入口为 `src/finqa_agent/agent.py`，底层命令行入口为 `python -m src.finqa_agent.run`。

## 8. 验收

### 8.1 代码与测试

```powershell
python -m compileall -q src scripts generate_answer.py run_preprocess.py run_reproduce.py
python -m unittest discover -s tests -v
```

### 8.2 结果完整性

```powershell
python -m scripts.validate_submission_readiness `
  --config config/config.ultra.yaml `
  --profile reproduction `
  --results <output>\run\results_b_reproduction.json
```

### 8.3 提交包验收

这是比赛专用的历史工具，不是公开源码包的验收命令。需要额外准备原数据、结果、日志和原方法 PDF；直接对本仓库运行会因缺少这些材料失败。公开源码安全检查请运行 `python scripts/audit_public_release.py`。

```powershell
python -m scripts.validate_submission_bundle .
```

最终提交包同时包含 `SUBMISSION_MANIFEST.json` 和 `SUBMISSION_AUDIT.json`，用于核对文件哈希、原始材料覆盖、参考运行与 Token 汇总。

## 9. 合规与复现边界

- 正式运行从本轮官方原始输入重新定位文档并生成答案。
- `use_prior_api_answer_memory=false`、`cross_round_memory=false`、`runtime_entity_memory=false`。
- 不按 qid、固定实体、测试集顺序、榜单反馈或答案分布路由。
- 不使用 embedding、向量数据库、非 Qwen 模型或外部隐蔽 Agent 参与正式答题、排序、校验或 reasoning 生成。
- 不手工挑选、改写或补全模型输出，不人工修改 Token usage。
- API Key、`.env`、临时目录和运行缓存不进入提交包。
- 最终答案必须能够由源码、配置、完整 Prompt、API 原始响应、证据链和日志复核。

## 10. 最终提交清单

下列是当时的比赛提交包结构，不是本公开仓库的文件清单。

提交目录的关键内容包括：

```text
answer.csv
README.md
requirements.txt
generate_answer.py
generate_answer.sh
run_preprocess.py
run_reproduce.py
preprocess.sh
reproduce.sh
config/
src/
scripts/
data/
processed_data/
logs/
docs/
evidence.json
SUBMISSION_MANIFEST.json
SUBMISSION_AUDIT.json
队伍信息表.docx
```

`README.md`、方法 PDF、`reproduction` 配置、参考运行日志和审计清单共同构成最终复现口径。
