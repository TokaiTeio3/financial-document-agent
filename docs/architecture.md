# 重构版：沿着一次请求读代码

以下路径相对于 `financial_qa_agent/`。这份说明描述当前代码，不描述还没实现的生产架构。

## 从接口到状态图

`app/main.py` 在启动时创建 `CorpusRegistry` 和 `FinancialQAAgent`。前端向 `POST /api/chat/stream` 发送问题，接口调用 `agent.stream()`，将图节点产生的增量变成 NDJSON。

`app/agent/graph.py` 的 `_compile()` 定义四个主节点：

```text
START → prepare → decompose → Send(worker × N) → finalize → END
```

`prepare` 提取锚点与领域提示。`decompose` 通过 Max 的 `submit_decomposition` tool call 提交 1–6 个子问题。`_dispatch_workers()` 为每个子问题创建 `Send`。每个 worker 最多执行一个检索调用，然后由 Flash 做结构化压缩。所有 worker 返回后，`finalize` 使用 Flash 汇总，不再把整份原始检索上下文送给 Max。

演示模式用一个原问题子任务和确定性排序替代模型调用，验证同一张图的输入输出。它不模拟真实模型的拆分质量、模型用量或准确率。

## 状态如何合并

状态定义在 `app/agent/state.py`，结果 Schema 在 `app/schemas.py`。

| 状态字段 | 谁写入 | 合并方式 / 含义 |
|---|---|---|
| `question` | 请求 | 原始问题，不由 worker 覆盖 |
| `keywords`, `candidate_domains` | prepare | 启发式检索与领域提示 |
| `subquestions` | decompose | 完整子问题计划 |
| `subquestion` | Send | 当前 worker 的局部任务 |
| `subquestion_results` | 各 worker | `operator.add` 追加结构化结果 |
| `evidence` | 各 worker | `operator.add` 追加压缩证据 |
| `token_usage` | 模型调用节点 | `operator.add` 追加每次 usage |
| `trace` | 各节点 | `operator.add` 追加公开执行摘要 |
| `final_answer` | finalize | 结构化答案字典 |

追加式 reducer 解决并行返回时的覆盖问题，暂不负责事实去重、冲突裁决或跨 worker 操作数共享。状态中仍保留兼容字段，例如 `messages`、`calculations`、`refined_query`；不能据此认为当前图已有计算循环或补检索闭环。

## 锚点与 candidate_domains 在哪里

`app/retrieval/bm25.py` 的 `query_anchors()` 是纯程序逻辑：

- 提取英文标识、数字和百分数。
- 按连接词拆分中文连续片段，保留 2–18 字候选。
- 补充连续四字片段，去重后按长度排序，最多取 24 项。

它不是实体识别模型，也不能保证每个片段都有业务含义。BM25 分词使用数字、英文和中文二/三元字符。完整锚点在标题中命中时给予较高加分，在正文中命中时给予较小加分。

`app/agent/graph.py` 的 `DOMAIN_RULES` / `_route_domains()` 根据问题包含的领域关键词计分，保留最多两个正分领域，写入 `candidate_domains`。这只是初始提示。真实模型模式下，Planner 可以在子问题里提交 `preferred_domain`；若提交了合法领域，Flash 仅能调用该领域的检索工具，否则可以从全部五个工具中选择。模型不调用工具时还有确定性领域兜底。

## 领域 SKILL 的实际含义

五个目录位于 `app/skills/`，包含能力说明 `SKILL.md` 与 Python 工具工厂。`app/skills/base.py` 将它们封装为 LangChain `StructuredTool`：输入为 `query`、可选 `document_ids` 和公开 `reason`；输出为统一的 `SearchHit`。

这里的 SKILL 是项目内的领域能力封装。当前运行时通过 Python 注册工具及 description，并没有自动发现或动态加载任意外部 SKILL，也没有将所有 `SKILL.md` 自动注入模型上下文。

| 领域目录 | 工具 |
|---|---|
| financial_reports | search_financial_reports |
| financial_contracts | search_financial_contracts |
| insurance | search_insurance_documents |
| regulatory | search_regulatory_documents |
| research | search_research_reports |

## 从页面到检索片段

`app/retrieval/corpus.py` 的 `chunk_file()` 读取 Markdown 页标记，为当前页添加前页尾部和后页头部，然后按字符长度滑动切块。默认页间 overlap 240 字、chunk 900 字、页内 overlap 120 字。`scripts/build_page_index.py` 将片段写到 `runtime_index/`，manifest 记录参数。

这是一种降低边缘漏检的机制，仍有两个局限：字符边界可能切断表格；片段的主 `page` 标记属于中心页，重叠引用尚未逐段保留真实页码。参数匹配只校验索引构建配置，不是完整的数据内容版本检查，文档修改后需要手动重建。

## 压缩、兜底与前端

Flash 压缩输出 `SubQuestionResult`：子问题、`answer_hint`、是否可回答、原文引用与限制。结构校验失败会进行有限重试；若引用为空或解析失败，可回退到 BM25 Top-3 摘要。Schema 保证结构，不保证 quote 是原文的精确子串，也不保证引用支持结论，目前还缺硬性事实与引用校验。

前端 `app/web/static/app.js` 读取 NDJSON，展示 `trace`、`token_usage`、`answer`、`error` 和结束状态。模型调用的输入、输出 Token 分别汇总到模型名；并行 worker 的轨迹按节点完成情况到达，不保证是严格串行“思考顺序”。现在是节点完成后更新，不是模型内部逐 Token 的实时进度。

`app/tools/calculator.py` 定义 `calculate_finance`，基于 AST 算术执行。它在工具映射中，但 worker 只绑定检索工具，finalize 也没有计算工具调用。因此计算主节点属于后续工作，不能用竞赛版的计算能力替当前 Demo 背书。
