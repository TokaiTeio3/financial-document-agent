from __future__ import annotations

import json
import re
import uuid
from collections.abc import AsyncIterator
from typing import Any, TypeVar

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field

from app.agent.prompts import FINAL_PROMPT
from app.agent.state import AgentState
from app.config import Settings
from app.retrieval.bm25 import query_anchors
from app.retrieval.corpus import CorpusRegistry
from app.schemas import Domain, RelevantEvidence, StructuredAnswer, SubQuestion, SubQuestionResult
from app.skills import create_retrieval_tools
from app.tools.calculator import calculate_finance


DOMAIN_RULES: dict[str, tuple[str, ...]] = {
    "financial_reports": ("年报", "财报", "营业收入", "净利润", "现金流", "研发投入", "资产负债"),
    "financial_contracts": ("募集说明书", "债券", "合同", "发行", "票面利率", "担保", "违约"),
    "insurance": ("保险", "保费", "被保险人", "免责", "理赔", "现金价值", "等待期"),
    "regulatory": ("证监会", "监管", "办法", "规定", "条例", "指引", "披露", "处罚"),
    "research": (
        "研报", "研究报告", "行业", "评级", "目标价", "盈利预测", "投资建议",
        "低利率", "资产荒", "资产配置", "高股息", "FVOCI", "券商自营", "分红险",
    ),
}

TOOL_BY_DOMAIN = {
    "financial_reports": "search_financial_reports",
    "financial_contracts": "search_financial_contracts",
    "insurance": "search_insurance_documents",
    "regulatory": "search_regulatory_documents",
    "research": "search_research_reports",
}

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class DecompositionSubmission(BaseModel):
    subquestions: list[SubQuestion] = Field(min_length=1, max_length=6)
    reason: str = Field(description="一句可公开的拆分理由")


@tool(args_schema=DecompositionSubmission)
def submit_decomposition(subquestions: list[SubQuestion], reason: str) -> str:
    """提交已经拆分好的、每个可由一次文档检索回答的原子子问题计划。"""
    del subquestions, reason
    return "accepted"


def _route_domains(question: str) -> list[str]:
    scored = [
        (domain, sum(1 for keyword in keywords if keyword.lower() in question.lower()))
        for domain, keywords in DOMAIN_RULES.items()
    ]
    scored.sort(key=lambda item: item[1], reverse=True)
    return [domain for domain, score in scored if score > 0][:2]


def _public_trace(step: int, node: str, title: str, summary: str, details: dict | None = None) -> dict:
    return {
        "step": step,
        "node": node,
        "status": "completed",
        "title": title,
        "summary": summary,
        "details": details or {},
    }


def _usage_from_message(message: BaseMessage | None, model: str, stage: str) -> dict[str, Any] | None:
    if message is None:
        return None
    usage = getattr(message, "usage_metadata", None) or {}
    response_metadata = getattr(message, "response_metadata", None) or {}
    raw_usage = response_metadata.get("token_usage") or response_metadata.get("usage") or {}
    input_tokens = usage.get("input_tokens", raw_usage.get("prompt_tokens", 0)) or 0
    output_tokens = usage.get("output_tokens", raw_usage.get("completion_tokens", 0)) or 0
    total_tokens = usage.get("total_tokens", raw_usage.get("total_tokens", input_tokens + output_tokens)) or 0
    return {
        "model": model,
        "stage": stage,
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "total_tokens": int(total_tokens),
    }


class FinancialQAAgent:
    def __init__(self, settings: Settings, registry: CorpusRegistry):
        self.settings = settings
        self.registry = registry
        self.retrieval_tools = create_retrieval_tools(registry, settings.search_top_k)
        self.tools = self.retrieval_tools + [calculate_finance]
        self.tool_map = {item.name: item for item in self.tools}
        self.planner_model = None
        self.worker_model = None
        self.planner_with_tool = None
        self.worker_with_tools = None
        if not settings.demo_mode:
            from langchain_openai import ChatOpenAI

            common = {
                "api_key": settings.api_key,
                "base_url": settings.base_url,
                "temperature": 0,
                "timeout": settings.request_timeout,
                "max_retries": 1,
            }
            planner_name = settings.planner_model or settings.answer_model or settings.model
            self.planner_model = ChatOpenAI(model=planner_name, **common)
            self.worker_model = ChatOpenAI(model=settings.model, **common)
            # Qwen Max thinking mode supports function calling but rejects forced tool_choice.
            self.planner_with_tool = self.planner_model.bind_tools([submit_decomposition])
            self.worker_with_tools = self.worker_model.bind_tools(self.retrieval_tools)
        self.graph = self._compile()

    def _compile(self):
        builder = StateGraph(AgentState)
        builder.add_node("prepare", self._prepare)
        builder.add_node("decompose", self._decompose)
        builder.add_node("worker", self._worker)
        builder.add_node("finalize", self._finalize)
        builder.add_edge(START, "prepare")
        builder.add_edge("prepare", "decompose")
        builder.add_conditional_edges("decompose", self._dispatch_workers, ["worker"])
        builder.add_edge("worker", "finalize")
        builder.add_edge("finalize", END)
        return builder.compile()

    def _prepare(self, state: AgentState) -> dict:
        question = state["question"].strip()
        keywords = query_anchors(question)[:10]
        domains = _route_domains(question)
        return {
            "messages": [HumanMessage(content=question)],
            "keywords": keywords,
            "candidate_domains": domains,
            "subquestions": [],
            "subquestion_results": [],
            "evidence": [],
            "calculations": [],
            "token_usage": [],
            "trace": [_public_trace(
                1,
                "prepare",
                "解析问题",
                f"识别出 {len(keywords)} 个检索锚点，初始领域提示为 {', '.join(domains) or '未确定'}。",
                {"keywords": keywords, "candidate_domains": domains},
            )],
        }

    def _fallback_subquestions(self, state: AgentState) -> list[SubQuestion]:
        domain = state.get("candidate_domains", [None])[0] if state.get("candidate_domains") else None
        return [SubQuestion(id="sq-1", question=state["question"], preferred_domain=domain, purpose="直接检索原问题")]

    def _decompose(self, state: AgentState) -> dict:
        usage_events: list[dict[str, Any]] = []
        reason = "演示模式将原问题作为单个可检索子问题。"
        if self.settings.demo_mode:
            subquestions = self._fallback_subquestions(state)
        else:
            planner_name = self.settings.planner_model or self.settings.answer_model or self.settings.model
            prompt = (
                "你是金融问答规划器。必须调用 submit_decomposition 工具。"
                "把问题拆成1到6个原子子问题，每个子问题必须能够通过一次文档检索回答。"
                "保留主体、文档名、年份、指标、条款和选项语义；不要回答问题。"
                "preferred_domain 只能从 financial_reports、financial_contracts、insurance、regulatory、research 中选择。"
                "purpose 只写一句公开目的，不输出隐藏思维链。"
            )
            message = self.planner_with_tool.invoke([
                SystemMessage(content=prompt),
                HumanMessage(content=state["question"]),
            ])
            usage = _usage_from_message(message, planner_name, "decompose")
            if usage:
                usage_events.append(usage)
            subquestions = []
            if message.tool_calls:
                args = message.tool_calls[0].get("args", {})
                reason = str(args.get("reason", "Max 已生成原子检索计划。"))
                raw_items = args.get("subquestions", [])
                if isinstance(raw_items, str):
                    try:
                        raw_items = json.loads(raw_items)
                    except json.JSONDecodeError:
                        raw_items = []
                for index, item in enumerate(raw_items[:6], 1):
                    try:
                        parsed = SubQuestion.model_validate(item)
                        subquestions.append(parsed.model_copy(update={"id": parsed.id or f"sq-{index}"}))
                    except Exception:
                        continue
            if not subquestions:
                subquestions = self._fallback_subquestions(state)
                reason = "Max 未返回有效计划，使用原问题作为单个子问题兜底。"
        payload = [item.model_dump() for item in subquestions]
        return {
            "subquestions": payload,
            "token_usage": usage_events,
            "trace": [_public_trace(
                2,
                "decompose",
                "Max 拆分问题",
                f"拆分为 {len(payload)} 个可单次检索回答的子问题。{reason}",
                {
                    "model": None if self.settings.demo_mode else (self.settings.planner_model or self.settings.answer_model),
                    "subquestions": payload,
                },
            )],
        }

    @staticmethod
    def _dispatch_workers(state: AgentState):
        return [
            Send("worker", {
                "question": state["question"],
                "subquestion": item,
                "candidate_domains": state.get("candidate_domains", []),
            })
            for item in state.get("subquestions", [])
        ]

    @staticmethod
    def _invoke_structured(
        model,
        schema: type[SchemaT],
        messages: list[BaseMessage],
        retries: int,
    ) -> tuple[SchemaT | None, int, str | None]:
        """Compatibility helper for schema-validating providers and focused unit tests."""
        runner = model.with_structured_output(schema, include_raw=True)
        retry_messages = list(messages)
        last_error: str | None = None
        for attempt in range(1, retries + 2):
            try:
                result = runner.invoke(retry_messages)
                parsed = result.get("parsed") if isinstance(result, dict) else result
                if parsed is not None:
                    return parsed, attempt, None
                last_error = str(result.get("parsing_error") or "结构化结果为空")
            except Exception as exc:
                last_error = str(exc)
            if attempt <= retries:
                retry_messages.append(HumanMessage(content=(
                    "上一份 JSON 未通过 Schema 校验，请只返回符合 Schema 的 JSON。"
                    f"错误：{last_error[:400]}"
                )))
        return None, retries + 1, last_error

    def _invoke_worker_structured(
        self,
        schema: type[SchemaT],
        messages: list[BaseMessage],
        stage: str,
        retries: int | None = None,
        model=None,
    ) -> tuple[SchemaT | None, list[dict[str, Any]], str | None, int]:
        target_model = model or self.worker_model
        runner = target_model.with_structured_output(schema, include_raw=True)
        retry_messages = list(messages)
        usage_events: list[dict[str, Any]] = []
        last_error: str | None = None
        retry_count = self.settings.structured_call_retries if retries is None else retries
        for attempt in range(1, retry_count + 2):
            try:
                result = runner.invoke(retry_messages)
                raw = result.get("raw") if isinstance(result, dict) else None
                usage = _usage_from_message(raw, self.settings.model, stage)
                if usage:
                    usage_events.append(usage)
                parsed = result.get("parsed") if isinstance(result, dict) else result
                if parsed is not None:
                    return parsed, usage_events, None, attempt
                last_error = str(result.get("parsing_error") or "结构化结果为空")
            except Exception as exc:
                last_error = str(exc)
            if attempt <= retry_count:
                retry_messages.append(HumanMessage(content=(
                    "上一份 JSON 未通过 Schema 校验，请只返回符合 Schema 的 JSON。"
                    f"错误：{last_error[:400]}"
                )))
        return None, usage_events, last_error, retry_count + 1

    def _select_tool(self, subquestion: SubQuestion) -> tuple[AIMessage, list[dict[str, Any]]]:
        preferred = subquestion.preferred_domain
        tools = self.retrieval_tools
        if preferred in TOOL_BY_DOMAIN:
            tools = [self.tool_map[TOOL_BY_DOMAIN[preferred]]]
        worker = self.worker_model.bind_tools(tools)
        message = worker.invoke([
            SystemMessage(content=(
                "你是检索执行器。针对这个原子子问题选择并调用一次最合适的检索工具。"
                "query 必须保留主体、年份、指标或条款；reason 只写一句公开理由。不要回答问题。"
            )),
            HumanMessage(content=subquestion.question),
        ])
        usage = _usage_from_message(message, self.settings.model, "retrieval_planning")
        return message, [usage] if usage else []

    def _compress_evidence(
        self,
        subquestion: SubQuestion,
        raw_hits: list[dict[str, Any]],
    ) -> tuple[SubQuestionResult, list[dict[str, Any]], str | None]:
        if self.settings.demo_mode:
            return self._deterministic_compression(subquestion, raw_hits), [], None

        compact_hits = [
            {
                "document_id": item.get("document_id"),
                "source": item.get("source"),
                "page": item.get("page"),
                "score": item.get("score"),
                "content": str(item.get("content", ""))[:1400],
            }
            for item in raw_hits
        ]
        parsed, usage, error, _ = self._invoke_worker_structured(
            SubQuestionResult,
            [
                SystemMessage(content=(
                    "你是金融证据压缩器。仅保留能够直接回答子问题的原文，输出符合 Schema 的 JSON 对象。"
                    "quote 必须忠实摘自输入片段，不得补写；删除背景、目录和无关段落。"
                    "answer_hint 是基于证据的一句简短结论，不输出隐藏思维链。"
                )),
                HumanMessage(content=json.dumps({
                    "subquestion_id": subquestion.id,
                    "subquestion": subquestion.question,
                    "retrieved_chunks": compact_hits,
                }, ensure_ascii=False)),
            ],
            "evidence_compression",
        )
        if parsed is not None:
            parsed = parsed.model_copy(update={
                "subquestion_id": subquestion.id,
                "subquestion": subquestion.question,
            })
            if not parsed.evidence and raw_hits:
                deterministic = self._deterministic_compression(subquestion, raw_hits)
                parsed = parsed.model_copy(update={
                    "evidence": deterministic.evidence,
                    "limitations": [*parsed.limitations, "Flash 未返回引用，已保留 BM25 Top-3 片段作为可审计兜底。"],
                })
            return parsed, usage, error
        fallback = self._deterministic_compression(subquestion, raw_hits)
        fallback = fallback.model_copy(update={
            "limitations": [f"Flash 结构化压缩失败，使用确定性 Top-3 兜底：{(error or '未知错误')[:180]}"],
        })
        return fallback, usage, error

    @staticmethod
    def _deterministic_compression(
        subquestion: SubQuestion,
        raw_hits: list[dict[str, Any]],
    ) -> SubQuestionResult:
        evidence = [
            RelevantEvidence(
                document_id=item["document_id"],
                source=item["source"],
                page=item.get("page"),
                quote=re.sub(r"\s+", " ", item.get("content", ""))[:260],
                relevance="BM25 与锚点排序靠前",
            )
            for item in raw_hits[:3]
        ]
        return SubQuestionResult(
            subquestion_id=subquestion.id,
            subquestion=subquestion.question,
            answer_hint="已保留与子问题最相关的检索证据。",
            answerable=bool(evidence),
            evidence=evidence,
        )

    def _worker(self, state: AgentState) -> dict:
        subquestion = SubQuestion.model_validate(state["subquestion"])
        usage_events: list[dict[str, Any]] = []
        if self.settings.demo_mode:
            domain = subquestion.preferred_domain or (_route_domains(subquestion.question) or ["financial_reports"])[0]
            tool_name = TOOL_BY_DOMAIN[domain]
            args = {"query": subquestion.question, "document_ids": None, "reason": "演示模式确定性检索"}
        else:
            message, selection_usage = self._select_tool(subquestion)
            usage_events.extend(selection_usage)
            if message.tool_calls:
                call = message.tool_calls[0]
                tool_name = call["name"]
                args = call.get("args", {})
            else:
                domain = subquestion.preferred_domain or (_route_domains(subquestion.question) or ["financial_reports"])[0]
                tool_name = TOOL_BY_DOMAIN[domain]
                args = {"query": subquestion.question, "document_ids": None, "reason": "Flash 未调用工具，使用领域兜底"}
        tool = self.tool_map.get(tool_name)
        try:
            result = tool.invoke(args) if tool else {"error": f"未知工具：{tool_name}"}
        except Exception as exc:
            result = {"error": str(exc)}
        raw_hits = result if isinstance(result, list) else []
        compressed, compression_usage, compression_error = self._compress_evidence(subquestion, raw_hits)
        usage_events.extend(compression_usage)
        compressed_dict = compressed.model_dump()
        return {
            "subquestion_results": [compressed_dict],
            "evidence": [item.model_dump() for item in compressed.evidence],
            "token_usage": usage_events,
            "trace": [_public_trace(
                2 + int(re.sub(r"\D", "", subquestion.id) or "1"),
                "worker",
                f"Flash 处理子问题 {subquestion.id}",
                f"调用 {tool_name} 获得 {len(raw_hits)} 段，压缩后保留 {len(compressed.evidence)} 条证据。",
                {
                    "subquestion": subquestion.model_dump(),
                    "tool": tool_name,
                    "query": args.get("query"),
                    "raw_hit_count": len(raw_hits),
                    "retained_evidence": compressed_dict["evidence"],
                    "answer_hint": compressed.answer_hint,
                    "compression_fallback": compression_error is not None,
                },
            )],
        }

    def _demo_answer(self, state: AgentState) -> StructuredAnswer:
        results = state.get("subquestion_results", [])
        evidence = [item for result in results for item in result.get("evidence", [])]
        return StructuredAnswer(
            answer="已完成问题拆分、逐子问题检索和证据压缩。当前为演示模式。",
            key_points=[result.get("answer_hint", "") for result in results if result.get("answer_hint")],
            citations=[
                {
                    "document_id": item["document_id"],
                    "source": item["source"],
                    "page": item.get("page"),
                    "quote": item["quote"][:180],
                }
                for item in evidence[:8]
            ],
            confidence="medium" if evidence else "low",
            limitations=[] if evidence else ["未检索到相关证据。"],
        )

    def _finalize(self, state: AgentState) -> dict:
        usage_events: list[dict[str, Any]] = []
        error = None
        if self.settings.demo_mode:
            answer = self._demo_answer(state)
        else:
            target_model = getattr(self, "worker_model", None) or getattr(self, "answer_model", None)
            answer, usage_events, error, attempts = self._invoke_worker_structured(
                StructuredAnswer,
                [
                    SystemMessage(content=FINAL_PROMPT),
                    HumanMessage(content=json.dumps({
                        "question": state["question"],
                        "subquestion_results": state.get("subquestion_results", []),
                    }, ensure_ascii=False)),
                ],
                "final_answer",
                retries=self.settings.final_answer_retries,
                model=target_model,
            )
            if answer is None:
                if not state.get("subquestion_results"):
                    answer = StructuredAnswer(
                        answer="模型多次返回了无法通过结构校验的结果，请稍后重试。",
                        confidence="low",
                        limitations=[f"最后错误：{(error or '未知错误')[:200]}"],
                    )
                else:
                    answer = self._demo_answer(state).model_copy(update={
                        "limitations": [f"Flash 最终结构化失败，展示压缩证据兜底：{(error or '未知错误')[:200]}"],
                    })
        if self.settings.demo_mode:
            attempts = 0
        payload = answer.model_dump()
        return {
            "final_answer": payload,
            "token_usage": usage_events,
            "trace": [_public_trace(
                100,
                "finalize",
                "生成答案",
                f"基于 {len(state.get('subquestion_results', []))} 个子问题的压缩证据生成结构化回答。",
                {
                    "model": None if self.settings.demo_mode else self.settings.model,
                    "confidence": payload["confidence"],
                    "citation_count": len(payload["citations"]),
                    "fallback_used": error is not None,
                    "attempts": attempts,
                },
            )],
        }

    def _agent(self, state: AgentState) -> dict:
        """Legacy refined-query adapter retained for compatibility; the new graph uses workers."""
        domain = state.get("candidate_domains", ["financial_reports"])[0]
        tool_name = TOOL_BY_DOMAIN.get(domain, "search_financial_reports")
        message = AIMessage(content="", tool_calls=[{
            "name": tool_name,
            "args": {
                "query": state.get("refined_query") or state["question"],
                "document_ids": None,
                "reason": "使用证据评估产生的新检索词。",
            },
            "id": f"legacy-{uuid.uuid4().hex[:10]}",
            "type": "tool_call",
        }])
        return {"messages": [message]}

    async def stream(self, question: str) -> AsyncIterator[dict[str, Any]]:
        totals: dict[str, dict[str, int]] = {}
        initial: AgentState = {"question": question}
        async for update in self.graph.astream(initial, stream_mode="updates"):
            for node, delta in update.items():
                for usage in delta.get("token_usage", []):
                    model = usage["model"]
                    bucket = totals.setdefault(model, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "calls": 0})
                    bucket["input_tokens"] += usage["input_tokens"]
                    bucket["output_tokens"] += usage["output_tokens"]
                    bucket["total_tokens"] += usage["total_tokens"]
                    bucket["calls"] += 1
                    yield {
                        "type": "token_usage",
                        "data": {
                            "latest": usage,
                            "models": totals,
                            "grand_total": sum(item["total_tokens"] for item in totals.values()),
                        },
                    }
                for event in delta.get("trace", []):
                    yield {"type": "trace", "node": node, "data": event}
                if "final_answer" in delta:
                    yield {"type": "answer", "data": delta["final_answer"]}
