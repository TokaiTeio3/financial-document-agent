from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    # 用户输入与对话上下文
    question: str
    messages: Annotated[list[AnyMessage], add_messages]

    # Agent 的显式工作状态（可序列化、可观测）
    keywords: list[str]
    candidate_domains: list[str]
    domain_reason: str
    tool_rounds: int
    subquestions: list[dict[str, Any]]
    subquestion: dict[str, Any]
    subquestion_results: Annotated[list[dict[str, Any]], operator.add]
    evidence: Annotated[list[dict[str, Any]], operator.add]
    calculations: Annotated[list[dict[str, Any]], operator.add]
    evidence_sufficient: bool
    evidence_feedback: str
    refined_query: str | None
    token_usage: Annotated[list[dict[str, Any]], operator.add]

    # 面向前端的安全执行轨迹，不保存或暴露隐藏思维链
    trace: Annotated[list[dict[str, Any]], operator.add]
    final_answer: dict[str, Any]
    error: str | None
