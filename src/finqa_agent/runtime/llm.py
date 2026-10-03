from __future__ import annotations

import os
import threading
import time
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

from .audit import RunAuditLogger
from ..retrieval.evidence_trace import (
    extract_evidence_table,
    infer_option_from_messages,
    table_fingerprint,
)


_QUESTION_CONTEXT: ContextVar[str] = ContextVar("finqa_question_context", default="")
_ROUTE_CONTEXT: ContextVar[str] = ContextVar("finqa_route_context", default="unclassified")


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0

    def add(self, other: "Usage") -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.total_tokens += other.total_tokens

    def to_dict(self) -> Dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class LLMResult:
    content: str
    usage: Usage
    model: str = ""
    finish_reason: str = ""


class TokenBudgetExceeded(RuntimeError):
    """在可能超出所配置运行预算的调用前抛出。"""


class QwenClient:
    def __init__(
        self,
        model: str,
        api_base: str,
        api_key: Optional[str] = None,
        fallback_model: Optional[str] = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: Optional[int] = None,
        enable_thinking: bool = False,
        token_budget: int = 0,
        timeout_seconds: float = 180.0,
        max_retries: int = 2,
        audit_logger: Optional[RunAuditLogger] = None,
    ) -> None:
        key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not key:
            raise ValueError("Missing DASHSCOPE_API_KEY")
        self.client = OpenAI(
            api_key=key,
            base_url=api_base,
            timeout=timeout_seconds,
            max_retries=max_retries,
        )
        self.model = model
        self.fallback_model = fallback_model or model
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed
        self.enable_thinking = enable_thinking
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.audit = audit_logger
        self.total_usage = Usage()
        self.token_budget = max(0, int(token_budget))
        self._reserved_tokens = 0
        self._usage_lock = threading.Lock()
        self._call_lock = threading.Lock()
        self._call_sequence = 0

    @staticmethod
    def set_question_context(qid: str) -> None:
        _QUESTION_CONTEXT.set(qid)

    @staticmethod
    def set_route_context(route: str) -> None:
        _ROUTE_CONTEXT.set(route or "unclassified")

    def chat(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        temperature: Optional[float] = None,
    ) -> LLMResult:
        started = time.time()
        qid = _QUESTION_CONTEXT.get()
        route = _ROUTE_CONTEXT.get()
        reservation = self._estimate_prompt_tokens(messages) + max_tokens
        self._reserve(reservation)
        try:
            try:
                return self._chat_once(
                    messages, self.model, max_tokens, temperature, started, qid, route, False
                )
            except Exception as exc:
                if not self._should_fallback(exc):
                    raise
                return self._chat_once(
                    messages,
                    self.fallback_model,
                    max_tokens,
                    temperature,
                    started,
                    qid,
                    route,
                    True,
                )
        finally:
            with self._usage_lock:
                self._reserved_tokens = max(0, self._reserved_tokens - reservation)

    def _chat_once(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        max_tokens: int,
        temperature: Optional[float],
        started: float,
        qid: str,
        route: str,
        fallback: bool,
    ) -> LLMResult:
        with self._call_lock:
            self._call_sequence += 1
            call_id = self._call_sequence
        effective_temperature = self.temperature if temperature is None else temperature
        parameters: Dict[str, Any] = {
            "temperature": effective_temperature,
            "top_p": self.top_p,
            "max_tokens": max_tokens,
            "enable_thinking": self.enable_thinking,
            "timeout_seconds": self.timeout_seconds,
            "client_max_retries": self.max_retries,
        }
        if self.seed is not None:
            parameters["seed"] = self.seed
        if self.audit is not None:
            evidence_rows = extract_evidence_table(messages)
            if evidence_rows:
                table_id = f"{qid}:{call_id}:{route}"
                self.audit.event(
                    "evidence_table",
                    qid=qid,
                    call_id=call_id,
                    route=route,
                    option=infer_option_from_messages(messages),
                    table_id=table_id,
                    table_sha256=table_fingerprint(evidence_rows),
                    row_count=len(evidence_rows),
                    rows=evidence_rows,
                )
            self.audit.event(
                "api_request",
                qid=qid,
                call_id=call_id,
                route=route,
                model=model,
                fallback=fallback,
                parameters=parameters,
                messages=messages,
            )
        request: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": effective_temperature,
            "top_p": self.top_p,
            "max_tokens": max_tokens,
            "extra_body": {"enable_thinking": self.enable_thinking},
        }
        if self.seed is not None:
            request["seed"] = self.seed
        try:
            response = self.client.chat.completions.create(**request)
        except Exception as exc:
            if self.audit is not None:
                self.audit.event(
                    "api_error",
                    qid=qid,
                    call_id=call_id,
                    route=route,
                    model=model,
                    fallback=fallback,
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
            raise
        usage = Usage(
            prompt_tokens=getattr(response.usage, "prompt_tokens", 0),
            completion_tokens=getattr(response.usage, "completion_tokens", 0),
            total_tokens=getattr(response.usage, "total_tokens", 0),
            latency_ms=(time.time() - started) * 1000,
        )
        with self._usage_lock:
            self.total_usage.add(usage)
        choice = response.choices[0]
        result = LLMResult(
            content=choice.message.content or "",
            usage=usage,
            model=getattr(response, "model", model),
            finish_reason=choice.finish_reason or "",
        )
        if self.audit is not None:
            self.audit.event(
                "api_response",
                qid=qid,
                call_id=call_id,
                route=route,
                requested_model=model,
                response_model=result.model,
                fallback=fallback,
                finish_reason=result.finish_reason,
                usage=usage.to_dict(),
                latency_ms=usage.latency_ms,
                raw_output=result.content,
            )
        return result

    @staticmethod
    def _should_fallback(exc: Exception) -> bool:
        if isinstance(exc, APIStatusError):
            return exc.status_code == 429 or exc.status_code >= 500
        return isinstance(exc, (APITimeoutError, APIConnectionError))

    @staticmethod
    def _estimate_prompt_tokens(messages: List[Dict[str, Any]]) -> int:
        # 目标模型下中文文本接近每字符一个 token。
        # 英文字符占比较高的结构化数据或代码更省令牌，因此这里有意高估。
        chars = sum(len(str(message.get("content", ""))) for message in messages)
        return chars + 64 * len(messages)

    def _reserve(self, tokens: int) -> None:
        if not self.token_budget:
            return
        with self._usage_lock:
            projected = self.total_usage.total_tokens + self._reserved_tokens + tokens
            if projected > self.token_budget:
                raise TokenBudgetExceeded(
                    f"token budget would be exceeded: projected={projected}, budget={self.token_budget}"
                )
            self._reserved_tokens += tokens
