from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


Domain = Literal[
    "financial_reports",
    "financial_contracts",
    "insurance",
    "regulatory",
    "research",
]


class DomainDecision(BaseModel):
    domains: list[Domain] = Field(min_length=1, max_length=2)
    reason: str = Field(description="一句可公开的领域判断理由")


class EvidenceAssessment(BaseModel):
    sufficient: bool
    reason: str = Field(description="一句可公开的证据充分性判断")
    refined_query: str | None = Field(default=None, description="证据不足时生成的新检索词")
    recommended_domain: Domain | None = None


class SearchHit(BaseModel):
    domain: Domain
    document_id: str
    source: str
    chunk_id: str
    page: int | None = None
    score: float
    matched_anchors: list[str] = Field(default_factory=list)
    content: str


class Citation(BaseModel):
    document_id: str
    source: str
    page: int | None = None
    quote: str


class StructuredAnswer(BaseModel):
    answer: str = Field(description="直接回答问题，信息不足时明确说明")
    selected_options: list[str] = Field(default_factory=list, description="选择题所选选项字母，非选择题留空")
    final_value: str | None = Field(default=None, description="计算题最终结果，非计算题留空")
    key_points: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    confidence: Literal["low", "medium", "high"] = "medium"
    limitations: list[str] = Field(default_factory=list)

    @field_validator("selected_options", mode="before")
    @classmethod
    def coerce_selected_options(cls, value):
        if value is None:
            return []
        if isinstance(value, str):
            return [letter for letter in value.upper() if letter in "ABCD"]
        return value

    @field_validator("final_value", mode="before")
    @classmethod
    def coerce_final_value(cls, value):
        return None if value is None else str(value)

    @field_validator("key_points", "limitations", mode="before")
    @classmethod
    def coerce_text_lists(cls, value):
        if value is None:
            return []
        if isinstance(value, str):
            return [] if value.strip() in {"", "无", "none", "None"} else [value]
        if isinstance(value, list):
            output = []
            for item in value:
                if isinstance(item, dict):
                    point = item.get("point") or item.get("summary") or item.get("text") or ""
                    evidence = item.get("evidence") or item.get("detail") or ""
                    output.append("：".join(filter(None, (str(point), str(evidence)))))
                else:
                    output.append(str(item))
            return output
        return [str(value)]

    @field_validator("citations", mode="before")
    @classmethod
    def coerce_citations(cls, value):
        return [] if value is None else value

    @field_validator("confidence", mode="before")
    @classmethod
    def coerce_confidence(cls, value):
        mapping = {"低": "low", "中": "medium", "高": "high"}
        return mapping.get(str(value), value)


class SubQuestion(BaseModel):
    id: str
    question: str
    preferred_domain: Domain | None = None
    purpose: str = ""


class RelevantEvidence(BaseModel):
    document_id: str
    source: str
    page: int | None = None
    quote: str
    relevance: str = ""


class SubQuestionResult(BaseModel):
    subquestion_id: str
    subquestion: str
    answer_hint: str = ""
    answerable: bool = True
    evidence: list[RelevantEvidence] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)

    @field_validator("limitations", mode="before")
    @classmethod
    def coerce_result_limitations(cls, value):
        if value is None:
            return []
        return [value] if isinstance(value, str) else value


class ChatRequest(BaseModel):
    question: str = Field(min_length=2, max_length=4000)


class TraceEvent(BaseModel):
    step: int
    node: str
    status: Literal["started", "completed", "failed"]
    title: str
    summary: str
    details: dict[str, Any] = Field(default_factory=dict)
