from __future__ import annotations

import json

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.retrieval.corpus import CorpusRegistry
from app.schemas import Domain


class SearchInput(BaseModel):
    query: str = Field(description="保留主体、年份、指标或条款名称的精炼检索式")
    document_ids: list[str] | str | None = Field(default=None, description="已知文档 ID 时用于限定检索范围")
    reason: str = Field(description="可展示给用户的简短调用理由，不要包含隐藏思维链")


def build_search_tool(
    registry: CorpusRegistry,
    domain: Domain,
    name: str,
    description: str,
    top_k: int,
) -> StructuredTool:
    def search(query: str, document_ids: list[str] | str | None = None, reason: str = "") -> list[dict]:
        del reason
        if isinstance(document_ids, str):
            try:
                parsed = json.loads(document_ids)
                document_ids = parsed if isinstance(parsed, list) else [str(parsed)]
            except json.JSONDecodeError:
                document_ids = [item.strip() for item in document_ids.split(",") if item.strip()]
        return [hit.model_dump() for hit in registry.get(domain).search(query, top_k, document_ids)]

    search.__name__ = name
    return StructuredTool.from_function(
        func=search,
        name=name,
        description=description,
        args_schema=SearchInput,
    )
