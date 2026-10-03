from __future__ import annotations

from app.retrieval.corpus import CorpusRegistry
from app.skills.financial_contracts import create_tool as contracts_tool
from app.skills.financial_reports import create_tool as reports_tool
from app.skills.insurance import create_tool as insurance_tool
from app.skills.regulatory import create_tool as regulatory_tool
from app.skills.research import create_tool as research_tool


def create_retrieval_tools(registry: CorpusRegistry, top_k: int):
    return [
        reports_tool(registry, top_k),
        contracts_tool(registry, top_k),
        insurance_tool(registry, top_k),
        regulatory_tool(registry, top_k),
        research_tool(registry, top_k),
    ]


__all__ = ["create_retrieval_tools"]
