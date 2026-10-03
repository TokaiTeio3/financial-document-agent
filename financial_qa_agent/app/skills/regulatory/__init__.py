from app.skills.base import build_search_tool


def create_tool(registry, top_k):
    return build_search_tool(registry, "regulatory", "search_regulatory_documents", "检索证监会法规、规章和附件，适合适用条件、义务、期限、程序、禁止性规定和处罚。", top_k)
