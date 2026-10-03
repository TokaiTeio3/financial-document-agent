from app.skills.base import build_search_tool


def create_tool(registry, top_k):
    return build_search_tool(registry, "research", "search_research_reports", "检索券商与行业研究报告，适合行业观点、公司预测、估值、风险提示和投资逻辑。", top_k)
