from app.skills.base import build_search_tool


def create_tool(registry, top_k):
    return build_search_tool(registry, "financial_contracts", "search_financial_contracts", "检索债券募集说明书与金融合同，适合发行要素、权利义务、利率、期限、违约和担保条款。", top_k)
