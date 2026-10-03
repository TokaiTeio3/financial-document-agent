from app.skills.base import build_search_tool


def create_tool(registry, top_k):
    return build_search_tool(registry, "insurance", "search_insurance_documents", "检索保险条款，适合责任范围、免责、等待期、保费、现金价值、理赔与退保问题。", top_k)
