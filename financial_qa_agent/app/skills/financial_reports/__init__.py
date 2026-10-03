from app.skills.base import build_search_tool


def create_tool(registry, top_k):
    return build_search_tool(registry, "financial_reports", "search_financial_reports", "检索上市公司年度报告与财务报表，适合收入、利润、现金流、研发投入和跨年度指标对比。", top_k)
