from __future__ import annotations

import re
from typing import Sequence


_CARDINALITY = {"两": 2, "二": 2, "三": 3}


def expected_option_cardinality(stem: str) -> int:
    """返回题目明确要求的选项数量，否则返回零。"""
    match = re.search(r"哪([两二三])项|选择([两二三])项", str(stem))
    if not match:
        return 0
    return _CARDINALITY.get(match.group(1) or match.group(2), 0)


def needs_cross_document_research_scope(
    stem: str,
    option_doc_ids: Sequence[str],
    candidate_doc_ids: Sequence[str],
) -> bool:
    """防止通用别名把跨报告比较错误收缩到单份报告。"""
    return bool(
        re.search(
            r"二者|两家|四家|四类|多家|多类|多个行业|"
            r"不同.{0,8}(?:行业|机构|领域)|行业均|趋势都|"
            r"各(?:自|领域)|比较",
            str(stem),
        )
        and len(option_doc_ids) < 2
        and len(candidate_doc_ids) > 1
    )


def is_dense_quantitative_cross_document_comparison(
    stem: str,
    options: Sequence[str],
    named_doc_count: int,
) -> bool:
    """识别适合在一个联合视图中高效处理的表格型比较。

    本函数有意只使用结构信号，不包含题号、实体名称、文档标题或预期答案。
    """
    if named_doc_count < 2:
        return False
    text = "\n".join([str(stem), *[str(item) for item in options]])
    numeric_markers = re.findall(
        r"(?:T\s*\+\s*\d+|\d+(?:\.\d+)?\s*(?:%|年|万元|亿元))",
        text,
        flags=re.IGNORECASE,
    )
    absence_claims = re.findall(
        r"(?:未量化|未披露|未提及|没有披露|仅定性|不涉及)",
        text,
    )
    return len(numeric_markers) >= 5 and len(absence_claims) <= 2
