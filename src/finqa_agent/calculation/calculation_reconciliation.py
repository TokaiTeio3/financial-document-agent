from __future__ import annotations

import re
from typing import Dict, List, Mapping


def ratio_point_tail_from_reasoning(reasoning: str, question: str) -> str:
    """根据原始比率重新计算复合答案末尾的百分点数值。"""
    if not (
        "原始" in question
        and re.search(r"占比|比例", question)
        and "百分点" in question
        and re.search(r"答案格式.{0,100}[；;]", question)
    ):
        return ""
    ratios = re.findall(
        r"(?<![\d.])(\d[\d,]*(?:\.\d+)?)\s*/\s*"
        r"(\d[\d,]*(?:\.\d+)?)(?![\d.])",
        reasoning,
    )
    if len(ratios) < 2:
        return ""
    counts: Dict[tuple[str, str], int] = {}
    for ratio in ratios:
        counts[ratio] = counts.get(ratio, 0) + 1
    recurrent = sorted(
        counts,
        key=lambda item: (-counts[item], ratios.index(item)),
    )
    selected = (
        recurrent[:2]
        if len(recurrent) >= 2 and counts[recurrent[1]] >= 2
        else ratios[-2:]
    )
    try:
        (left_num, left_den), (right_num, right_den) = selected
        left = float(left_num.replace(",", "")) / float(left_den.replace(",", ""))
        right = float(right_num.replace(",", "")) / float(right_den.replace(",", ""))
    except (ValueError, ZeroDivisionError):
        return ""
    return f"{abs(left - right) * 100:.2f}"


def percent_rate_tail_from_reasoning(reasoning: str, question: str) -> str:
    """将小数形式的比率输入转换为不带百分号的百分数结果。"""
    compact_question = re.sub(r"\s+", "", question)
    if not (
        "不带%" in compact_question
        and "资产负债率" in compact_question
        and "权益乘数" in compact_question
        and re.search(r"资产收益率|ROA", compact_question, re.I)
        and re.search(r"净资产收益率|ROE", compact_question, re.I)
    ):
        return ""
    liability = re.search(
        r"(?:资产负债率\s*(?:为|=)?|asset_liability_ratio\s*=)\s*"
        r"(?:(0\.\d+)|(\d+(?:\.\d+)?)\s*/\s*100)",
        reasoning,
        re.I,
    )
    roe = re.search(
        r"(?:(?:加权平均)?净资产收益率\s*(?:为|=)?|"
        r"(?:roe|return_on_equity)\s*=)\s*"
        r"(?:(0\.\d+)|(\d+(?:\.\d+)?)\s*/\s*100)",
        reasoning,
        re.I,
    )
    if not liability or not roe:
        return ""
    try:
        liability_value = float(liability.group(1) or liability.group(2)) / (
            1.0 if liability.group(1) else 100.0
        )
        roe_value = float(roe.group(1) or roe.group(2)) / (
            1.0 if roe.group(1) else 100.0
        )
        multiplier = 1.0 / (1.0 - liability_value)
        rate_points = roe_value / multiplier * 100.0
    except (ValueError, ZeroDivisionError):
        return ""
    return f"{rate_points:.2f}"


def rank_table_anchors(
    anchors: List[Dict[str, str]],
    question: str,
    limit: int,
) -> List[Dict[str, str]]:
    """按照当前题目要求的指标对结构化行排序。"""
    metric_terms = [
        term
        for term in (
            "EBITDA", "毛利率", "营业收入", "净利润", "现金流量", "资产负债率",
            "每股", "每10股", "分红", "同比", "占比", "比例",
        )
        if term in question
    ]
    subject_terms = re.findall(r"[\u4e00-\u9fff]{2,10}(?:板块|业务|收入|利润)", question)

    def score(anchor: Mapping[str, str]) -> tuple[int, int]:
        text = str(anchor.get("context", "")) + " " + str(anchor.get("row", ""))
        value = 20 * sum(term in text for term in metric_terms)
        value += 8 * sum(term in text for term in subject_terms)
        if "EBITDA" in question.upper() and "EBITDA" in text.upper():
            value += 40
        if "原始金额" in question and re.search(r"百万元|千元|万元", text):
            value += 24
        value += 2 * len(re.findall(r"\d+(?:\.\d+)?%?", str(anchor.get("row", ""))))
        return value, -len(text)

    return sorted(anchors, key=score, reverse=True)[: max(1, limit)]


def rounded_intermediate_from_reasoning(reasoning: str, question: str) -> str:
    """修正与其明确中间结果相矛盾的最终数值。"""
    precision_match = re.search(r"保留([一二三四])位小数", question + reasoning)
    precision_map = {"一": 1, "二": 2, "三": 3, "四": 4}
    precision = precision_map.get(precision_match.group(1), 2) if precision_match else 2
    candidates = re.findall(
        rf"(?:=|≈)\s*([-+]?\d+\.\d{{{precision + 1},}})"
        r"\s*(?:亿元|万元|元|%|个百分点)?",
        reasoning,
    )
    if not candidates:
        return ""
    from decimal import Decimal, ROUND_HALF_UP

    value = Decimal(candidates[-1]).quantize(
        Decimal(1).scaleb(-precision),
        rounding=ROUND_HALF_UP,
    )
    return f"{value:.{precision}f}"


def ordered_subject_answer_from_reasoning(reasoning: str, question: str) -> str:
    """保留按要求格式排列两个主体顺序的 API 结论。"""
    if not re.search(r"答案格式.{0,120}>.{0,80}[；;]", question, re.S):
        return ""
    conclusion_pattern = re.compile(
        r"(?:(?:答案|更正|修正|输出|结果)[^：:\n]{0,12}[：:]|应为)\s*"
        r"([A-Za-z\u4e00-\u9fff（）()·]{2,30})\s*>\s*"
        r"([A-Za-z\u4e00-\u9fff（）()·]{2,30})\s*[；;]\s*"
        r"([-+]?\d+(?:\.\d+)?)"
    )
    candidates = conclusion_pattern.findall(reasoning)
    for left, right, value in reversed(candidates):
        left = left.strip()
        right = right.strip()
        if left in question and right in question:
            return f"{left}>{right}；{value}"
    return ""
