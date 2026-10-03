"""供生产审计使用的可观测推理等级分类。

该分类只记录既有路由选择，不修改 Prompt、检索、答案或 API 参数。
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Dict, Mapping


@dataclass(frozen=True)
class ReasoningAllocation:
    level: str
    name: str
    route: str
    signals: tuple[str, ...]
    document_count: int
    option_count: int
    numeric_checkpoint_count: int
    configured_max_tokens: int
    may_bridge: bool

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def assign_reasoning_level(
    question: Mapping[str, object],
    *,
    route: str,
    document_count: int,
    profile: Mapping[str, object],
) -> ReasoningAllocation:
    text = "\n".join(
        [
            str(question.get("question", "")),
            *[
                str(value)
                for value in (question.get("options") or {}).values()
            ],
        ]
    )
    answer_format = str(question.get("answer_format", ""))
    option_count = len(question.get("options") or {})
    numeric_count = len(
        re.findall(
            r"(?<!\d)\d+(?:\.\d+)?(?:%|个百分点|万元|亿元|元|年|日|周岁|倍)?",
            text,
        )
    )
    years = set(re.findall(r"20\d{2}", text))
    signals: list[str] = []
    if answer_format == "calc" or "calculation" in route:
        signals.append("deterministic_calculation")
    if document_count >= 2:
        signals.append("cross_document")
    if len(years) >= 3:
        signals.append("multi_year_series")
    if numeric_count >= 4:
        signals.append("dense_numeric_checkpoints")
    if re.search(r"全部|均|无论|仅|唯一|完整|任何", text):
        signals.append("quantifier_scope")
    if re.search(r"免责|除外|例外|但书|不适用|不得|无需", text):
        signals.append("exception_or_negation")
    if re.search(r"比较|排序|高于|低于|增加额|减少额|差值", text):
        signals.append("comparison_or_ranking")
    if "option" in route:
        signals.append("independent_option_binding")
    if "bridge" in route or "review" in route:
        signals.append("evidence_gap_recheck")

    deep = bool(
        answer_format == "calc"
        or "calculation" in route
        or len(years) >= 3
        or numeric_count >= 6
        or document_count >= 3
        or (
            "quantifier_scope" in signals
            and "exception_or_negation" in signals
        )
    )
    analytical = bool(
        option_count >= 2
        or document_count >= 2
        or numeric_count >= 3
        or "comparison_or_ranking" in signals
        or "exception_or_negation" in signals
    )
    if deep:
        level, name = "L3", "deep_verification"
    elif analytical:
        level, name = "L2", "analytical_adjudication"
    else:
        level, name = "L1", "focused_extraction"

    if "calculation" in route:
        configured_max_tokens = 1600 + int(
            profile.get("compact_calc_reasoning_max_tokens", 260)
        )
    elif "option" in route:
        configured_max_tokens = int(
            profile.get("compact_option_max_tokens", 320)
        )
    elif "review" in route or "bridge" in route:
        configured_max_tokens = int(
            profile.get("compact_bridge_max_tokens", 360)
        )
    else:
        configured_max_tokens = int(profile.get("answer_max_tokens", 750))

    return ReasoningAllocation(
        level=level,
        name=name,
        route=route,
        signals=tuple(signals),
        document_count=max(0, int(document_count)),
        option_count=option_count,
        numeric_checkpoint_count=numeric_count,
        configured_max_tokens=configured_max_tokens,
        may_bridge=bool(
            profile.get("compact_joint_unknown_fallback_domains")
            or profile.get("compact_invalid_answer_review")
        ),
    )
