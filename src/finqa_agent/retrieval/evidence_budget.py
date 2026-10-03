from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence


_NUMBER_PATTERN = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?%?")
_HIGH_RISK_PATTERNS = {
    "calculation": re.compile(r"计算|比例|增长率|变动率|合计|差额|排序|保留.*小数|百分"),
    "universal_quantifier": re.compile(r"无论|均应|全部|一律|任何|所有|必然|绝不"),
    "condition_or_exception": re.compile(r"除外|除非|但书|前提|条件|仅当|不得|不包括|豁免"),
    "temporal": re.compile(r"生效|施行|截至|届满|工作日|年度|期末|次日|期间"),
    "comparison": re.compile(r"相比|高于|低于|最多|最少|最大|最小|由高到低|分别"),
    "multi_evidence": re.compile(r"分别|各自|综合|多份|横向|同时|以及|且"),
}


@dataclass(frozen=True)
class EvidenceBudget:
    mode: str
    tier: str
    risk_score: int
    risk_signals: tuple[str, ...]
    base_top_k: int
    recommended_top_k: int
    base_excerpt_chars: int
    recommended_excerpt_chars: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceSufficiency:
    score: float
    sufficient: bool
    signals: tuple[str, ...]
    hit_count: int
    distinct_doc_count: int
    numeric_checkpoint_coverage: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def plan_evidence_budget(
    *,
    stem: str,
    option: str,
    candidate_doc_count: int,
    bound_doc_count: int,
    retrieval_quality: Mapping[str, object],
    base_top_k: int,
    base_excerpt_chars: int,
    mode: str = "shadow",
) -> EvidenceBudget:
    """为单个选项规划有界且不依赖具体实体的证据预算。

    控制器只检查结构风险信号，绝不包含题号、文档实体、答案或由评测标签派生的规则。
    """
    text = f"{stem}\n{option}"
    signals = [
        name for name, pattern in _HIGH_RISK_PATTERNS.items() if pattern.search(text)
    ]
    score = len(signals)
    if len(_NUMBER_PATTERN.findall(text)) >= 2:
        score += 1
        signals.append("multiple_numeric_checkpoints")
    if candidate_doc_count > 1 and bound_doc_count != 1:
        score += 2
        signals.append("cross_document_binding")
    reasons = [str(item) for item in retrieval_quality.get("reasons", [])]
    if reasons:
        score += min(2, len(reasons))
        signals.append("retrieval_repair_signal")

    if score >= 6:
        tier, top_k_scale, chars_scale = "expanded", 1.5, 1.30
    elif score >= 3:
        tier, top_k_scale, chars_scale = "balanced", 1.25, 1.15
    else:
        tier, top_k_scale, chars_scale = "lean", 1.0, 0.90

    recommended_top_k = max(base_top_k, round(base_top_k * top_k_scale))
    recommended_excerpt_chars = max(240, round(base_excerpt_chars * chars_scale))
    return EvidenceBudget(
        mode=mode if mode in {"shadow", "active"} else "shadow",
        tier=tier,
        risk_score=score,
        risk_signals=tuple(dict.fromkeys(signals)),
        base_top_k=base_top_k,
        recommended_top_k=recommended_top_k,
        base_excerpt_chars=base_excerpt_chars,
        recommended_excerpt_chars=recommended_excerpt_chars,
    )


def assess_evidence_sufficiency(
    *,
    option: str,
    evidence_texts: Sequence[str],
    evidence_doc_ids: Sequence[str] = (),
    expected_top_k: int,
    expected_doc_count: int,
    threshold: float = 0.62,
) -> EvidenceSufficiency:
    """评估召回证据是否覆盖选项级决策检查点。"""
    joined = "\n".join(evidence_texts)
    hit_count = len(evidence_texts)
    signals: list[str] = []

    hit_score = min(1.0, hit_count / max(1, expected_top_k))
    if hit_score < 1:
        signals.append("hit_budget_not_filled")

    distinct_doc_count = (
        len(set(evidence_doc_ids))
        if evidence_doc_ids
        else min(hit_count, max(1, expected_doc_count))
    )
    doc_score = min(1.0, distinct_doc_count / max(1, expected_doc_count))
    if doc_score < 1:
        signals.append("document_coverage_low")

    checkpoints = tuple(dict.fromkeys(_NUMBER_PATTERN.findall(option)))
    if checkpoints:
        present = sum(checkpoint in joined for checkpoint in checkpoints)
        numeric_score = present / len(checkpoints)
        if numeric_score < 1:
            signals.append("numeric_checkpoint_missing")
    else:
        numeric_score = 1.0

    constraint_patterns = [
        pattern for name, pattern in _HIGH_RISK_PATTERNS.items()
        if name in {"universal_quantifier", "condition_or_exception", "temporal"}
        and pattern.search(option)
    ]
    if constraint_patterns:
        constraint_score = sum(bool(pattern.search(joined)) for pattern in constraint_patterns)
        constraint_score /= len(constraint_patterns)
        if constraint_score < 1:
            signals.append("constraint_evidence_missing")
    else:
        constraint_score = 1.0

    score = round(
        0.35 * hit_score
        + 0.20 * doc_score
        + 0.30 * numeric_score
        + 0.15 * constraint_score,
        4,
    )
    sufficient = score >= threshold
    # 显式数值断言在对应数值缺失时还不能用于判定，
    # 即便检索器已经填满所有页面槽位。
    if checkpoints and numeric_score < 1.0:
        sufficient = False
    return EvidenceSufficiency(
        score=score,
        sufficient=sufficient,
        signals=tuple(signals),
        hit_count=hit_count,
        distinct_doc_count=distinct_doc_count,
        numeric_checkpoint_coverage=round(numeric_score, 4),
    )
