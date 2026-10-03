"""逐选项紧凑推理与定向复核路由。"""

from __future__ import annotations

import hashlib
import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Sequence

from ..core.agent_types import AnswerResult
from .answer_reconciliation import (
    audit_full_year_dividend_difference,
    explicit_segment_verdict,
    reconcile_cross_option_numeric_claims,
    repair_cross_option_ranking_answer,
    repair_numeric_self_consistency_verdict,
    repair_parallel_duty_omission_verdict,
    repair_region_share_answer_from_reasoning,
    repair_zero_balance_condition_verdict,
    should_apply_reasoning_answer,
)
from ..calculation.calculation_reconciliation import (
    percent_rate_tail_from_reasoning,
    rank_table_anchors,
    ratio_point_tail_from_reasoning,
    rounded_intermediate_from_reasoning,
)
from ..core.domain_policies import (
    CONTRACT_PARALLEL_CLAUSE_RULE,
    FINANCIAL_CROSS_OPTION_RULE,
    RESEARCH_FUNCTIONAL_QUALITY_RULE,
)
from ..core.entities import (
    CALC_TERMS,
    DOMAIN_TERMS,
    NEGATIVE_TERMS,
    compact,
    dedupe,
    extract_constraint_terms,
    extract_entities,
)
from ..retrieval.evidence_budget import assess_evidence_sufficiency, plan_evidence_budget
from ..retrieval.index import SearchHit, canonical
from ..runtime.io import normalize_answer, normalize_calculation_answer
from ..runtime.llm import Usage
from ..retrieval.planner import QueryPlan
from ..core.prompt_policies import (
    build_scoped_bridge_prompt,
    build_scoped_calculation_prompt,
    build_scoped_joint_prompt,
    build_scoped_option_prompt,
)
from ..core.question_routing import (
    expected_option_cardinality,
    is_dense_quantitative_cross_document_comparison,
    needs_cross_document_research_scope,
)
from .reasoning_expansion import expand_reasoning_from_counted_prompt



class CompactOptionRoutesMixin:
    def _compact_option_answer(
        self,
        question: Dict[str, object],
        discovery: Dict[str, object],
        plan: QueryPlan,
        context_hits: Sequence[SearchHit],
        quality: Dict[str, object],
        repair_steps: List[Dict[str, object]],
        usage: Usage,
        calc_results: Sequence[Dict[str, object]] = (),
    ) -> AnswerResult:
        """运行小型独立选项裁决器，避免选项间干扰。"""
        qid = str(question.get("qid", ""))
        domain = str(question.get("domain", ""))
        stem = str(question.get("question", ""))
        answer_format = str(question.get("answer_format", "multi"))
        options = question.get("options") or {}
        doc_ids = [
            str(item)
            for item in (discovery.get("inferred_doc_ids") or question.get("doc_ids") or [])
        ]
        top_k = int(self.profile.get("compact_option_hits", 3))
        excerpt_chars = int(self.profile.get("compact_excerpt_chars", 650))
        if domain == "research":
            top_k = int(self.profile.get("compact_research_option_hits", 6))
            excerpt_chars = int(self.profile.get("compact_research_excerpt_chars", 600))
        elif domain == "financial_reports":
            excerpt_chars = int(self.profile.get("compact_financial_excerpt_chars", 900))
        elif domain == "financial_contracts":
            excerpt_chars = int(self.profile.get("compact_contract_excerpt_chars", 1000))
        elif domain == "insurance":
            excerpt_chars = int(self.profile.get("compact_insurance_excerpt_chars", 1000))
        elif domain == "regulatory":
            excerpt_chars = int(self.profile.get("compact_regulatory_excerpt_chars", 900))
        max_tokens = int(self.profile.get("compact_option_max_tokens", 320))
        decisions: List[Dict[str, object]] = []
        stem_doc_ids = self._docs_named_in_option(domain, doc_ids, stem)

        per_domain_hit_keys = {
            "financial_reports": "compact_financial_option_hits",
            "financial_contracts": "compact_contract_option_hits",
            "insurance": "compact_insurance_option_hits",
            "regulatory": "compact_regulatory_option_hits",
        }
        domain_hit_key = per_domain_hit_keys.get(domain)
        if domain_hit_key:
            top_k = int(self.profile.get(domain_hit_key, top_k))
        if (
            domain == "regulatory"
            and re.search(
                r"定期报告|披露程序|非交易时段|汇款|门槛|可疑|"
                r"施行|生效|同时|期限",
                stem,
            )
        ):
            top_k = max(top_k, 3)
            excerpt_chars = max(excerpt_chars, 380)
        if (
            domain == "research"
            and re.search(r"共同|四家|多家|品牌化|不同行业|深层特征", stem)
        ):
            top_k = max(top_k, 4)
            excerpt_chars = max(excerpt_chars, 220)
        for letter, raw_option in sorted(options.items()):
            option = str(raw_option)
            binding_domains = {
                str(item)
                for item in self.profile.get("option_entity_binding_domains", [])
            }
            option_doc_ids = (
                self._docs_named_in_option(domain, doc_ids, option)
                if not binding_domains or domain in binding_domains
                else []
            )
            research_cross_scope = (
                domain == "research"
                and needs_cross_document_research_scope(
                    stem,
                    option_doc_ids,
                    doc_ids,
                )
            )
            if research_cross_scope:
                option_doc_ids = list(doc_ids)
            if (
                domain == "insurance"
                and len(stem_doc_ids) == 1
                and len(option_doc_ids) != 1
            ):
                # 题干中唯一出现的产品名会约束“投保人可以申请……”这类通用条款选项。
                # 多个模糊选项匹配通常是条款词汇，不应当作实体。
                option_doc_ids = list(stem_doc_ids)
            if (
                domain == "financial_contracts"
                and len(stem_doc_ids) == 1
                and stem_doc_ids[0] in doc_ids
                and not re.search(
                    r"(?:三|多|若干)份|各(?:文件|募集说明书|报告)|横向比较",
                    stem,
                )
            ):
                # 题干中明确点名的募集说明书/协议约束全部通用条款选项。
                # 模糊条款词汇不能把选项重定向到另一份合同。
                option_doc_ids = list(stem_doc_ids)
            queries = dedupe(
                [
                    f"{stem}\n{option}",
                    option,
                    *(
                        [
                            f"{option} 年度 原始金额 "
                            + " ".join(
                                term
                                for term in DOMAIN_TERMS.get(
                                    "financial_reports",
                                    [],
                                )
                                if term in self.index.question_text(question)
                            ),
                            "年度 2025 2024 同比 "
                            + " ".join(
                                term
                                for term in DOMAIN_TERMS.get(
                                    "financial_reports",
                                    [],
                                )
                                if term in self.index.question_text(question)
                            ),
                        ]
                        if domain == "financial_reports"
                        and re.search(r"同比|下降|增长|百分点", option)
                        else []
                    ),
                    *(
                        [
                            f"{option} 中期分红 年末方案 全年合计 剩余待分配",
                            "每10股 中期 已实施 年末 剩余待派 全年现金分红",
                        ]
                        if domain == "financial_reports"
                        and re.search(r"分红|派息|派现", stem + option)
                        else []
                    ),
                    *(
                        [
                            f"{option} 风险因素 目录 相关风险",
                            f"{' '.join(extract_entities(option, domain)[:6])} 风险因素",
                        ]
                        if re.search(r"未(?:提及|披露|列明|量化|说明)", option)
                        else []
                    ),
                    *[
                        f"{phrase} {'责任免除' if domain == 'insurance' else ''}".strip()
                        for quoted in re.findall(r"[“\"]([^”\"]{2,40})[”\"]", stem)
                        for phrase in re.split(r"[/／、]", quoted)
                        if len(phrase.strip()) >= 2
                    ],
                    " ".join(
                        dedupe(
                            [
                                *extract_entities(stem, domain),
                                *extract_entities(option, domain),
                            ],
                            18,
                        )
                    ),
                ]
            )
            budget = plan_evidence_budget(
                stem=stem,
                option=option,
                candidate_doc_count=len(doc_ids),
                bound_doc_count=len(option_doc_ids),
                retrieval_quality=quality,
                base_top_k=top_k,
                base_excerpt_chars=excerpt_chars,
                mode=str(self.profile.get("evidence_budget_controller", "off")),
            )
            budget_top_k = (
                budget.recommended_top_k if budget.mode == "active" else top_k
            )
            budget_excerpt_chars = (
                budget.recommended_excerpt_chars
                if budget.mode == "active"
                else excerpt_chars
            )
            self._event(
                "evidence_budget_planned",
                qid,
                option=str(letter),
                decision=budget.to_dict(),
            )
            option_hits = self._option_balanced_hits(
                domain,
                option_doc_ids or doc_ids,
                queries,
                budget_top_k,
            )
            if domain == "financial_reports" and "EBITDA" in option.upper():
                option_hits = self._merge_hits(
                    self.index.search(
                        domain,
                        option_doc_ids or doc_ids,
                        ["EBITDA EBITDA率 营业收入"],
                        top_k=2,
                    ),
                    option_hits,
                )
            structural_hits: List[SearchHit] = []
            if domain == "financial_reports":
                structural_hits = self._financial_statement_structure_hits(
                    option_doc_ids or doc_ids,
                    option,
                )
                if structural_hits:
                    option_hits = self._merge_hits(
                        self._with_adjacent_pages(domain, structural_hits),
                        option_hits,
                    )
                comparison_years = dedupe(
                    re.findall(r"20\d{2}", f"{stem}\n{option}"),
                    5,
                )
                if (
                    len(comparison_years) >= 3
                    and re.search(r"(?:下降|降低|回升|回落|先.+后)", option)
                ):
                    metric_terms = dedupe(
                        [
                            *re.findall(r"[A-Z][A-Z0-9._-]{1,18}", option),
                            *[
                                term
                                for term in DOMAIN_TERMS["financial_reports"]
                                if term in option
                            ],
                            *comparison_years,
                        ],
                        12,
                    )
                    trend_hits = self._exact_term_hits(
                        domain,
                        option_doc_ids or doc_ids,
                        metric_terms,
                        query=option,
                        per_document=2,
                    )
                    option_hits = self._merge_hits(trend_hits, option_hits)
                if re.search(r"分地区|境外收入", stem + "\n" + option):
                    region_terms = dedupe(
                        [
                            "分地区",
                            "境外",
                            "中国",
                            "营业收入",
                            *re.findall(r"20\d{2}", stem + "\n" + option),
                        ],
                        10,
                    )
                    option_hits = self._merge_hits(
                        self._exact_term_hits(
                            domain,
                            option_doc_ids or doc_ids,
                            region_terms,
                            query=option,
                            per_document=2,
                        ),
                        option_hits,
                    )
                if re.search(r"(?:分红|派息|派现).{0,30}(?:比例|占)", option):
                    dividend_terms = dedupe(
                        [
                            "现金分红",
                            "归属于上市公司股东的净利润",
                            "比例",
                            *re.findall(r"20\d{2}", stem + "\n" + option),
                        ],
                        10,
                    )
                    option_hits = self._merge_hits(
                        self._exact_term_hits(
                            domain,
                            option_doc_ids or doc_ids,
                            dividend_terms,
                            query=option,
                            per_document=2,
                        ),
                        option_hits,
                    )
            constraint_focus = extract_constraint_terms(
                f"{stem}\n{option}",
                domain,
            )
            if domain == "financial_reports" and re.search(
                r"分地区|境外收入",
                stem + "\n" + option,
            ):
                constraint_focus = dedupe(
                    [
                        "分地区",
                        "境外",
                        "中国",
                        "营业收入",
                        *re.findall(r"20\d{2}", stem + "\n" + option),
                        *constraint_focus,
                    ],
                    20,
                )
            if domain == "financial_reports" and re.search(
                r"(?:分红|派息|派现).{0,30}(?:比例|占)",
                option,
            ):
                constraint_focus = dedupe(
                    [
                        "现金分红",
                        "归属于上市公司股东的净利润",
                        "比例",
                        *re.findall(r"20\d{2}", stem + "\n" + option),
                        *constraint_focus,
                    ],
                    20,
                )
            if domain == "insurance" and re.search(r"\d+\s*周岁", option):
                constraint_focus = dedupe(
                    [
                        "身故给付比例",
                        "年龄对应",
                        "周岁",
                        *constraint_focus,
                    ],
                    20,
                )
            if domain == "insurance" and "一般医疗保险金" in option:
                constraint_focus = dedupe(
                    [
                        "一般医疗保险金",
                        "意外伤害事故",
                        "等待期后",
                        "住院医疗费用",
                        *constraint_focus,
                    ],
                    20,
                )
            if domain == "research" and "分位" in option:
                constraint_focus = dedupe(
                    [
                        "近三年",
                        "分位",
                        "两融",
                        "ETF",
                        *constraint_focus,
                    ],
                    20,
                )
            research_technical_anchors: List[str] = []
            if domain == "research":
                research_technical_anchors = dedupe(
                    [
                        *re.findall(r"[A-Z][A-Z0-9._-]{2,18}", option),
                        *[
                            term
                            for term in DOMAIN_TERMS.get("research", [])
                            if term in option and len(compact(term)) >= 3
                        ],
                    ],
                    12,
                )
                constraint_focus = dedupe(
                    [*research_technical_anchors, *constraint_focus],
                    20,
                )
            if (
                domain in {"insurance", "regulatory"}
                or (
                    domain == "research"
                    and (
                        "分位" in option
                        or research_technical_anchors
                        or research_cross_scope
                    )
                )
            ) and constraint_focus:
                exact_hits = self._exact_term_hits(
                    domain,
                    option_doc_ids or doc_ids,
                    constraint_focus,
                    query=option,
                    per_document=(
                        4
                        if domain == "research" and "分位" in option
                        else 2
                    ),
                )
                focused_hits = self.index.search(
                    domain,
                    option_doc_ids or doc_ids,
                    constraint_focus,
                    top_k=2,
                )
                option_hits = self._merge_hits(
                    exact_hits,
                    self._merge_hits(focused_hits, option_hits),
                )
                if (
                    domain == "research"
                    and "分位" in option
                    and exact_hits
                ):
                    best_doc_id = exact_hits[0].page.doc_id
                    same_report_hits = [
                        hit
                        for hit in option_hits
                        if hit.page.doc_id == best_doc_id
                    ]
                    if len(same_report_hits) >= 2:
                        anchor_hits: List[SearchHit] = []
                        for anchor in ("两融", "ETF"):
                            candidates = [
                                hit
                                for hit in same_report_hits
                                if anchor in hit.page.text
                            ]
                            if candidates:
                                anchor_hits.append(
                                    max(candidates, key=lambda hit: hit.score)
                                )
                        option_hits = [
                            *anchor_hits,
                            *[
                                hit
                                for hit in same_report_hits
                                if hit not in anchor_hits
                            ],
                        ]
            if re.search(r"未(?:提及|披露|列明|量化|说明)", option):
                absence_hits = self.index.search(
                    domain,
                    option_doc_ids or doc_ids,
                    dedupe(
                        [
                            f"{' '.join(extract_entities(option, domain)[:8])} 风险因素",
                            "风险因素 募投项目 实施效果 产能消化",
                        ],
                        4,
                    ),
                    top_k=6,
                )
                option_hits = self._merge_hits(absence_hits, option_hits)
            self._event(
                "option_document_binding_completed",
                qid,
                option=str(letter),
                candidate_doc_ids=doc_ids,
                bound_doc_ids=option_doc_ids,
                fallback_to_all=not bool(option_doc_ids),
            )
            if (
                domain == "insurance"
                or re.search(r"(?:T\+\d|至T\+\d|每年|逐年|明细表|测算表)", option)
            ):
                option_hits = self._with_adjacent_pages(domain, option_hits)
                self._event(
                    "adjacent_evidence_expanded",
                    qid,
                    option=str(letter),
                    reason=(
                        "insurance_clause_continuity"
                        if domain == "insurance"
                        else "multi_period_or_table_statement"
                    ),
                    evidence=[hit.to_dict(chars=0) for hit in option_hits],
                )
            sufficiency = assess_evidence_sufficiency(
                option=option,
                evidence_texts=[hit.page.text for hit in option_hits],
                evidence_doc_ids=[hit.page.doc_id for hit in option_hits],
                expected_top_k=budget_top_k,
                expected_doc_count=max(1, len(option_doc_ids or doc_ids)),
                threshold=float(
                    self.profile.get("evidence_sufficiency_threshold", 0.62)
                ),
            )
            self._event(
                "evidence_sufficiency_assessed",
                qid,
                option=str(letter),
                assessment=sufficiency.to_dict(),
            )
            evidence_chars = budget_excerpt_chars
            if re.search(r"(?:T\+\d|至T\+\d|每年|逐年|明细表|测算表)", option):
                evidence_chars = max(evidence_chars, 1600)
            if domain == "insurance" and re.search(r"\d+\s*周岁|年龄.{0,8}(?:区间|档)", option):
                # 年龄分档条款通常位于连续行。
                # 保留相邻边界，避免单行截断后被绑定到错误年龄。
                evidence_chars = max(evidence_chars, 600)
            if (
                domain == "financial_reports"
                and re.search(r"合并", option)
                and re.search(r"(?:母公司|公司口径|单体)", option)
            ):
                evidence_chars = max(evidence_chars, 900)
            hard_hit_cap = int(
                self.profile.get("compact_hard_evidence_hit_cap", 0)
            )
            if hard_hit_cap > 0:
                option_hits = self._cap_hits_preserving_documents(
                    option_hits,
                    hard_hit_cap,
                    minimum_per_document=(
                        2
                        if domain in {"financial_reports", "financial_contracts"}
                        and len(option_doc_ids or doc_ids) > 1
                        else 1
                    ),
                )
            hard_excerpt_cap = int(
                self.profile.get("compact_hard_excerpt_chars_cap", 0)
            )
            if hard_excerpt_cap > 0:
                evidence_chars = min(evidence_chars, hard_excerpt_cap)
            if (
                domain == "financial_reports"
                and re.search(r"合并", option)
                and re.search(r"(?:母公司|公司口径|单体)", option)
            ):
                evidence_chars = max(evidence_chars, 900)
            evidence = self._format_hits(
                option_hits,
                chars=evidence_chars,
                limit=len(option_hits),
                focus_query=" ".join(constraint_focus) or option,
            )
            if structural_hits:
                metric_focus = (
                    "经营活动产生的现金流量净额"
                    if re.search(r"现金流|经营活动", option)
                    else "营业收入"
                )
                metric_rows = []
                for hit in structural_hits[:2]:
                    metric_rows.append(
                        f"doc={hit.page.doc_id} page={hit.page.page_id}\n"
                        + self._focused_excerpt(
                            hit.page.text,
                            metric_focus,
                            900,
                        )
                    )
                evidence += "\n\n结构化报表指标行：\n" + "\n\n".join(metric_rows)
            direct_term_matches: List[Dict[str, str]] = []
            if constraint_focus:
                seen_direct_matches: set[tuple[str, str, str]] = set()
                for hit in option_hits:
                    for phrase in constraint_focus:
                        position = hit.page.text.find(phrase)
                        if position < 0:
                            continue
                        start = max(0, position - 80)
                        end = min(len(hit.page.text), position + len(phrase) + 100)
                        quote = re.sub(
                            r"\s+",
                            " ",
                            hit.page.text[start:end],
                        ).strip()
                        key = (hit.page.doc_id, hit.page.page_id, phrase)
                        if key in seen_direct_matches:
                            continue
                        seen_direct_matches.add(key)
                        direct_term_matches.append(
                            {
                                "doc_id": hit.page.doc_id,
                                "page_id": hit.page.page_id,
                                "term": phrase,
                                "quote": quote,
                            }
                        )
                if direct_term_matches:
                    self._event(
                        "literal_constraint_matches",
                        qid,
                        option=str(letter),
                        matches=direct_term_matches[:12],
                    )
                    if (
                        domain in {"insurance", "regulatory"}
                        or (
                            domain == "research"
                            and re.search(
                                r"分位|净流入|净流出|历史高位|历史低位",
                                stem + "\n" + option,
                            )
                        )
                        or (
                            domain == "financial_reports"
                            and re.search(
                                r"分地区|境外收入|"
                                r"(?:分红|派息|派现).{0,30}(?:比例|占)",
                                stem + "\n" + option,
                            )
                        )
                    ):
                        best_match = direct_term_matches[0]
                        evidence = (
                            "原子约束直接命中："
                            f"doc={best_match['doc_id']} "
                            f"page={best_match['page_id']} "
                            f"term={best_match['term']} "
                            f"quote={best_match['quote'][:220]}\n"
                            + evidence
                        )
            research_rule = (
                "这是研报综合判断：可由多份材料中的行业属性、事实前提和因果关系作分析性归纳；"
                "“最大/最小/最符合实际”不要求原文存在显式排名，但选项中的每个事实前提必须有依据。"
                "对带符号的资金净流量，低百分位且数值为负代表净流出更极端，不得把‘净流出低分位’"
                "误解成小幅流出。品牌化比较可从历史渠道偏B端、消费端仍在建设、产品标准化程度和"
                "品牌溢价培育阶段推断相对难度。含‘且/因为/均/已有’的选项必须逐一核对全部前提，"
                "不得把企业其他业务或母体的渠道能力直接当成新业务已具备的渠道。"
                "资产配置要做穿透判断：直接减持某类债权或非标，不等于总固收暴露下降；若同时"
                "增配且主要配置债券型、货币型基金，应把基金底层固收暴露计入整体判断。"
                "判断资金流入/流出必须先确定观察主体：资金从银行表内存款迁出，与同一笔资金流入"
                "银行理财、保险或基金等承接载体可以同时成立；不得把来源端的流出方向机械套到"
                "接收端，从而否定‘理财应对资金流入’等载体视角表述。"
                "比较不同资金主体风险偏好时，可把分散在多份材料中的负债约束、波动容忍度、"
                "杠杆参与和长期收益目标联合归纳；这些分别有证据即可支持风险偏好分化，"
                "不要求原文用一句话同时点名全部主体。"
                if domain == "research"
                else ""
            )
            brand_rule = (
                "本题涉及品牌化相对难度：用统一五维框架比较各案例——历史渠道偏企业端还是"
                "消费端、产品差异是否易被消费者感知、既有消费者心智、可继承渠道、质量信任"
                "建立成本。材料显示某业务长期偏企业端、消费品牌和溢价仍在培育时，这些事实"
                "足以支持相对难度推断；正在打造品牌不等于品牌化已经容易完成。"
                if domain == "research" and re.search(r"品牌(?:化)?.{0,8}难度", stem)
                else ""
            )
            financial_rule = (
                "财报题必须逐项核对同一发行人、同一年度和合并/母公司口径；百分比相对变化"
                "与百分点绝对变化分开计算。全年现金分红要区分中期已实施金额与年末方案，"
                "只相加一次；若年度分配目标先扣除中期、再按剩余待派金额报价，则该报价"
                "不含中期，全年还应加回中期。表中本期、上年同期、调整前/调整后不得串列。所有中间值不先"
                "四舍五入，再按选项精度核验。若选项给出年度同比数值，且证据存在连续两年的年度原始值，"
                "必须用年度值独立验算；不能仅因同一百分数也出现在季度表中就把年度结论判错。"
                "若选项说占比提高约若干个百分点，且同一口径披露占比直接相减与其吻合，应判成立，"
                "不得因隐藏小数位或另一个不同口径表格无端否定。若选项说利润率下降幅度，必须区分"
                "百分点差(old-new)和相对降幅((old-new)/old)：把百分点差写成相对降幅应判错。"
                "同比变动率为负数与‘同比下降’语义一致，不能仅因负号与‘下降’同现就判错。"
                "若连续两年的年度原始金额可直接算出选项同比，即使同一百分数也出现在季度列，"
                "仍应以年度原始金额独立验算后支持年度结论。选项使用‘约’且由原始金额计算与"
                "展示占比相减仅产生0.01个百分点尾差时，应视为四舍五入误差，不得判错。"
                "若证据给出同一年度Q1至Q4或第一至第四季度四个互斥期间的原始金额，允许且必须"
                "相加得到全年金额；这属于证据内算术，不得仅因未另列全年合计而判为缺证。"
                if domain == "financial_reports"
                else ""
            )
            quantifier_rule = (
                "选项含“无论、均、全部、一律、任何”等全称量词时，证据必须覆盖其声称的全部情形；"
                "若证据只在“可疑、达到门槛、满足条件、除外情形”等前提下成立，应判为 contradicted，"
                "不得把有条件规则扩张为无条件规则。先确定量词的作用域：选项在“无论”之前已经给出的"
                "主语限定仍然有效，例如“某类交易无论金额”只量化金额，并未取消“某类交易”这一限定；"
                "只有选项明确声称“无论是否满足该条件”时才覆盖条件内外。允许选项用同义上位表达压缩"
                "法规条件，例如选项主语已经限定为“可疑交易/可疑情形”，就保留了证据中“有合理理由怀疑"
                "涉嫌特定风险”的前提，不能再以省略该前提为由判错。"
            )
            regulatory_scope_rule = (
                "法规选项必须精确核对义务对象和事项范围：上位概念与其某个子任务不能互相替代，"
                "例如某项识别核实、资料保存或差异反馈的期限，不得自动扩张成全部尽调活动的期限。"
                "不同办法、不同报告主体使用相似术语时，应以题干所指制度和文档标题确定适用规则。"
                "选项准确陈述一项并行义务时，不得仅因没有同时复述另一项反馈、核实或报告义务而"
                "判错，除非选项使用‘仅、只、无需、完整流程’等排他表述。"
                "多部办法存在相似义务时，必须将期限绑定到各自办法和施行日，不能移植生效日期。"
                "题干同时给出多个触发条件时，各项法定义务通常并行适用；客户拒绝配合不当然"
                "取消因交易异常而产生的进一步核实义务。"
                if domain == "regulatory"
                else ""
            )
            insurance_scope_rule = (
                "保险年龄分档必须有覆盖该年龄的明确区间，不得用行业惯例或相邻区间反推。"
                "题干若概括询问某类风险是否明确列明，则核反应/核辐射/核爆炸/核污染/放射性污染"
                "等同类术语，或恐怖活动/恐怖袭击等同义术语，均可支持类别层面的列明；"
                "只有题干要求逐字一致或完整列举时，才要求全部示例词同时出现。"
                "必须检查全部编号证据；若较后的证据片段直接出现目标免责词，不得因为较早片段"
                "只展示了不完整列表就判为缺证。"
                "借款公式若为‘(现金价值-欠款及利息)×比例’，而选项明确限定不存在未偿还"
                "借款、利息或欠交保费，则扣减项为零，公式可确定地简化为现金价值×该比例。"
                if domain == "insurance"
                else ""
            )
            contract_scope_rule = (
                "软件研发、信息系统实施等非传统生产线项目若不直接形成可量化产能，可据此判断"
                "传统新增产能消化风险不适用；不得仅因没有同名风险标题而判为证据不足。"
                if domain == "financial_contracts"
                else ""
            )
            prompt = f"""你是金融长文档的单选项证据裁决微代理。本次只判断一个选项，不决定整题答案。
裁决标准：
- supported：证据直接支持，或能由证据完成确定的算术、时间、条件及因果推导。
- contradicted：主体、日期、数值、单位、条件、例外或逻辑与证据冲突。
- unknown：关键事实确实缺证，不能用常识补足。
- 核对法规施行时点、保险责任/除外、财务口径以及绝对变化和相对变化。
- 若题干给出判断时点，选项称某规则“已于该日生效”，应判断实际施行日是否不晚于该时点；只有“自该日起/生效日为”才要求日期完全相同。
- 若选项准确陈述一个基础公式或规则，不得仅因它没有同时复述另一条上限、例外而判错，除非选项声称完整、唯一或无条件。
- {research_rule}
- {brand_rule}
- {financial_rule}
- {quantifier_rule}
- {regulatory_scope_rule}
- {insurance_scope_rule}
- {contract_scope_rule}
只输出严格JSON：{{"verdict":"supported|contradicted|unknown","reasoning":"不超过100字并引用证据编号"}}

题目：{stem}
待判断选项{letter}：{option}
本轮原始文档定向证据：
{evidence}
"""
            if self.profile.get("compact_domain_scoped_prompt", False):
                prompt = build_scoped_option_prompt(
                    domain=domain,
                    stem=stem,
                    letter=str(letter),
                    option=option,
                    evidence=evidence,
                )
            self._event(
                "prompt_routed",
                qid,
                route="qwen_compact_option_judge",
                option=str(letter),
                prompt_chars=len(prompt),
                prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                evidence_count=len(option_hits),
                max_tokens=max_tokens,
            )
            response = self.llm.chat(
                [{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
            )
            usage.add(response.usage)
            parsed = self._parse_answer(response.content, "extract")
            verdict = str(parsed.get("verdict", "")).lower()
            if verdict not in {"supported", "contradicted", "unknown"}:
                match = re.search(
                    r"\b(supported|contradicted|unknown)\b",
                    response.content,
                    flags=re.I,
                )
                verdict = match.group(1).lower() if match else "unknown"
            api_reasoning = str(parsed.get("reasoning", "")).strip() or response.content.strip()
            reconciled = self._verdict_from_reasoning(api_reasoning)
            if reconciled:
                verdict = reconciled
            consistent = self._verdict_from_option_consistency(option, api_reasoning)
            if consistent:
                verdict = consistent
            verdict = explicit_segment_verdict(api_reasoning, verdict)
            verdict = repair_parallel_duty_omission_verdict(
                option,
                api_reasoning,
                verdict,
            )
            verdict = repair_numeric_self_consistency_verdict(
                option,
                api_reasoning,
                verdict,
            )
            verdict = repair_zero_balance_condition_verdict(
                option,
                api_reasoning,
                verdict,
            )
            deterministic_exclusion = False
            if domain == "financial_reports":
                audit_doc_ids = set(option_doc_ids or doc_ids)
                audit_evidence = "\n".join(
                    page.text
                    for page in self.index.pages_by_domain.get(domain, [])
                    if page.doc_id in audit_doc_ids
                )
                audited_verdict = audit_full_year_dividend_difference(
                    option,
                    api_reasoning,
                    audit_evidence,
                    verdict,
                )
                deterministic_exclusion = (
                    verdict != "contradicted"
                    and audited_verdict == "contradicted"
                )
                verdict = audited_verdict
            presence = self._document_presence_verdict(
                question,
                option,
                doc_ids,
                verdict,
            )
            if presence:
                verdict = presence
            if not deterministic_exclusion:
                verdict = explicit_segment_verdict(api_reasoning, verdict)
            decision = {
                "option": str(letter),
                "verdict": verdict,
                "reasoning": api_reasoning,
                "usage": response.usage.to_dict(),
                "evidence": [hit.to_dict(chars=0) for hit in option_hits],
                "deterministic_exclusion": deterministic_exclusion,
            }
            if (
                domain == "financial_reports"
                and re.search(r"分红|派息|派现", stem + option)
            ):
                decision["evidence_highlight"] = "\n".join(
                    self._focused_excerpt(hit.page.text, hit.query, 240)
                    for hit in option_hits
                )[:600]
            decisions.append(decision)
            self._event(
                "compact_option_judgment_completed",
                qid,
                option=str(letter),
                verdict=verdict,
                api_reasoning=api_reasoning,
                usage=response.usage.to_dict(),
            )

        # 在调用未知项桥接前，先解析首轮 API 推理中已有的算术和排序结论。
        reconcile_cross_option_numeric_claims(decisions, options)
        if domain == "financial_reports":
            audit_doc_ids = set(doc_ids)
            audit_evidence = "\n".join(
                page.text
                for page in self.index.pages_by_domain.get(domain, [])
                if page.doc_id in audit_doc_ids
            )
            for item in decisions:
                letter = str(item.get("option", ""))
                item["verdict"] = audit_full_year_dividend_difference(
                    str(options.get(letter, "")),
                    str(item.get("reasoning", "")),
                    audit_evidence,
                    str(item.get("verdict", "")),
                )

        if domain in {
            "research",
            "financial_reports",
            "financial_contracts",
            "insurance",
            "regulatory",
        }:
            if (
                domain == "financial_reports"
                and re.search(
                    r"全年.{0,20}(?:分红|派现)|(?:分红|派现).{0,20}全年",
                    stem + "\n" + "\n".join(str(value) for value in options.values()),
                )
            ):
                for item in decisions:
                    option_text_for_risk = str(options.get(str(item.get("option", "")), ""))
                    if (
                        item.get("verdict") != "contradicted"
                        and re.search(
                            r"(?:分红|派现).{0,24}相差|相差.{0,24}(?:分红|派现)",
                            option_text_for_risk,
                        )
                    ):
                        item["verdict"] = "unknown"
            unknown = [
                item for item in decisions if item.get("verdict") == "unknown"
            ]
            if (
                domain == "insurance"
                and re.search(r"保单贷款|借款", stem)
            ):
                unknown = []
            if unknown:
                brand_bridge_rule = (
                    "品牌化相对难度按历史企业端/消费端渠道、产品差异可感知性、消费者心智、"
                    "渠道继承性和质量信任成本统一比较。若材料显示消费品牌与溢价仍在培育，"
                    "可据此推断转向消费端的相对困难；不得把‘正在建设’当成‘已经容易’。"
                    if re.search(r"品牌(?:化)?.{0,8}难度", stem)
                    and domain == "research"
                    else ""
                )
                bridge_instruction = (
                    "你是研报跨文档因果桥接代理。允许从不同行业材料的明确事实前提进行"
                    "分析性比较和机制归纳；“最大/最小/共同逻辑”不要求原文给出排名表。"
                    if domain == "research"
                    else (
                    "你是财报跨表格口径桥接代理。必须联合查阅同一发行人的相关年度、"
                    "合并与母公司报表和相邻表格，重新用原始值验算不确定项。严格区分"
                    "百分比与百分点、调整前后口径；若年度方案写明扣除已实施中期分红，"
                    "且随后按剩余待分配金额给出每股/每10股金额，则该金额只代表剩余待派，"
                    "全年金额=已派中期+剩余待派；不得把剩余待派误称为‘已含中期’。"
                    if domain == "financial_reports"
                    else
                    "你是金融合同跨章节桥接代理。必须回到题干指定的募集说明书、协议或报告书，"
                    "联合核对定义、触发条件、期限、计算公式、除外条款和争议解决章节；不得用"
                    "另一发行人或另一份合同的相似条款替代。"
                    )
                )
                bridge_analysis_rule = (
                    "“最符合实际”类题目应比较材料中呈现的渠道基础、技术门槛、产品差异化"
                    "和用户认知，不得仅因原文未逐字写出“难度最大”而拒绝有充分事实基础的"
                    "结论。技术带来的效率、可靠性、强度、精度或品质改善均可作为提质或性能"
                    "提升的证据。带符号净流量位于极低百分位且为负值，代表净流出程度极端。"
                    if domain == "research"
                    else
                    "逐项核对发行人、年度、合并/母公司及调整前后口径；重新执行必要算术，"
                    "所有中间值保持原始精度。带‘且/均/由…至…’的选项每个子命题都须成立。"
                )
                bridge_hits: List[SearchHit] = []
                bridge_top_k = int(
                    self.profile.get(
                        "compact_financial_bridge_hits"
                        if domain == "financial_reports"
                        else (
                            "compact_insurance_bridge_hits"
                            if domain == "insurance"
                            else "compact_research_bridge_hits"
                        ),
                        16,
                    )
                )
                bridge_excerpt_chars = int(
                    self.profile.get(
                        "compact_financial_bridge_excerpt_chars"
                        if domain == "financial_reports"
                        else (
                            "compact_insurance_bridge_excerpt_chars"
                            if domain == "insurance"
                            else "compact_research_bridge_excerpt_chars"
                        ),
                        700,
                    )
                )
                if (
                    len(unknown) >= 2
                    and (
                        self.profile.get(
                            "compact_bridge_wide_on_multiple_unknowns",
                            True,
                        )
                        or domain
                        in {
                            str(item)
                            for item in self.profile.get(
                                "compact_bridge_wide_domains",
                                [],
                            )
                        }
                    )
                ):
                    # 多个未决选项通常意味着真实的跨文档比较，而不是漏掉单个条款。
                    # 只在这条困难路由中保留更宽的证据覆盖。
                    bridge_top_k = max(bridge_top_k, 12)
                    bridge_excerpt_chars = max(bridge_excerpt_chars, 550)
                if (
                    domain == "financial_reports"
                    and re.search(
                        r"分地区|境外|地区收入",
                        stem + "\n" + "\n".join(
                            str(options.get(str(item.get("option", "")), ""))
                            for item in unknown
                        ),
                    )
                ):
                    bridge_top_k = max(bridge_top_k, 16)
                    bridge_excerpt_chars = max(bridge_excerpt_chars, 700)
                for item in unknown:
                    letter = str(item["option"])
                    option = str(options.get(letter, ""))
                    deep_queries = dedupe(
                        [
                            f"{stem}\n{option}",
                            option,
                            " ".join(
                                dedupe(
                                    [
                                        *extract_entities(stem, domain),
                                        *extract_entities(option, domain),
                                    ],
                                    24,
                                )
                            ),
                        ]
                    )
                    if domain == "financial_reports":
                        if re.search(r"原始披露|调整前", stem + option):
                            deep_queries.insert(
                                0,
                                "2024年年报 原始披露 调整前 归母净利润 基本每股收益",
                            )
                        if "EBITDA" in option.upper():
                            deep_queries.insert(
                                0,
                                "EBITDA 营业收入 EBITDA率 2025",
                            )
                        if "营业收入" in option:
                            deep_queries.append(
                                "合并及公司利润表 营业收入 2025年度 2024年度 公司"
                            )
                        if re.search(r"现金流|经营活动", option):
                            deep_queries.append(
                                "合并及公司现金流量表 经营活动产生的现金流量净额 "
                                "2025年度 2024年度 公司"
                            )
                        if re.search(r"分红|派息|派发现金", option):
                            deep_queries.append(
                                "利润分配 扣除 已分派 中期 剩余待分配 每10股 全年"
                            )
                            deep_queries.append(
                                "中期 已实施 每10股 现金分红 年末 全年",
                            )
                        if re.search(r"境外|分地区|地区收入", option):
                            deep_queries.append(
                                "分地区 营业收入 中国 境外 本年 上年 占比"
                            )
                    elif domain == "insurance":
                        if "诉讼时效" in stem:
                            deep_queries.insert(
                                0,
                                "诉讼时效 请求给付保险金 2年 二年",
                            )
                    bridge_doc_ids = (
                        self._docs_named_in_option(domain, doc_ids, option)
                        or doc_ids
                    )
                    if (
                        domain == "financial_contracts"
                        and len(stem_doc_ids) == 1
                        and stem_doc_ids[0] in doc_ids
                        and not re.search(
                            r"(?:三|多|若干)份|各(?:文件|募集说明书|报告)|横向比较",
                            stem,
                        )
                    ):
                        bridge_doc_ids = list(stem_doc_ids)
                    option_bridge_hits = self._option_balanced_hits(
                        domain,
                        bridge_doc_ids,
                        deep_queries,
                        top_k=bridge_top_k,
                    )
                    if domain == "financial_reports":
                        structural_hits = self._financial_statement_structure_hits(
                            bridge_doc_ids,
                            option,
                        )
                        option_bridge_hits = self._merge_hits(
                            structural_hits,
                            option_bridge_hits,
                        )
                        option_bridge_hits = self._with_adjacent_pages(
                            domain,
                            option_bridge_hits,
                        )
                    elif domain == "insurance":
                        option_bridge_hits = self._with_adjacent_pages(
                            domain,
                            option_bridge_hits,
                        )
                    bridge_hits.extend(option_bridge_hits)
                bridge_hits = self._merge_hits(bridge_hits, context_hits)
                bridge_limit = min(
                    24,
                    max(
                        bridge_top_k,
                        (12 if domain == "insurance" else 8) * len(unknown),
                    ),
                )
                hard_bridge_hit_cap = int(
                    self.profile.get("compact_hard_bridge_hit_cap", 0)
                )
                if hard_bridge_hit_cap > 0:
                    bridge_limit = min(bridge_limit, hard_bridge_hit_cap)
                hard_bridge_excerpt_cap = int(
                    self.profile.get("compact_hard_bridge_excerpt_chars_cap", 0)
                )
                if hard_bridge_excerpt_cap > 0:
                    bridge_excerpt_chars = min(
                        bridge_excerpt_chars,
                        hard_bridge_excerpt_cap,
                    )
                bridge_prompt = f"""{bridge_instruction}
首轮单文档裁决存在不确定项，请仅使用本轮原始文档背景证据联合判断这些选项。
{bridge_analysis_rule}
含“且/因为/均/已有”的选项须核对每个前提及其业务归属，不能把其他板块或母体的数据
移植给目标业务。
{brand_bridge_rule}
必须返回“待复核选项”中的每一个字母键，不得只回答其中一项，也不得遗漏 unknown 项。
只输出严格JSON：
{{"judgments":{{"A":{{"verdict":"supported|contradicted|unknown","reasoning":"不超过100字"}}}}}}

题目：{stem}
待复核选项：
{json.dumps({str(item['option']): str(options.get(item['option'], '')) for item in unknown}, ensure_ascii=False)}
首轮裁决：
{json.dumps({str(item['option']): item['reasoning'] for item in unknown}, ensure_ascii=False)}
跨文档背景证据：
{self._format_hits(
    bridge_hits,
    chars=bridge_excerpt_chars,
    limit=bridge_limit,
)}
"""
                if self.profile.get("compact_domain_scoped_prompt", False):
                    bridge_prompt = build_scoped_bridge_prompt(
                        domain=domain,
                        stem=stem,
                        unresolved_options={
                            str(item["option"]): str(
                                options.get(item["option"], "")
                            )
                            for item in unknown
                        },
                        first_reasoning={
                            str(item["option"]): item["reasoning"]
                            for item in unknown
                        },
                        evidence=self._format_hits(
                            bridge_hits,
                            chars=bridge_excerpt_chars,
                            limit=bridge_limit,
                        ),
                    )
                bridge_max_tokens = int(
                    self.profile.get(
                        "compact_insurance_bridge_max_tokens"
                        if domain == "insurance"
                        else "compact_bridge_max_tokens",
                        600,
                    )
                )
                self._event(
                    "prompt_routed",
                    qid,
                    route="qwen_compact_unknown_bridge",
                    options=[str(item["option"]) for item in unknown],
                    prompt_chars=len(bridge_prompt),
                    prompt_sha256=hashlib.sha256(bridge_prompt.encode("utf-8")).hexdigest(),
                    max_tokens=bridge_max_tokens,
                )
                bridge = self.llm.chat(
                    [{"role": "user", "content": bridge_prompt}],
                    max_tokens=bridge_max_tokens,
                )
                usage.add(bridge.usage)
                parsed_bridge = self._parse_answer(bridge.content, "extract")
                judgments = parsed_bridge.get("judgments")
                if isinstance(judgments, dict):
                    for decision in unknown:
                        letter = str(decision["option"])
                        revised = judgments.get(letter)
                        if not isinstance(revised, dict):
                            continue
                        verdict = str(revised.get("verdict", "")).lower()
                        api_reasoning = str(revised.get("reasoning", "")).strip()
                        if verdict in {"supported", "contradicted"}:
                            reconciled = self._verdict_from_reasoning(api_reasoning)
                            if reconciled:
                                verdict = reconciled
                            consistent = self._verdict_from_option_consistency(
                                str(options.get(letter, "")),
                                api_reasoning,
                            )
                            if consistent:
                                verdict = consistent
                            verdict = explicit_segment_verdict(
                                api_reasoning,
                                verdict,
                            )
                            verdict = repair_parallel_duty_omission_verdict(
                                str(options.get(letter, "")),
                                api_reasoning,
                                verdict,
                            )
                            verdict = repair_numeric_self_consistency_verdict(
                                str(options.get(letter, "")),
                                api_reasoning,
                                verdict,
                            )
                            verdict = repair_zero_balance_condition_verdict(
                                str(options.get(letter, "")),
                                api_reasoning,
                                verdict,
                            )
                            decision["verdict"] = verdict
                            decision["reasoning"] = api_reasoning or decision["reasoning"]
                            decision["bridge_usage"] = bridge.usage.to_dict()
                self._event(
                    "compact_unknown_bridge_completed",
                    qid,
                    api_reasoning=bridge.content,
                    decisions=decisions,
                    usage=bridge.usage.to_dict(),
                )

        # 桥接调用可能返回陈旧 JSON verdict，但 API 解释已经得出相反结论。
        # 所有 API 调用结束后统一校准最终片段，避免答案选择和提交 reasoning 分叉。
        for decision in decisions:
            option_text = str(options.get(str(decision.get("option", "")), ""))
            segment = str(decision.get("reasoning", ""))
            verdict = explicit_segment_verdict(
                segment,
                str(decision.get("verdict", "")).lower(),
            )
            verdict = repair_zero_balance_condition_verdict(
                option_text,
                segment,
                verdict,
            )
            decision["verdict"] = verdict
        reconcile_cross_option_numeric_claims(decisions, options)
        target = (
            "contradicted"
            if self._asks_for_incorrect_stem(stem)
            else "supported"
        )
        selected = [
            str(item["option"])
            for item in decisions
            if item["verdict"] == target
        ]
        answer = normalize_answer("".join(selected), answer_format)
        reasoning = "；".join(
            f"{item['option']}项：{item['reasoning']}"
            for item in decisions
        )
        review_domains = {
            str(item)
            for item in self.profile.get("compact_joint_review_domains", [])
        }
        invalid_cardinality = (
            (answer_format in {"mcq", "tf"} and len(selected) != 1)
            or (answer_format == "multi" and not selected)
        )
        strong_contract_stem = (
            len(stem_doc_ids) == 1 and stem_doc_ids[0] in doc_ids
        )
        comparative_contract = (
            domain == "financial_contracts"
            and not strong_contract_stem
            and len(doc_ids) > 1
        )
        financial_consistency_risk = (
            domain == "financial_reports"
            and (
                any(
                    re.search(
                        r"季度.{0,80}(?:全年|年度)|(?:全年|年度).{0,80}季度",
                        str(item.get("reasoning", "")),
                    )
                    for item in decisions
                )
                or (
                    re.search(
                        r"全年.{0,16}(?:分红|派现)|(?:分红|派现).{0,16}全年",
                        stem + "\n" + "\n".join(str(value) for value in options.values()),
                    )
                )
                or (
                    "合并" in stem
                    and "母公司" in stem
                    and sum(
                        bool(re.search(r"现金流|营业收入", str(value)))
                        for value in options.values()
                    )
                    >= 2
                )
                or (
                    re.search(r"约|百分点|相对增幅|比重|比例", stem)
                    and any(
                        re.search(
                            r"矛盾|冲突|误用|尾差|不精确|此前|首轮|重新验算|复核计算",
                            str(item.get("reasoning", "")),
                        )
                        for item in decisions
                    )
                )
                or any(
                    "约" in str(options.get(str(item.get("option", "")), ""))
                    and item.get("verdict") in {"contradicted", "unknown"}
                    and re.search(
                        r"计算.{0,30}(?:不符|缺证)|关键.{0,16}缺证|无法(?:核实|确认|计算)",
                        str(item.get("reasoning", "")),
                    )
                    for item in decisions
                )
            )
        )
        if (
            (
                self.profile.get("compact_uncertain_joint_review", True)
                and (
                    domain in review_domains
                    or comparative_contract
                    or financial_consistency_risk
                )
            )
            or (
                invalid_cardinality
                and self.profile.get("compact_invalid_answer_review", False)
            )
        ):
            reviewed_answer, reviewed_reasoning, review_usage = self._compact_decision_review(
                question,
                decisions,
                context_hits,
            )
            usage.add(review_usage)
            if reviewed_answer:
                reviewed_answer = repair_cross_option_ranking_answer(
                    reviewed_answer,
                    reviewed_reasoning,
                    options,
                    answer_format,
                )
                reviewed_answer = normalize_answer(
                    "".join(
                        letter
                        for letter in reviewed_answer
                        if not any(
                            str(item.get("option", "")) == letter
                            and bool(item.get("deterministic_exclusion", False))
                            for item in decisions
                        )
                    ),
                    answer_format,
                )
                answer = reviewed_answer
                reasoning = reviewed_reasoning
                self._event(
                    "compact_joint_review_completed",
                    qid,
                    selected_answer=answer,
                    api_reasoning=reasoning,
                    usage=review_usage.to_dict(),
                )
        if domain == "financial_reports":
            ranking_reasoning = "；".join(
                f"{item.get('option', '')}：{item.get('reasoning', '')}"
                for item in decisions
            )
            answer = repair_cross_option_ranking_answer(
                answer,
                ranking_reasoning,
                options,
                answer_format,
            )
            answer = normalize_answer(
                "".join(
                    letter
                    for letter in answer
                    if not any(
                        str(item.get("option", "")) == letter
                        and bool(item.get("deterministic_exclusion", False))
                        for item in decisions
                    )
                ),
                answer_format,
            )
        minimum_multi_options = int(
            self.profile.get("multi_min_options", 0)
        )
        if (
            answer_format == "multi"
            and minimum_multi_options > 0
            and len(answer) < minimum_multi_options
        ):
            reviewed_answer, reviewed_reasoning, review_usage = (
                self._compact_minimum_multi_review(
                    question,
                    answer,
                    reasoning,
                    self._format_hits(context_hits, chars=500, limit=10),
                    minimum_multi_options,
                )
            )
            usage.add(review_usage)
            if len(reviewed_answer) >= minimum_multi_options:
                answer = reviewed_answer
                reasoning = reviewed_reasoning
        answer = repair_region_share_answer_from_reasoning(
            answer,
            reasoning,
            options,
            answer_format,
        )
        self._event(
            "compact_option_answer_completed",
            qid,
            target_verdict=target,
            selected_answer=answer,
            decisions=decisions,
            usage=usage.to_dict(),
        )
        return AnswerResult(
            qid=qid,
            answer=answer,
            reasoning=reasoning,
            usage=usage,
            metadata={
                "route": "compact_option_adjudication",
                "doc_discovery": discovery,
                "query_plan": plan.to_dict(),
                "query_quality": quality,
                "query_repair_steps": repair_steps,
                "option_adjudication": decisions,
            },
        )

    def _compact_minimum_multi_review(
        self,
        question: Dict[str, object],
        answer: str,
        reasoning: str,
        evidence: str,
        minimum: int,
    ) -> tuple[str, str, Usage]:
        qid = str(question.get("qid", ""))
        stem = str(question.get("question", ""))
        options = question.get("options") or {}
        prompt = f"""你是金融长文档多选题基数复核代理。该数据集的多选题至少有{minimum}个正确选项，
首轮答案只有{len(answer)}项，因此必然遗漏。请重新逐项核对本轮原始证据，重点检查未入选项；
不得仅为凑数量猜测，也不得使用文档外的实体知识。输出至少{minimum}个按字母排序的选项。
只输出严格JSON：{{"answer":"AB或ACD等","reasoning":"不超过180字，说明补选依据"}}

题目：{stem}
选项：{json.dumps(options, ensure_ascii=False)}
首轮答案与API推理：{answer}；{reasoning}
本轮原始证据：
{evidence[:7000]}
"""
        self._event(
            "prompt_routed",
            qid,
            route="qwen_minimum_multi_cardinality_review",
            prompt_chars=len(prompt),
            prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            max_tokens=420,
        )
        response = self.llm.chat(
            [{"role": "user", "content": prompt}],
            max_tokens=420,
        )
        parsed = self._parse_answer(response.content, "multi")
        reviewed_answer = normalize_answer(
            str(parsed.get("answer", "")),
            "multi",
        )
        reviewed_reasoning = (
            str(parsed.get("reasoning", "")).strip()
            or response.content.strip()
        )
        self._event(
            "minimum_multi_cardinality_review_completed",
            qid,
            selected_answer=reviewed_answer,
            api_reasoning=reviewed_reasoning,
            usage=response.usage.to_dict(),
        )
        return reviewed_answer, reviewed_reasoning, response.usage

    def _compact_decision_review(
        self,
        question: Dict[str, object],
        decisions: Sequence[Dict[str, object]],
        context_hits: Sequence[SearchHit],
    ) -> tuple[str, str, Usage]:
        """调用一个 API 仲裁器，仅解决本次运行中的选项级冲突。"""
        qid = str(question.get("qid", ""))
        domain = str(question.get("domain", ""))
        stem = str(question.get("question", ""))
        answer_format = str(question.get("answer_format", "multi"))
        options = question.get("options") or {}
        option_text = "\n".join(
            f"{key}. {value}" for key, value in sorted(options.items())
        )
        cardinality = {
            "mcq": "这是单选题，answer必须且只能有一个字母。",
            "tf": "这是判断/单选题，answer必须且只能有一个字母。",
            "multi": "这是多选题，answer由所有成立选项的字母按顺序组成。",
        }.get(answer_format, "严格按题意输出答案。")
        research_review_rule = (
            "研报题允许从文档事实作分析性归纳：明确的极低/极高百分位本身足以支持极端分化；"
            "“最符合实际、最大/最小、深层特征”应比较材料所呈现的相对难度与机制，不要求"
            "原文逐字出现结论；带符号净流量为负且处极低百分位即表示净流出程度极端，不得"
            "按绝对值误读。效率、可靠性、强度、精度、品质等改善均属于提质或性能改善。"
            "品牌化难度可由历史渠道偏企业端、消费端建设阶段、品类标准化和品牌溢价培育情况"
            "比较推断。含‘且/因为/均/已有’的选项必须全部前提同属目标业务；企业其他板块或"
            "母体的全球渠道不能证明一个新业务已经拥有全球化渠道。"
            if domain == "research"
            else ""
        )
        brand_review_rule = (
            "本题应用品牌迁移五维比较框架：历史渠道偏企业端或消费端、产品差异的消费者"
            "可感知性、既有消费者心智、渠道可继承性、质量信任建立成本。原始材料若显示"
            "一项业务长期偏企业端且消费品牌/溢价仍在培育，可据此判断其转向消费端的相对"
            "难度，不要求报告直接写出‘最难’；品牌建设已有进展也不代表挑战不存在。"
            if domain == "research" and re.search(r"品牌(?:化)?.{0,8}难度", stem)
            else ""
        )
        financial_review_rule = (
            "财报复核须重新验算每个数字，并确保发行人、年度、合并/母公司、调整前/后口径"
            "一致；严格区分相对百分比与百分点。全年分红只把中期与年末方案各计一次，不能"
            "把全年合计再次与中期相加。若方案先从年度分配目标中扣除已派中期，再用剩余金额"
            "计算每股/每10股金额，则该报价是剩余待派而非含中期，全年应等于两者相加。"
            "中间过程使用原始数值，不先四舍五入。选项使用“约”时，应按原始值及题定精度"
            "判断，不能因用已展示的两位中间比例相减产生0.01个百分点尾差而判错。最终答案"
            "必须包含reasoning中已明确判为正确、成立或基本吻合的全部选项，不得再以首轮"
            "裁决或‘保守’为由排除。同一题各选项已经抽取的同年度原始数值应交叉复用："
            "若一个裁决给出年度营收、另一个给出年度现金流，不得在第三项换成其他期间数值；"
            "合并与母公司同名项目必须按表头列位置分别核对，不能把合并数重复当成母公司数。"
            + FINANCIAL_CROSS_OPTION_RULE
            if domain == "financial_reports"
            else ""
        )
        contract_review_rule = (
            "合同跨文件比较中，‘完全一致、均、所有’是强断言，必须逐份列举触发条件和"
            "全部例外；任何一份多出转股、回购目的、期限或表决条件，均不能判为完全一致。"
            "不得把软件研发或信息系统项目的市场接受度、实施效果风险，机械等同于传统制造业"
            "新增生产线的产能消化风险；应按项目性质和选项限定语义判断。"
            if domain == "financial_contracts"
            else ""
        )
        compact_decisions = [
            {
                "option": item.get("option", ""),
                "verdict": item.get("verdict", ""),
                "reasoning": item.get("reasoning", ""),
                "evidence_highlight": item.get("evidence_highlight", ""),
            }
            for item in decisions
        ]
        prompt = f"""你是金融长文档问答的联合复核代理。仅使用本轮原始文档证据和首轮API裁决，
独立确定整题最佳答案。逐项裁决可能因证据被切分而过严，你必须结合题干、选项整体语义和
跨片段关系复核；选项核心规则得到直接支持时，不得只因它附带了可由其他证据支持的补充说明
就判错。若补充说明与证据冲突，仍应判错。不要用外部常识补充具体事实。
{research_review_rule}
{brand_review_rule}
{financial_review_rule}
{contract_review_rule}
{cardinality}
只输出严格JSON：{{"answer":"A或AC等","reasoning":"不超过180字，引用首轮裁决或背景证据"}}

领域：{domain}
题目：{stem}
选项：
{option_text}
首轮独立裁决：
{json.dumps(compact_decisions, ensure_ascii=False)}
本轮原始文档背景证据：
{self._format_hits(
    context_hits,
    chars=int(self.profile.get("compact_joint_review_excerpt_chars", 500)),
    limit=12,
)}
"""
        self._event(
            "prompt_routed",
            qid,
            route="qwen_compact_joint_review",
            prompt_chars=len(prompt),
            prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            max_tokens=int(self.profile.get("compact_joint_review_max_tokens", 500)),
        )
        response = self.llm.chat(
            [{"role": "user", "content": prompt}],
            max_tokens=int(self.profile.get("compact_joint_review_max_tokens", 500)),
        )
        review_usage = Usage()
        review_usage.add(response.usage)
        parsed = self._parse_answer(response.content, answer_format)
        answer = normalize_answer(str(parsed.get("answer", "")), answer_format)
        reasoning = str(parsed.get("reasoning", "")).strip() or response.content.strip()
        reasoning_answer = self._answer_from_reasoning(reasoning, answer_format)
        if reasoning_answer and not self._asks_for_incorrect_stem(stem):
            answer = reasoning_answer
        if answer_format == "multi":
            corrected_letters = re.findall(
                r"([A-D])(?:项)?.{0,12}?"
                r"(?:修正为\s*SUPPORTED|应(?:为)?SUPPORTED|应成立|亦成立|"
                r"应判(?:为)?SUPPORTED)",
                reasoning.upper(),
            )
            consistent_letters = [
                str(item.get("option", ""))
                for item in decisions
                if self._verdict_from_option_consistency(
                    str(options.get(str(item.get("option", "")), "")),
                    str(item.get("reasoning", "")),
                )
                == "supported"
            ]
            if corrected_letters or consistent_letters:
                answer = normalize_answer(
                    answer
                    + "".join(corrected_letters)
                    + "".join(consistent_letters),
                    answer_format,
                )
        if self._has_explicit_arithmetic_contradiction(reasoning):
            repair_prompt = f"""你是金融问答算术一致性修复代理。上一次API复核的reasoning中存在
至少一个显式加减等式不成立。请回到题目、选项和本轮证据，重新计算涉及该等式的选项，
同时复核其他选项。不得沿用错误等式，不得因首轮标签而保守取舍。
只输出严格JSON：{{"answer":"A或AC等","reasoning":"不超过180字，写出修正后的关键算式"}}

题目：{stem}
选项：
{option_text}
上一次API输出：
{response.content}
本轮原始文档背景证据：
{self._format_hits(context_hits, chars=700, limit=14)}
"""
            self._event(
                "prompt_routed",
                qid,
                route="qwen_arithmetic_consistency_repair",
                prompt_chars=len(repair_prompt),
                prompt_sha256=hashlib.sha256(
                    repair_prompt.encode("utf-8")
                ).hexdigest(),
                max_tokens=int(
                    self.profile.get("compact_joint_review_max_tokens", 500)
                ),
            )
            repaired = self.llm.chat(
                [{"role": "user", "content": repair_prompt}],
                max_tokens=int(
                    self.profile.get("compact_joint_review_max_tokens", 500)
                ),
            )
            review_usage.add(repaired.usage)
            repaired_parsed = self._parse_answer(
                repaired.content,
                answer_format,
            )
            repaired_answer = normalize_answer(
                str(repaired_parsed.get("answer", "")),
                answer_format,
            )
            repaired_reasoning = (
                str(repaired_parsed.get("reasoning", "")).strip()
                or repaired.content.strip()
            )
            if repaired_answer:
                answer = repaired_answer
                reasoning = repaired_reasoning
            self._event(
                "arithmetic_consistency_repaired",
                qid,
                answer=answer,
                api_reasoning=reasoning,
                usage=repaired.usage.to_dict(),
            )
        if answer_format in {"mcq", "tf"} and len(answer) != 1:
            answer = ""
        return answer, reasoning, review_usage

    @staticmethod
    def _has_explicit_arithmetic_contradiction(reasoning: str) -> bool:
        """在不提供答案的前提下检查 API 写出的基础等式。"""
        equations = re.findall(
            r"([-+]?\d+(?:\.\d+)?)\s*(?:元|%|个百分点|分)?\s*"
            r"(加|减|\+|-)\s*"
            r"([-+]?\d+(?:\.\d+)?)\s*(?:元|%|个百分点|分)?\s*"
            r"(?:等于|为|=)\s*"
            r"([-+]?\d+(?:\.\d+)?)",
            reasoning,
        )
        for left_text, operator, right_text, result_text in equations:
            left = float(left_text)
            right = float(right_text)
            stated = float(result_text)
            expected = left + right if operator in {"加", "+"} else left - right
            tolerance = max(0.011, abs(expected) * 0.0001)
            if abs(expected - stated) > tolerance:
                return True
        return False
