"""Qwen 生成的计算程序路由及受限执行。"""

from __future__ import annotations

import hashlib
import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Sequence

from ..core.agent_types import AnswerResult
from ..reasoning.answer_reconciliation import (
    audit_full_year_dividend_difference,
    explicit_segment_verdict,
    reconcile_cross_option_numeric_claims,
    repair_cross_option_ranking_answer,
    repair_numeric_self_consistency_verdict,
    repair_parallel_duty_omission_verdict,
    repair_zero_balance_condition_verdict,
    should_apply_reasoning_answer,
)
from .calculation_reconciliation import (
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
from ..core.entities import CALC_TERMS, DOMAIN_TERMS, NEGATIVE_TERMS, compact, dedupe, extract_entities
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
from ..reasoning.reasoning_expansion import expand_reasoning_from_counted_prompt



class CalculationRoutesMixin:
    def _run_calculation_tool(self, question: Dict[str, object], hits: Sequence[SearchHit]) -> tuple[Dict[str, object] | None, Usage]:
        domain = str(question.get("domain", ""))
        stem = str(question.get("question", ""))
        prompt_hits = list(hits)
        if domain in {
            "financial_reports",
            "financial_contracts",
            "insurance",
            "research",
        }:
            doc_ids = dedupe([hit.page.doc_id for hit in hits], 12)
            named_doc_ids = self._docs_named_across_clauses(
                domain,
                doc_ids,
                stem,
            )
            if not named_doc_ids:
                named_doc_ids = self._docs_named_in_option(domain, doc_ids, stem)
            if named_doc_ids:
                doc_ids = named_doc_ids
            years = dedupe(re.findall(r"20\d{2}年?", stem), 6)
            metric_terms = [
                term for term in DOMAIN_TERMS.get(domain, []) if term in stem
            ]
            metric_terms.extend(
                term
                for term in [
                    "营业收入",
                    "经营活动产生的现金流量净额",
                    "归属于上市公司股东的净利润",
                    "基本每股收益",
                    "现金分红",
                    "毛利率",
                    "身故保险金",
                    "退保",
                    "解除合同",
                    "现金价值",
                    "个人账户价值",
                    "保单账户价值",
                    "累计收益",
                    "手续费",
                ]
                if term in stem
            )
            metric_terms = dedupe(metric_terms, 8)
            targeted_hits: List[SearchHit] = []
            supplemental_targeted_hits: List[SearchHit] = []
            for doc_id in doc_ids:
                if domain == "financial_reports" and len(metric_terms) >= 2:
                    combined_query = " ".join(
                        dedupe([*metric_terms, *years, "主要会计数据", "合并"], 12)
                    )
                    targeted_hits.extend(
                        self.index.search(
                            domain,
                            [doc_id],
                            [combined_query],
                            top_k=3,
                        )
                    )
                for term in metric_terms:
                    query = " ".join(dedupe([term, *years], 8))
                    if query:
                        targeted_hits.extend(
                            self.index.search(
                                domain,
                                [doc_id],
                                [
                                    query,
                                    f"{query} 主要会计数据 合并",
                                ],
                                top_k=2,
                            )
                        )
                supplemental_queries: List[str] = []
                if domain == "financial_contracts" and "毛利率" in stem:
                    metric_subjects = dedupe(
                        re.findall(
                            r"([\u4e00-\u9fff]{2,12}(?:板块|业务|产品)).{0,10}毛利率",
                            stem,
                        ),
                        6,
                    )
                    supplemental_queries.extend(
                        f"{subject} 毛利率 主营业务毛利率"
                        for subject in metric_subjects
                    )
                if domain == "financial_reports" and "现金分红" in stem:
                    supplemental_queries.extend(
                        [
                            "每10股 中期 现金分红",
                            "每股 中期 现金分红 全年",
                        ]
                    )
                if domain == "insurance" and "身故保险金" in stem:
                    supplemental_queries.append(
                        "身故保险金 给付比例 年龄 基本保险金额"
                    )
                if domain == "insurance" and re.search(r"退保|解除", stem):
                    supplemental_queries.extend(
                        [
                            "退保费用比例 保单年度 个人账户价值",
                            "现金价值 解除合同 累计收益 比例",
                        ]
                    )
                if domain == "research" and "单车带电量" in stem:
                    supplemental_queries.extend(
                        [
                            "乘用车 单车带电量 2025 2026 kWh",
                            "乘用车单车带电量 历史实际值 预测值",
                        ]
                    )
                if supplemental_queries:
                    supplemental_targeted_hits.extend(
                        self.index.search(
                            domain,
                            [doc_id],
                            supplemental_queries,
                            top_k=min(4, len(supplemental_queries) * 2),
                        )
                    )
            all_targeted_hits = self._merge_hits(
                supplemental_targeted_hits,
                targeted_hits,
            )
            if all_targeted_hits:
                groups: Dict[str, List[SearchHit]] = {}
                for hit in all_targeted_hits:
                    groups.setdefault(hit.page.doc_id, []).append(hit)
                balanced_targeted_hits: List[SearchHit] = []
                rank = 0
                while len(balanced_targeted_hits) < 12 and any(
                    rank < len(group) for group in groups.values()
                ):
                    for group in groups.values():
                        if rank < len(group):
                            balanced_targeted_hits.append(group[rank])
                            if len(balanced_targeted_hits) >= 12:
                                break
                    rank += 1
                targeted_keys = {
                    (hit.page.doc_id, hit.page.page_id)
                    for hit in balanced_targeted_hits
                }
                prompt_hits = [
                    *balanced_targeted_hits,
                    *[
                        hit
                        for hit in prompt_hits
                        if (hit.page.doc_id, hit.page.page_id)
                        not in targeted_keys
                    ],
                ]
                self._event(
                    "calculation_metric_retrieval_completed",
                    str(question.get("qid", "")),
                    metric_terms=metric_terms,
                    doc_count=len(doc_ids),
                    targeted_hit_count=len(all_targeted_hits),
                    balanced_hit_count=len(balanced_targeted_hits),
                    hits=[hit.to_dict(chars=0) for hit in balanced_targeted_hits],
                )
        if (
            domain == "regulatory"
            and re.search(r"扣分|计分|得分|评分", stem)
            and re.search(r"分支机构|子公司|母公司|负责人|管理人员|从业人员", stem)
        ):
            doc_ids = dedupe([hit.page.doc_id for hit in hits], 12)
            object_terms = dedupe(
                re.findall(
                    r"分支机构|子公司|母公司|负责人|管理人员|从业人员|公司|机构",
                    stem,
                ),
                12,
            )
            modifier_queries = dedupe(
                [
                    *[
                        f"{term} 分别 减半 扣分 上限 按上述原则"
                        for term in object_terms
                    ],
                    "不同对象 分别 减半 扣分 上限 计算原则",
                ],
                12,
            )
            modifier_hits = self.index.search(
                domain,
                doc_ids,
                modifier_queries,
                top_k=8,
            )
            seen_pages: set[tuple[str, str]] = set()
            prompt_hits = []
            for hit in [*modifier_hits, *hits]:
                key = (hit.page.doc_id, hit.page.page_id)
                if key in seen_pages:
                    continue
                seen_pages.add(key)
                prompt_hits.append(hit)
            self._event(
                "calculation_modifier_retrieval_completed",
                str(question.get("qid", "")),
                queries=modifier_queries,
                hits=[hit.to_dict(chars=0) for hit in modifier_hits],
            )
        disclosed_metric_anchors: List[Dict[str, str]] = []
        table_value_anchors: List[Dict[str, str]] = []
        seen_table_anchors: set[tuple[str, str, str]] = set()
        for hit in prompt_hits:
            for line in hit.page.text.splitlines():
                compact_line = re.sub(r"\s+", " ", line).strip()
                if (
                    "|" not in compact_line
                    or len(re.findall(r"\d+(?:\.\d+)?%?", compact_line)) < 3
                ):
                    continue
                key = (hit.page.doc_id, hit.page.page_id, compact_line)
                if key in seen_table_anchors:
                    continue
                seen_table_anchors.add(key)
                row_position = hit.page.text.find(line)
                anchor_context_chars = int(
                    self.profile.get("calculation_table_anchor_context_chars", 200)
                )
                anchor_row_chars = int(
                    self.profile.get("calculation_table_anchor_row_chars", 400)
                )
                anchor_limit = int(
                    self.profile.get("calculation_table_anchor_limit", 10)
                )
                row_context = re.sub(
                    r"\s+",
                    " ",
                    hit.page.text[
                        max(0, row_position - anchor_context_chars):row_position
                    ],
                ).strip()
                table_value_anchors.append(
                    {
                        "doc_id": hit.page.doc_id,
                        "page_id": hit.page.page_id,
                        "context": row_context[-anchor_context_chars:],
                        "row": compact_line[:anchor_row_chars],
                    }
                )
                if len(table_value_anchors) >= max(40, anchor_limit):
                    break
            if len(table_value_anchors) >= 40:
                break
        table_value_anchors = rank_table_anchors(
            table_value_anchors,
            stem,
            int(self.profile.get("calculation_table_anchor_limit", 10)),
        )
        if domain == "financial_reports":
            seen_anchors: set[tuple[str, str, str]] = set()
            for hit in prompt_hits:
                for match in re.finditer(
                    r"(?:每\s*10\s*股|每股)[^。\n|]{0,100}?"
                    r"\d+(?:\.\d+)?\s*元",
                    hit.page.text,
                ):
                    context_start = max(
                        0,
                        hit.page.text.rfind("。", 0, match.start()) + 1,
                        match.start() - 140,
                    )
                    next_stop = hit.page.text.find("。", match.end())
                    context_end = (
                        min(len(hit.page.text), match.end() + 140)
                        if next_stop < 0
                        else min(next_stop + 1, match.end() + 140)
                    )
                    anchor = re.sub(
                        r"\s+",
                        " ",
                        hit.page.text[context_start:context_end],
                    ).strip()
                    key = (hit.page.doc_id, hit.page.page_id, anchor)
                    if key in seen_anchors:
                        continue
                    seen_anchors.add(key)
                    disclosed_metric_anchors.append(
                        {
                            "doc_id": hit.page.doc_id,
                            "page_id": hit.page.page_id,
                            "quote": anchor,
                        }
                    )
                    if len(disclosed_metric_anchors) >= 30:
                        break
                if len(disclosed_metric_anchors) >= 30:
                    break
        if disclosed_metric_anchors:
            self._event(
                "calculation_disclosed_metrics_anchored",
                str(question.get("qid", "")),
                anchors=disclosed_metric_anchors,
            )
        if table_value_anchors:
            self._event(
                "calculation_table_values_anchored",
                str(question.get("qid", "")),
                anchors=table_value_anchors,
            )
        calculation_chars_key = (
            "insurance_calculation_prompt_chars"
            if domain == "insurance"
            else (
                "regulatory_calculation_prompt_chars"
                if domain == "regulatory"
                else "calculation_prompt_chars"
            )
        )
        calculation_hits_key = (
            "insurance_calculation_prompt_hits"
            if domain == "insurance"
            else (
                "regulatory_calculation_prompt_hits"
                if domain == "regulatory"
                else "calculation_prompt_hits"
            )
        )
        calculation_prompt_chars = int(
            self.profile.get(
                calculation_chars_key,
                1600 if domain == "insurance" else 950,
            )
        )
        calculation_prompt_hits = int(
            self.profile.get(calculation_hits_key, 10)
        )
        if bool(self.profile.get("adaptive_calculation_evidence", False)):
            represented_docs = {
                hit.page.doc_id for hit in prompt_hits
            }
            multi_source_or_operand = bool(
                len(represented_docs) >= 2
                and (
                    len(re.findall(r"20\d{2}", stem)) >= 2
                    or re.search(
                        r"分别|合计|总和|排序|由高到低|差额|相比|比较|"
                        r"[；;].*[；;]|(?:与|和).{0,30}(?:与|和)",
                        stem,
                    )
                )
            )
            if multi_source_or_operand:
                calculation_prompt_chars = max(
                    calculation_prompt_chars,
                    int(
                        self.profile.get(
                            "adaptive_calculation_prompt_chars",
                            850,
                        )
                    ),
                )
                calculation_prompt_hits = max(
                    calculation_prompt_hits,
                    int(
                        self.profile.get(
                            "adaptive_calculation_prompt_hits",
                            9,
                        )
                    ),
                )
                self._event(
                    "calculation_evidence_budget_expanded",
                    str(question.get("qid", "")),
                    reason="multi_source_or_operand",
                    represented_doc_count=len(represented_docs),
                    prompt_chars_per_hit=calculation_prompt_chars,
                    prompt_hit_limit=calculation_prompt_hits,
                )
        prompt = (
            "你是金融问答系统的计算代码生成器。请只基于题目和证据生成一段安全 Python 代码，"
            "把最终计算结果赋值给变量 answer。不要读取文件，不要联网，不要 import。\n"
            "answer必须直接符合题目要求的最终格式；多项结果使用中文分号连接，不要使用字典、列表或解释性前缀。\n"
            "严格区分百分比的百分点变化(a-b)与相对变化率((a-b)/b)，按题目措辞选择公式。\n"
            "若总量=规模×强度且题设规模不变，增速应直接用强度之比计算，优先使用未四舍五入的驱动变量。\n"
            "若表格含历史实际值与预测值，题目说“从某年水平提升”时必须使用该年的历史实际值作"
            "基期；重复年份表头、旧预测或下一年预测不得替代历史实际值。应结合A/E标记、表题、"
            "列顺序以及相邻行的实际同比关系校验列对齐。\n"
            "若题目直接引用报告中展示的某年驱动指标“水平”，应使用表格明确展示的该指标值；"
            "不得用已四舍五入的总量除以规模反推隐藏小数，再覆盖直接展示的驱动指标。\n"
            "题目要求使用原始金额时，必须直接检索并使用原始金额，不得用已四舍五入的比率、占比或"
            "其他指标反推替代。所有中间量保持完整精度，只在最终答案按题意舍入；题目未明确精度时"
            "百分数默认保留两位小数。\n"
            "若题目明确指定使用年报披露的比率、每股或每10股数值，应直接采用该披露值，不要用"
            "总额重新计算并覆盖披露口径；只有题目明确要求使用原始金额时才从原始金额计算。\n"
            "处理全年分红时必须根据披露锚点周围的“中期、年末、本次、剩余待分配、全年”等"
            "期间限定判断数值关系；中期值和剩余年末值应相加，已明确写作全年合计的值不得再加。\n"
            "表格题必须同时核对表格标题和行名：构成占比、收入占比、毛利润占比均不等于毛利率；"
            "题目问毛利率时，只能采用明确位于毛利率表或由毛利润÷收入算出的数值。\n"
            "单位换算必须逐个变量处理：千元转亿元除以100000；元乘股数得到元，转亿元除以"
            "100000000。禁止把不同原始单位的两个金额先直接相减或共用同一个换算除数。\n"
            "保险计算必须从各合同分别提取给付或退还公式、年龄/年度系数、已领取金额、收益计入比例"
            "及手续费；不得仅把题目给定的账户价值、现金价值或已交保费直接相加。\n"
            "监管计分或处罚计算中，若措施分别作用于机构本体、分支机构、负责人等不同对象，"
            "必须分别从证据提取各对象对应的扣分、权重或系数，禁止因措施名称相同而默认数值相同。\n"
            "全年现金分红计算中，若年度目标先扣除已实施中期分红，再按剩余待分配金额给出"
            "每股或每10股报价，该报价是剩余待派而非全年合计；全年必须加回中期且只加一次。\n"
            "日期期限题必须优先用 date 与 timedelta 做日历运算：若题目明确受理/起始当日不计入，"
            "则截止日=date(题给日期)+timedelta(days=N)，不要手写逐月循环，也不要再额外加一天。\n"
            "如果证据不足以计算，输出：answer = ''。\n\n"
            f"题目：{question.get('question', '')}\n"
            f"文档直接披露的每股口径锚点："
            f"{json.dumps(disclosed_metric_anchors, ensure_ascii=False)}\n"
            f"文档候选表格数值行："
            f"{json.dumps(table_value_anchors, ensure_ascii=False)}\n"
            f"证据：\n{self._format_hits(prompt_hits, chars=calculation_prompt_chars, limit=calculation_prompt_hits)}"
        )
        if self.profile.get("compact_domain_scoped_prompt", False):
            calculation_evidence = self._format_hits(
                prompt_hits,
                chars=calculation_prompt_chars,
                limit=calculation_prompt_hits,
            )
            prompt = build_scoped_calculation_prompt(
                domain=domain,
                stem=str(question.get("question", "")),
                disclosed_metric_anchors=disclosed_metric_anchors,
                table_value_anchors=table_value_anchors,
                evidence=calculation_evidence,
            )
        max_calculation_prompt_chars = int(
            self.profile.get(
                "max_research_calculation_prompt_chars"
                if domain == "research"
                else "max_calculation_prompt_chars",
                0,
            )
        )
        if (
            max_calculation_prompt_chars > 0
            and len(prompt) > max_calculation_prompt_chars
        ):
            self._event(
                "calculation_prompt_skipped",
                str(question.get("qid", "")),
                prompt_chars=len(prompt),
                max_prompt_chars=max_calculation_prompt_chars,
                reason="joint_answer_is_more_token_efficient",
            )
            return None, Usage()
        self._event(
            "prompt_routed",
            str(question.get("qid", "")),
            route="qwen_calculation_program",
            prompt_chars=len(prompt),
            prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            evidence_count=min(len(prompt_hits), 10),
            max_tokens=1600,
        )
        response = self.llm.chat([{"role": "user", "content": prompt}], max_tokens=1600)
        total_usage = Usage()
        total_usage.add(response.usage)
        code = self._extract_code(response.content)
        literal_only_for_multi_operand = bool(
            re.search(
                r"合计|总计|总和|[二三四五六七八九十]份|"
                r"(?:20\d{2}.{0,30}){3,}|分别.{0,30}(?:20\d{2}|各期)",
                stem,
            )
            and re.fullmatch(
                r"\s*answer\s*=\s*['\"]?[-+]?\d+(?:\.\d+)?['\"]?\s*",
                code,
            )
        )
        if literal_only_for_multi_operand:
            provenance_prompt = (
                prompt
                + "\n\n上一次程序只有一个无来源的答案常量，无法审计多项合计。"
                "请为每个合同、主体或期间分别建立变量，写出证据支持的公式或取值，"
                "再显式相加得到answer。禁止直接把最终常量赋给answer。\n上一次程序：\n"
                + code
            )
            self._event(
                "calculation_operand_provenance_repair_requested",
                str(question.get("qid", "")),
                prompt_chars=len(provenance_prompt),
                prompt_sha256=hashlib.sha256(
                    provenance_prompt.encode("utf-8")
                ).hexdigest(),
            )
            provenance_response = self.llm.chat(
                [{"role": "user", "content": provenance_prompt}],
                max_tokens=1600,
            )
            total_usage.add(provenance_response.usage)
            provenance_code = self._extract_code(provenance_response.content)
            provenance_execution = self.executor.execute(provenance_code)
            if (
                provenance_code
                and provenance_execution.ok
                and provenance_execution.answer != ""
                and not re.fullmatch(
                    r"\s*answer\s*=\s*['\"]?[-+]?\d+(?:\.\d+)?['\"]?\s*",
                    provenance_code,
                )
            ):
                response = provenance_response
                code = provenance_code
        if (
            domain in {"financial_reports", "financial_contracts"}
            and self.profile.get("calculation_structured_anchor_retry", True)
            and re.fullmatch(r"\s*answer\s*=\s*['\"]{2}\s*", code)
            and (table_value_anchors or disclosed_metric_anchors)
        ):
            evidence_retry_prompt = (
                prompt
                + "\n\n上一次程序返回空答案，但结构化披露锚点中已有计算所需数值。"
                "请严格按题目指标、期间顺序和公式重新生成完整Python程序。"
                "不得返回空字符串，不得把占比、构成比例与题目所问指标混用。"
            )
            self._event(
                "calculation_structured_anchor_retry_requested",
                str(question.get("qid", "")),
                prompt_chars=len(evidence_retry_prompt),
                prompt_sha256=hashlib.sha256(
                    evidence_retry_prompt.encode("utf-8")
                ).hexdigest(),
            )
            retry_response = self.llm.chat(
                [{"role": "user", "content": evidence_retry_prompt}],
                max_tokens=1600,
            )
            total_usage.add(retry_response.usage)
            retry_code = self._extract_code(retry_response.content)
            retry_execution = self.executor.execute(retry_code)
            if (
                retry_code
                and retry_execution.ok
                and retry_execution.answer != ""
            ):
                response = retry_response
                code = retry_code
        direct_quote_recomputed = bool(
            re.search(r"每\s*10\s*股", stem)
            and re.search(r"/[^\n]{1,80}\*\s*10", code)
            and re.search(r"每\s*10\s*股.{0,40}\d+(?:\.\d+)?", prompt)
        )
        if (
            (
                "原始" in stem
                and re.search(r"反推|由.{0,40}(?:占比|比例).{0,20}(?:推|算)", code)
            )
            or direct_quote_recomputed
        ):
            violation = (
                "程序用总额和股数重新计算并覆盖了文档直接披露的每股/每10股口径。"
                if direct_quote_recomputed
                else "程序使用占比或比例反推了题目要求的原始金额。"
            )
            repair_prompt = (
                prompt
                + "\n\n上一次程序存在口径错误："
                + violation
                + "请优先采用证据直接披露且与题目指定口径一致的值；仅在没有直接披露时才换算。"
                "重新生成完整Python程序，不得沿用错误口径。\n上一次程序：\n"
                + code
            )
            self._event(
                "calculation_original_value_repair_requested",
                str(question.get("qid", "")),
                prompt_chars=len(repair_prompt),
                prompt_sha256=hashlib.sha256(
                    repair_prompt.encode("utf-8")
                ).hexdigest(),
            )
            repaired_response = self.llm.chat(
                [{"role": "user", "content": repair_prompt}],
                max_tokens=1600,
            )
            total_usage.add(repaired_response.usage)
            repaired_code = self._extract_code(repaired_response.content)
            repaired_execution = self.executor.execute(repaired_code)
            repaired_still_recomputes = bool(
                direct_quote_recomputed
                and re.search(r"/[^\n]{1,80}\*\s*10", repaired_code)
            )
            if repaired_still_recomputes:
                audit_prompt = (
                    repair_prompt
                    + "\n\n第二次审计：你仍然用总额÷股数×10重算了已有直接披露的每10股数值。"
                    "这是禁止的，即使重算值只相差0.01也不能覆盖披露口径。凡证据已直接写明"
                    "每股或每10股数值，必须把该披露数字直接赋给变量。请再次输出完整Python程序。"
                    "\n待审计程序：\n"
                    + repaired_code
                )
                self._event(
                    "calculation_disclosed_metric_audit_requested",
                    str(question.get("qid", "")),
                    prompt_chars=len(audit_prompt),
                    prompt_sha256=hashlib.sha256(
                        audit_prompt.encode("utf-8")
                    ).hexdigest(),
                )
                audited_response = self.llm.chat(
                    [{"role": "user", "content": audit_prompt}],
                    max_tokens=1400,
                )
                total_usage.add(audited_response.usage)
                audited_code = self._extract_code(audited_response.content)
                audited_execution = self.executor.execute(audited_code)
                if (
                    audited_code
                    and audited_execution.ok
                    and audited_execution.answer != ""
                    and not re.search(r"/[^\n]{1,80}\*\s*10", audited_code)
                ):
                    repaired_response = audited_response
                    repaired_code = audited_code
                    repaired_execution = audited_execution
            repaired_still_recomputes = bool(
                direct_quote_recomputed
                and re.search(r"/[^\n]{1,80}\*\s*10", repaired_code)
            )
            if (
                repaired_code
                and repaired_execution.ok
                and repaired_execution.answer != ""
                and not repaired_still_recomputes
            ):
                response = repaired_response
                code = repaired_code
        result = self.executor.execute(code)
        return {
            "llm_code": code,
            "api_response": response.content,
            "execution": result.to_dict(),
            "usage": total_usage.to_dict(),
        }, total_usage

    @staticmethod
    def _extract_code(text: str) -> str:
        match = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.S | re.I)
        if match:
            return match.group(1).strip()
        raw = str(text or "").strip()
        if re.search(r"(?m)^\s*answer\s*=", raw):
            try:
                compile(raw, "<qwen_calc_candidate>", "exec")
            except SyntaxError:
                pass
            else:
                # 模型经常返回不带代码围栏的合法多行程序。
                # 保留所有前置赋值；只截取最后一行 ``answer =`` 会导致 NameError。
                return raw
        assignments = re.findall(r"(?m)^\s*answer\s*=.*$", text)
        if assignments:
            return assignments[-1].strip()
        return text.strip()
