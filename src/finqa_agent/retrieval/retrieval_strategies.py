"""证据平衡与文档/页面扩展策略。"""

from __future__ import annotations

import hashlib
import json
import math
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
from ..core.entities import CALC_TERMS, DOMAIN_TERMS, NEGATIVE_TERMS, compact, dedupe, extract_entities
from .evidence_budget import assess_evidence_sufficiency, plan_evidence_budget
from .index import SearchHit, canonical
from ..runtime.io import normalize_answer, normalize_calculation_answer
from ..runtime.llm import Usage
from .planner import QueryPlan
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



class RetrievalStrategiesMixin:
    def _exact_term_hits(
        self,
        domain: str,
        doc_ids: Sequence[str],
        terms: Sequence[str],
        *,
        query: str,
        per_document: int = 2,
    ) -> List[SearchHit]:
        """在不使用语义或向量检索的情况下定位字面审计锚点。

        加引号的法律或保险术语属于高价值约束。BM25 仍可能把标题页或目录页排在
        实际包含该术语的条款之前。此确定性扫描会暴露对应原始页面，并让每份已绑定
        文档都获得同等的证据贡献机会。
        """
        allowed = {str(item) for item in doc_ids}
        normalized_terms = dedupe(
            [canonical(item) for item in terms if len(canonical(item)) >= 2],
            20,
        )
        if not normalized_terms:
            return []
        eligible_pages = [
            page
            for page in self.index.pages_by_domain.get(domain, [])
            if not allowed or page.doc_id in allowed
        ]
        page_frequency = {
            term: sum(
                1
                for page in eligible_pages
                if term in canonical(page.text)
            )
            for term in normalized_terms
        }
        grouped: Dict[str, List[SearchHit]] = {}
        for page in eligible_pages:
            page_text = canonical(page.text)
            matched = [term for term in normalized_terms if term in page_text]
            if not matched:
                continue
            positions = [
                (page_text.find(term), term)
                for term in matched
                if page_text.find(term) >= 0
            ]
            local_coverage = max(
                (
                    len(
                        {
                            other
                            for _, other in positions
                            if other in page_text[
                                max(0, position - 180):
                                min(len(page_text), position + 360)
                            ]
                        }
                    )
                    for position, _ in positions
                ),
                default=1,
            )
            # 优先使用稀有且更长的约束词，而不是高频标题词。
            # 每个组成部分仍保持透明的词法统计信号。
            score = 2000.0 + 250.0 * local_coverage + sum(
                (
                    100.0 + 15.0 * min(20, len(term))
                )
                / math.sqrt(max(1, page_frequency.get(term, 1)))
                + min(20, page_text.count(term))
                for term in matched
            )
            if (
                domain == "insurance"
                and "保险责任" in query
                and re.search(r"^#{1,4}\s*(?:\*\*)?保险责任", page.text, flags=re.M)
            ):
                score += 1200.0
            grouped.setdefault(page.doc_id, []).append(
                SearchHit(page, score, f"exact_term_anchor:{query}")
            )
        selected: List[SearchHit] = []
        for doc_id in sorted(grouped):
            selected.extend(
                sorted(grouped[doc_id], key=lambda hit: hit.score, reverse=True)[
                    : max(1, per_document)
                ]
            )
        return sorted(selected, key=lambda hit: hit.score, reverse=True)

    def _docs_named_across_clauses(
        self,
        domain: str,
        doc_ids: Sequence[str],
        text: str,
    ) -> List[str]:
        """在多实体题干中独立收集文档绑定。"""
        clauses = [
            item.strip()
            for item in re.split(r"[；;。]|(?<=[万元亿元])、|(?<=末)，", str(text))
            if len(compact(item)) >= 3
        ]
        matched: List[str] = []
        for clause in clauses:
            matched.extend(self._docs_named_in_option(domain, doc_ids, clause))
        return dedupe(matched, len(doc_ids) or 12)

    @staticmethod
    def _cap_hits_preserving_documents(
        hits: Sequence[SearchHit],
        cap: int,
        minimum_per_document: int = 1,
    ) -> List[SearchHit]:
        """限制上下文规模，同时不从比较中抹除任何文档。"""
        values = list(hits)
        if cap <= 0 or len(values) <= cap:
            return values
        protected: List[SearchHit] = []
        remainder: List[SearchHit] = []
        counts: Dict[str, int] = {}
        for hit in values:
            doc_id = hit.page.doc_id
            if counts.get(doc_id, 0) < max(1, minimum_per_document):
                counts[doc_id] = counts.get(doc_id, 0) + 1
                protected.append(hit)
            else:
                remainder.append(hit)
        effective_cap = max(cap, len(protected))
        return [*protected, *remainder][:effective_cap]

    def _option_balanced_hits(
        self,
        domain: str,
        doc_ids: Sequence[str],
        queries: Sequence[str],
        top_k: int,
    ) -> List[SearchHit]:
        """避免多文档比较被单份报告垄断。"""
        regular = self.index.search(domain, doc_ids, queries, top_k=top_k)
        if (
            domain not in {"financial_reports", "research", "financial_contracts"}
            or len(doc_ids) <= 1
        ):
            return regular
        seeded: List[SearchHit] = []
        seed_k = 2 if domain in {"financial_reports", "financial_contracts"} else 1
        for doc_id in doc_ids:
            seeded.extend(self.index.search(domain, [doc_id], queries, top_k=seed_k))
        result: List[SearchHit] = []
        seen: set[tuple[str, str]] = set()
        for hit in [*seeded, *regular]:
            key = (hit.page.doc_id, hit.page.page_id)
            if key in seen:
                continue
            seen.add(key)
            result.append(hit)
        limit = max(top_k, min(len(doc_ids), 5))
        if domain == "research":
            # 研报比较需要跨文档覆盖，但为每个选项播种所有候选报告会平方增长。
            # 四份独立排序报告可保留多样性，同时让证据胶囊可审计且成本可控。
            limit = max(limit, min(len(doc_ids), 4))
            # 候选发现顺序不是相关性排序。每份报告只播种一次，
            # 再按选项特定检索分数排序跨报告候选。
            result.sort(key=lambda hit: hit.score, reverse=True)
        if domain in {"financial_reports", "financial_contracts"}:
            limit = max(limit, top_k * 2, len(doc_ids) * 2)
        return result[:limit]

    def _financial_statement_structure_hits(
        self,
        doc_ids: Sequence[str],
        option_text: str,
    ) -> List[SearchHit]:
        """当词法 BM25 受样板文本干扰时，定位标准化报表标题。"""
        if not re.search(r"母公司|公司口径|合并口径", option_text):
            return []
        anchors: List[str] = []
        if "营业收入" in option_text:
            anchors.append("合并及公司利润表")
        if re.search(r"现金流|经营活动", option_text):
            anchors.append("合并及公司现金流量表")
        if not anchors:
            return []
        allowed = {str(item) for item in doc_ids}
        hits: List[SearchHit] = []
        for page in self.index.pages_by_domain.get("financial_reports", []):
            if page.doc_id not in allowed:
                continue
            header = canonical(page.text[:180])
            for anchor in anchors:
                if canonical(anchor) in header:
                    hits.append(
                        SearchHit(
                            page,
                            1000.0,
                            f"structural_anchor:{anchor}|{option_text}",
                        )
                    )
                    break
        return hits

    def _docs_named_in_option(
        self,
        domain: str,
        doc_ids: Sequence[str],
        option_text: str,
    ) -> List[str]:
        """使用从文档自身派生的别名把选项绑定到文档。"""
        binding_text = re.split(
            r"(?:条款|规定|明确|等待期|责任|若|因|中[，,：:]|中，被)",
            option_text,
            maxsplit=1,
        )[0]
        if len(compact(binding_text)) < 3:
            binding_text = option_text
        option_compact = canonical(binding_text)
        exact_scores: Dict[str, int] = {}
        all_domain_doc_ids = list(
            self.index.doc_title_aliases.get(domain, {}).keys()
        )
        candidate_doc_ids = (
            [str(doc_id) for doc_id in doc_ids]
            if doc_ids
            else all_domain_doc_ids
        )
        alias_doc_frequency: Dict[str, int] = {}
        for candidate_doc_id in all_domain_doc_ids:
            unique_aliases = {
                canonical(alias)
                for alias in self.index.doc_title_aliases.get(
                    domain,
                    {},
                ).get(candidate_doc_id, [])[:6]
                if len(canonical(alias)) >= 3
                and not re.search(r"(?:年度报告|报告全文|20\d{2}年)", canonical(alias))
            }
            for alias in unique_aliases:
                alias_doc_frequency[alias] = alias_doc_frequency.get(alias, 0) + 1
        for doc_id in candidate_doc_ids:
            aliases = self.index.doc_title_aliases.get(domain, {}).get(str(doc_id), [])
            matched_lengths = [
                len(canonical(alias))
                for alias in aliases[:6]
                if len(canonical(alias)) >= 3
                and canonical(alias) in option_compact
                and alias_doc_frequency.get(canonical(alias), 999) <= 2
            ]
            if matched_lengths:
                exact_scores[str(doc_id)] = max(matched_lengths)
        if domain == "financial_reports":
            # 年报标题常包含完整法律/行业名称，而题目使用正文引入的较短品牌前缀。
            # 从当前标题和选项动态推导绑定，而不是维护实体词典。
            generic_title_fragments = {
                "股份有限公司",
                "有限责任公司",
                "集团有限公司",
                "年度报告",
                "报告全文",
            }
            for doc_id in candidate_doc_ids:
                if doc_id in exact_scores:
                    continue
                aliases = self.index.doc_title_aliases.get(domain, {}).get(doc_id, [])
                fuzzy_lengths: List[int] = []
                for alias in aliases[:6]:
                    calias = canonical(alias)
                    block = SequenceMatcher(
                        None,
                        calias,
                        option_compact,
                        autojunk=False,
                    ).find_longest_match()
                    fragment = calias[block.a:block.a + block.size]
                    if (
                        block.a == 0
                        and len(fragment) >= 4
                        and fragment not in generic_title_fragments
                        and not re.fullmatch(r"20\d{2}年?", fragment)
                    ):
                        fuzzy_lengths.append(len(fragment))
                if fuzzy_lengths:
                    exact_scores[str(doc_id)] = max(fuzzy_lengths)
            matched_docs = list(exact_scores)
            option_years = set(re.findall(r"20\d{2}", option_text))
            if option_years:
                year_filtered = [
                    doc_id
                    for doc_id in matched_docs
                    if any(year in doc_id for year in option_years)
                ]
                if year_filtered:
                    matched_docs = year_filtered
            return matched_docs
        option_bigrams = {
            option_compact[index:index + 2]
            for index in range(max(0, len(option_compact) - 1))
        }
        generic_brand_terms = {
            "中国", "股份", "公司", "有限", "保险", "财产", "人寿", "在线",
        }
        ranked: List[tuple[int, int, int, str]] = []
        for doc_id in candidate_doc_ids:
            aliases = self.index.doc_title_aliases.get(domain, {}).get(doc_id, [])
            combined = canonical(" ".join(aliases))
            overlap = sum(term in combined for term in option_bigrams)
            exact_length = exact_scores.get(doc_id, 0)
            if not exact_length and overlap < 2:
                continue
            issuer_text = canonical(" ".join(aliases[:4]))
            brand_overlap = sum(
                term in issuer_text
                for term in option_bigrams
                if term not in generic_brand_terms
            )
            ranked.append((brand_overlap, overlap, exact_length, doc_id))
        if not ranked:
            return []
        best_brand = max(item[0] for item in ranked)
        brand_candidates = [item for item in ranked if item[0] == best_brand]
        best_overlap = max(item[1] for item in brand_candidates)
        overlap_candidates = [item for item in brand_candidates if item[1] == best_overlap]
        best_exact = max(item[2] for item in overlap_candidates)
        return [
            doc_id
            for _, _, exact, doc_id in overlap_candidates
            if exact == best_exact
        ]

    def _with_adjacent_pages(
        self,
        domain: str,
        hits: Sequence[SearchHit],
    ) -> List[SearchHit]:
        """为表格和多期间报表补充下一页。"""
        pages = self.index.pages_by_domain.get(domain, [])
        positions = {
            (page.doc_id, page.page_id): index
            for index, page in enumerate(pages)
        }
        result: List[SearchHit] = []
        seen: set[tuple[str, str]] = set()
        for hit in hits:
            key = (hit.page.doc_id, hit.page.page_id)
            position = positions.get(key)
            neighbors: List[SearchHit] = []
            if position is not None and position > 0:
                previous = pages[position - 1]
                if previous.doc_id == hit.page.doc_id:
                    neighbors.append(SearchHit(previous, hit.score * 0.75, hit.query))
            neighbors.append(hit)
            if position is not None and position + 1 < len(pages):
                following = pages[position + 1]
                if following.doc_id == hit.page.doc_id:
                    neighbors.append(SearchHit(following, hit.score * 0.8, hit.query))
            for neighbor in neighbors:
                neighbor_key = (neighbor.page.doc_id, neighbor.page.page_id)
                if neighbor_key in seen:
                    continue
                seen.add(neighbor_key)
                result.append(neighbor)
        # 让真实检索命中排在折扣邻页之前。
        # 后续硬截断时，邻页不能挤掉它本应补充上下文的精确页面。
        return sorted(result, key=lambda hit: hit.score, reverse=True)
