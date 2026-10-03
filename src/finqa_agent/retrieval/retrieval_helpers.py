"""底层检索、审计、证据格式化与基础 Prompt。"""

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



class RetrievalHelpersMixin:
    def _event(self, event: str, qid: str, **payload: object) -> None:
        if event == "prompt_routed":
            self.llm.set_route_context(str(payload.get("route", "unclassified")))
        if self.audit is not None:
            self.audit.event(event, qid=qid, **payload)

    def _retrieve(self, domain: str, doc_ids: Sequence[str], plan: QueryPlan) -> List[SearchHit]:
        queries = plan.all_queries()
        if not queries:
            queries = [" ".join(plan.useful_entities[:12])]
        return self.index.search(domain, doc_ids, queries, top_k=int(self.profile.get("retrieval_top_k", 24)))

    @staticmethod
    def _merge_hits(a: Sequence[SearchHit], b: Sequence[SearchHit]) -> List[SearchHit]:
        merged: List[SearchHit] = []
        seen: set[tuple[str, str]] = set()
        for hit in sorted(list(a) + list(b), key=lambda x: x.score, reverse=True):
            key = (hit.page.doc_id, hit.page.page_id)
            if key in seen:
                continue
            seen.add(key)
            merged.append(hit)
        return merged

    def _quality(self, question: Dict[str, object], plan: QueryPlan, hits: Sequence[SearchHit], doc_ids: Sequence[str]) -> Dict[str, object]:
        reasons: List[str] = []
        options = question.get("options") or {}
        evidence_text = "\n".join(h.page.text for h in hits[:20])
        for key, value in sorted(options.items()):
            opt_terms = extract_entities(str(value), str(question.get("domain", "")))
            if opt_terms and not any(term in evidence_text for term in opt_terms[:4]):
                reasons.append(f"option_{key}_low_coverage")
        if not hits:
            reasons.append("empty_retrieval")
        if doc_ids and len({h.page.doc_id for h in hits}) < min(len(doc_ids), max(1, len(doc_ids))):
            reasons.append("doc_coverage_low")
        if self._needs_calculation(question):
            qtext = self.index.question_text(question)
            if not re.search(r"\d", evidence_text) or not any(term in qtext + evidence_text for term in CALC_TERMS):
                reasons.append("calculation_basis_low")
        if any(term in self.index.question_text(question) for term in NEGATIVE_TERMS):
            if not any(term in evidence_text for term in NEGATIVE_TERMS):
                reasons.append("negative_term_low")
        return {"needs_repair": bool(reasons), "reasons": dedupe(reasons, 12), "hit_count": len(hits)}

    def _repair_queries(self, question: Dict[str, object], plan: QueryPlan, quality: Dict[str, object]) -> List[str]:
        domain = str(question.get("domain", ""))
        qtext = self.index.question_text(question)
        entities = dedupe([*plan.useful_entities, *extract_entities(qtext, domain)], 50)
        queries = [" ".join(entities[:12])]
        for key, value in sorted((question.get("options") or {}).items()):
            opt_terms = extract_entities(str(value), domain)
            queries.append(" ".join(dedupe([*opt_terms, *entities[:8]], 14)))
        if any("negative" in r for r in quality.get("reasons", [])):
            queries.append(" ".join(dedupe([*entities[:10], *NEGATIVE_TERMS], 18)))
        if any("calculation" in r for r in quality.get("reasons", [])):
            queries.append(" ".join(dedupe([*entities[:10], "公式", "比例", "金额", "合计", "计算"], 18)))
        return [q for q in dedupe(queries, 20) if q]

    def _answer_prompt(
        self,
        question: Dict[str, object],
        discovery: Dict[str, object],
        plan: QueryPlan,
        hits: Sequence[SearchHit],
        calc_results: Sequence[Dict[str, object]],
        compact: bool,
    ) -> str:
        options = question.get("options") or {}
        option_text = "\n".join(f"{k}. {v}" for k, v in sorted(options.items())) or "无选项"
        answer_format = str(question.get("answer_format", "mcq"))
        format_instruction = {
            "mcq": "单选题，answer 只能是 A/B/C/D 中的一个。",
            "multi": "多选题，answer 必须是按字母升序排列的选项组合，例如 AC 或 BCD。",
            "tf": "判断题，answer 只能是 A 或 B，按题目选项语义判断。",
            "calc": "计算题，answer 只写最终数字或题目要求的简短结果。",
            "extract": "抽取题，answer 只写题目要求抽取的内容。",
        }.get(answer_format, "按题目要求回答。")
        judgment_instruction = ""
        if answer_format in {"mcq", "multi", "tf"}:
            judgment_instruction = (
                '\n逐项判断每个选项，并在 option_judgments 中把每项标记为 '
                '"supported"、"contradicted" 或 "unknown"，同时列出 evidence_ids。'
                "最终答案必须与逐项判断一致。"
            )
        return f"""你是金融长文档问答 Agent。请严格基于给出的检索证据回答。

系统边界：
- 若选项涉及具体主体，必须确认该主体对应证据，不能用其他主体证据替代。
- 多选题逐项判断，注意“不得、无需、不承担、除外、不适用”等否定限制。
- 计算题优先使用证据中的公式和数值；若有本地 Python 执行结果，仍需核验证据是否支持。
- 严格区分百分点变化(a-b)与相对变化率((a-b)/b)，不得把二者混用。
- 涉及全年现金分红时，区分全年总额、本次年末分配额和已实施中期分红；按题目口径判断是否需要相加。
- 研究类综合判断允许对多份证据作有依据的归纳，不要求原文逐字出现选项结论，但不得补充证据外事实。
- 只能使用证据，不得凭模型记忆补充具体事实。证据不足时将 confidence 标为 low。

输出严格 JSON：
{{
  "answer": "...",
  "confidence": "high|medium|low",
  "option_judgments": {{"A": {{"verdict": "supported|contradicted|unknown", "evidence_ids": [1]}}}},
  "reasoning": "用简短句子说明关键证据编号和取舍"
}}
先在内部完成全部核对，只输出一次最终 JSON；不要在输出中反复推翻或修改结论。
reasoning 不超过180个汉字，answer 与 option_judgments 必须完全一致。
{judgment_instruction}

领域：{question.get("domain", "")}
题型：{answer_format}
答案格式要求：{format_instruction}

题目：
{question.get("question", "")}

选项：
{option_text}

候选文档ID（仅由当前文档集合定位）：
{json.dumps(discovery.get("inferred_doc_ids", []), ensure_ascii=False)}

计算工具结果：
{json.dumps(calc_results, ensure_ascii=False)[:2500]}

检索证据：
{self._format_hits(hits, chars=int(self.profile.get("page_chars", 2200)), limit=int(self.profile.get("prompt_pages", 18)))}
"""

    @staticmethod
    def _format_hits(
        hits: Sequence[SearchHit],
        chars: int,
        limit: int,
        focus_query: str = "",
    ) -> str:
        rows = []
        for i, hit in enumerate(hits[:limit], 1):
            aliases = dedupe(hit.page.aliases, 3)
            if str(hit.query).startswith("structural_anchor:"):
                text = hit.page.text[: max(chars, 900)]
            else:
                # 检索 query 常包含完整题干和长实体别名。
                # 对选项级证据胶囊，选项本身更能解释为何选择某段摘录。
                # 这也能避免通用标题词挤掉长页面末尾的精确条款。
                text = RetrievalHelpersMixin._focused_excerpt(
                    hit.page.text,
                    focus_query or hit.query,
                    chars,
                )
            rows.append(
                f"[{i}] domain={hit.page.domain} doc_id={hit.page.doc_id} page={hit.page.page_id} "
                f"doc_aliases={json.dumps(aliases, ensure_ascii=False)} score={hit.score:.2f}\n{text}"
            )
        return "\n\n".join(rows) or "未检索到证据"

    @staticmethod
    def _focused_excerpt(text: str, query: str, chars: int) -> str:
        if len(str(text)) <= chars:
            return str(text)
        raw_lines = [re.sub(r"\s+", " ", line).strip() for line in str(text).splitlines()]
        lines = [line for line in raw_lines if line]
        if not lines:
            return ""
        chinese_chunks = re.findall(r"[\u4e00-\u9fff]{2,}", query)
        quoted_terms = [
            compact(item)
            for item in re.findall(r"[“\"']([^”\"']{2,40})[”\"']", query)
            if compact(item)
        ]
        chinese_ngrams = [
            chunk[index:index + size]
            for chunk in chinese_chunks
            for size in (3, 2)
            for index in range(max(0, len(chunk) - size + 1))
        ]
        terms = dedupe(
            [
                *extract_entities(query),
                *re.findall(r"[\u4e00-\u9fff]{2,10}|[A-Za-z0-9._%-]{2,20}", query),
                *chinese_ngrams,
            ],
            160,
        )
        terms = [term for term in terms if len(compact(term)) >= 2]
        if any(len(line) > chars for line in lines):
            flat = re.sub(r"\s+", " ", str(text)).strip()
            positions: set[int] = set()
            compact_terms = [compact(term) for term in terms if compact(term)]
            for term in compact_terms:
                cursor = 0
                for _ in range(24):
                    position = flat.find(term, cursor)
                    if position < 0:
                        break
                    positions.add(position)
                    cursor = position + max(1, len(term))
            if positions:
                candidates: List[tuple[float, int, str]] = []
                for position in positions:
                    start = max(0, min(len(flat) - chars, position - chars // 3))
                    excerpt = flat[start:start + chars]
                    matched = {term for term in compact_terms if term in excerpt}
                    score = sum(
                        (2.0 + min(10.0, len(term) / 2))
                        / (1.0 + 0.35 * max(0, flat.count(term) - 1))
                        for term in matched
                    )
                    candidates.append((score, start, excerpt))
                # 优先选择 query 词最密集的窗口，
                # 而不是可能属于其他条款的第一个数字位置。
                return max(candidates, key=lambda item: (item[0], item[1]))[2]
        scored: List[tuple[float, int]] = []
        for index, line in enumerate(lines):
            compact_line = compact(line)
            score = 0.0
            for term in terms:
                cterm = compact(term)
                if cterm and cterm in compact_line:
                    score += 2.0 + min(8.0, len(cterm) / 2)
            if any(term in compact_line for term in quoted_terms):
                score += 50.0
            if "|" in line and re.search(r"\d", line):
                score += 1.5
            if score:
                scored.append((score, index))
        if not scored:
            return "\n".join(lines)[:chars]
        ranked = sorted(scored, reverse=True)[:5]
        ordered_indices: List[int] = []
        seen_indices: set[int] = set()
        for rank_position, (_, index) in enumerate(ranked):
            radius = 3 if "|" in lines[index] else 1
            for neighbor in range(max(0, index - radius), min(len(lines), index + radius + 1)):
                if neighbor not in seen_indices:
                    seen_indices.add(neighbor)
                    ordered_indices.append(neighbor)
            if "|" in lines[index] and rank_position == 0:
                table_start = index
                while table_start > 0 and "|" in lines[table_start - 1] and index - table_start < 8:
                    table_start -= 1
                table_indices = list(range(table_start, min(index + 3, len(lines))))
                ordered_indices = table_indices + [
                    item for item in ordered_indices if item not in set(table_indices)
                ]
                seen_indices.update(table_indices)
        # 将 query 匹配最密集的条款放在最前。
        # 长篇单页监管文本常在无关条款中出现相同金额；
        # 若按文档顺序拼接，模型容易锚定更早出现的条款。
        excerpt = "\n".join(lines[index] for index in ordered_indices)
        return excerpt[:chars]
