"""顶层题目生命周期与路由分发。"""

from __future__ import annotations

import hashlib
import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from ..core.agent_types import AnswerResult
from .answer_reconciliation import (
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
from .reasoning_policy import assign_reasoning_level



class AgentOrchestrationMixin:
    def _record_reasoning_allocation(
        self,
        question: Dict[str, object],
        route: str,
        discovery: Dict[str, object],
        **decision: object,
    ) -> None:
        allocation = assign_reasoning_level(
            question,
            route=route,
            document_count=len(discovery.get("inferred_doc_ids") or []),
            profile=self.profile,
        )
        qid = str(question.get("qid", ""))
        self._event(
            "route_decision_made",
            qid,
            selected_route=route,
            decision=decision,
        )
        self._event(
            "reasoning_level_assigned",
            qid,
            allocation=allocation.to_dict(),
        )

    def _start_question_lifecycle(
        self,
        question: Dict[str, object],
    ) -> Tuple[Usage, str, str, str]:
        qid = str(question.get("qid", ""))
        self.llm.set_question_context(qid)
        domain = str(question.get("domain", ""))
        answer_format = str(question.get("answer_format", "mcq"))
        self._event(
            "question_started",
            qid,
            domain=domain,
            answer_format=answer_format,
            profile=self.profile_name,
        )
        return Usage(), qid, domain, answer_format

    def _discover_question_documents(
        self,
        question: Dict[str, object],
        qid: str,
        domain: str,
    ) -> Dict[str, object]:
        discovery = self.index.locate_docs(
            question,
            max_docs=int(self.profile.get("max_docs", 6)),
        )
        self._event("document_discovery_completed", qid, discovery=discovery)
        question_entities = extract_entities(self.index.question_text(question), domain)
        self.memory.remember(
            qid,
            domain,
            question_entities,
            "local_question_extractor",
        )
        self._event("question_entities_extracted", qid, entities=question_entities)
        return discovery

    def _build_question_query_plan(
        self,
        question: Dict[str, object],
        qid: str,
        domain: str,
        usage: Usage,
    ) -> QueryPlan:
        planner_domains = {
            str(item)
            for item in self.profile.get("llm_query_planner_domains", [])
        }
        if (
            self.profile.get("llm_query_planner", True)
            and (not planner_domains or domain in planner_domains)
        ):
            self._event("prompt_routed", qid, route="qwen_query_planner")
            plan = self.planner.build(question)
        else:
            self._event("prompt_routed", qid, route="local_query_planner")
            plan = self.planner.build_local(question)
        usage.add(plan.usage)
        self._event("query_plan_completed", qid, plan=plan.to_dict())
        self.memory.remember(
            qid,
            domain,
            plan.useful_entities,
            "qwen_query_planner",
            plan.usage.to_dict(),
        )
        return plan

    def _retrieve_question_evidence(
        self,
        question: Dict[str, object],
        qid: str,
        domain: str,
        discovery: Dict[str, object],
        plan: QueryPlan,
    ) -> Tuple[List[str], List[SearchHit], Dict[str, object], List[Dict[str, object]]]:
        doc_ids = discovery.get("inferred_doc_ids") or list(question.get("doc_ids") or [])
        hits = self._retrieve(domain, doc_ids, plan)
        self._event(
            "retrieval_completed",
            qid,
            doc_ids=list(doc_ids),
            query_count=len(plan.all_queries()),
            hit_count=len(hits),
            hits=[hit.to_dict(chars=0) for hit in hits],
        )
        quality = self._quality(question, plan, hits, doc_ids)
        self._event("retrieval_quality_checked", qid, quality=quality)
        repair_steps: List[Dict[str, object]] = []
        if quality["needs_repair"]:
            repaired_queries = self._repair_queries(question, plan, quality)
            repair_hits = self.index.search(
                domain,
                doc_ids,
                repaired_queries,
                top_k=int(self.profile.get("retrieval_top_k", 24)),
            )
            hits = self._merge_hits(hits, repair_hits)
            repair_steps.append(
                {"reason": quality["reasons"], "queries": repaired_queries}
            )
            quality = self._quality(question, plan, hits, doc_ids)
            self._event(
                "retrieval_repaired",
                qid,
                queries=repaired_queries,
                quality=quality,
                hit_count=len(hits),
            )
        return list(doc_ids), hits, quality, repair_steps

    def answer_question(self, question: Dict[str, object]) -> AnswerResult:
        usage, qid, domain, answer_format = self._start_question_lifecycle(question)
        remembered = self.answer_memory.get(qid)
        if remembered:
            return self._finalize_from_prior_api_memory(question, remembered)

        discovery = self._discover_question_documents(question, qid, domain)
        plan = self._build_question_query_plan(question, qid, domain, usage)
        doc_ids, hits, quality, repair_steps = self._retrieve_question_evidence(
            question,
            qid,
            domain,
            discovery,
            plan,
        )

        if self.profile.get("compact_joint_answer", False):
            compact_calc_results: List[Dict[str, object]] = []
            python_domains = {
                str(item)
                for item in self.profile.get("python_tool_domains", [])
            }
            if (
                answer_format == "calc"
                and self.profile.get("python_tool", True)
                and (not python_domains or domain in python_domains)
                and domain
                not in {
                    str(item)
                    for item in self.profile.get(
                        "skip_compact_calc_domains",
                        [],
                    )
                }
            ):
                calc_result, calc_usage = self._run_calculation_tool(question, hits)
                usage.add(calc_usage)
                if calc_result:
                    compact_calc_results.append(calc_result)
                self._event(
                    "calculation_completed",
                    qid,
                    result=calc_result,
                    usage=calc_usage.to_dict(),
                )
                execution = (
                    calc_result.get("execution")
                    if isinstance(calc_result, dict)
                    else None
                )
                execution_answer = (
                    str(execution.get("answer", "")).strip()
                    if isinstance(execution, dict) and execution.get("ok")
                    else ""
                )
                if (
                    self.profile.get("compact_calc_direct", False)
                    and execution_answer
                ):
                    self._record_reasoning_allocation(
                        question,
                        "compact_calculation_program",
                        discovery,
                        answer_format=answer_format,
                        calculation_execution_ok=True,
                        direct_calculation_enabled=True,
                    )
                    answer = normalize_calculation_answer(execution_answer, str(question.get("question", "")))
                    reasoning_prompt = f"""你是金融计算结果解释代理。下面的Python程序由API根据本轮
原始文档证据生成，并已在受限环境安全执行。请用不超过140个汉字说明关键原始数值、公式、
口径和最终结果。不得改变执行结果，不得添加程序之外的具体事实。
只输出严格JSON：{{"reasoning":"..."}}

题目：{question.get('question', '')}
API生成程序：
{calc_result.get('llm_code', '')}
安全执行结果：{execution_answer}
"""
                    reasoning_usage = Usage()
                    if self.profile.get("compact_calc_reasoning_call", True):
                        self._event(
                            "prompt_routed",
                            qid,
                            route="qwen_compact_calculation_reasoning",
                            prompt_chars=len(reasoning_prompt),
                            prompt_sha256=hashlib.sha256(
                                reasoning_prompt.encode("utf-8")
                            ).hexdigest(),
                            max_tokens=int(
                                self.profile.get(
                                    "compact_calc_reasoning_max_tokens",
                                    260,
                                )
                            ),
                        )
                        reasoning_response = self.llm.chat(
                            [{"role": "user", "content": reasoning_prompt}],
                            max_tokens=int(
                                self.profile.get(
                                    "compact_calc_reasoning_max_tokens",
                                    260,
                                )
                            ),
                        )
                        usage.add(reasoning_response.usage)
                        reasoning_usage = reasoning_response.usage
                        reasoning_parsed = self._parse_answer(
                            reasoning_response.content,
                            "extract",
                        )
                        api_reasoning = (
                            str(reasoning_parsed.get("reasoning", "")).strip()
                            or reasoning_response.content.strip()
                        )
                    else:
                        api_reasoning = (
                            str(calc_result.get("api_response", "")).strip()
                            or str(calc_result.get("llm_code", "")).strip()
                        )
                    calculation_trace = (
                        str(calc_result.get("llm_code", ""))
                        + "\n"
                        + api_reasoning
                    )
                    percent_rate_tail = percent_rate_tail_from_reasoning(
                        calculation_trace,
                        str(question.get("question", "")),
                    )
                    if percent_rate_tail and re.search(r"[；;]", answer):
                        parts = re.split(r"[；;]", answer)
                        parts[-1] = percent_rate_tail
                        answer = normalize_calculation_answer(
                            "；".join(parts),
                            str(question.get("question", "")),
                        )
                    self._event(
                        "compact_calculation_answer_completed",
                        qid,
                        selected_answer=answer,
                        api_reasoning=api_reasoning,
                        execution=execution,
                        usage=usage.to_dict(),
                    )
                    return AnswerResult(
                        qid=qid,
                        answer=answer,
                        reasoning=api_reasoning,
                        usage=usage,
                        metadata={
                            "route": "compact_calculation_program",
                            "doc_discovery": discovery,
                            "query_plan": plan.to_dict(),
                            "query_quality": quality,
                            "query_repair_steps": repair_steps,
                            "calculation_results": compact_calc_results,
                            "calculation_reasoning_usage": (
                                reasoning_usage.to_dict()
                            ),
                        },
                    )
            joint_domains = {
                str(item)
                for item in self.profile.get("compact_joint_domains", [])
            }
            prefer_option_adjudication = domain not in joint_domains
            financial_temporal_risk = False
            financial_special_metric_risk = False
            if domain == "financial_contracts" and domain in joint_domains:
                contract_candidate_docs = [
                    str(item)
                    for item in discovery.get("inferred_doc_ids", [])
                ]
                contract_named_docs = dedupe(
                    [
                        doc_id
                        for text in [
                            str(question.get("question", "")),
                            *[
                                str(value)
                                for value in (question.get("options") or {}).values()
                            ],
                        ]
                        for doc_id in self._docs_named_in_option(
                            domain,
                            contract_candidate_docs,
                            text,
                        )
                    ],
                    len(contract_candidate_docs),
                )
                prefer_option_adjudication = (
                    len(
                        self._docs_named_in_option(
                            domain,
                            [
                                str(item)
                                for item in discovery.get("inferred_doc_ids", [])
                            ],
                            str(question.get("question", "")),
                        )
                    )
                    != 1
                    or self._asks_for_incorrect_stem(
                        str(question.get("question", ""))
                    )
                    or bool(
                        re.search(
                            r"(?:三|多|若干)份|各(?:文件|募集说明书|报告)|横向比较",
                            str(question.get("question", "")),
                        )
                    )
                )
                if any(
                    len(compact(subject)) > 12
                    for subject in re.findall(
                        r"[“\"]([^”\"]{2,80})[”\"]",
                        str(question.get("question", "")),
                    )
                ):
                    prefer_option_adjudication = True
                if (
                    self.profile.get("contract_quantitative_joint", False)
                    and is_dense_quantitative_cross_document_comparison(
                        str(question.get("question", "")),
                        [
                            str(value)
                            for value in (question.get("options") or {}).values()
                        ],
                        len(contract_named_docs),
                    )
                ):
                    prefer_option_adjudication = False
            if domain == "insurance" and domain in joint_domains:
                candidate_docs = [
                    str(item)
                    for item in discovery.get("inferred_doc_ids", [])
                ]
                named_docs = dedupe(
                    [
                        doc_id
                        for text in [
                            str(question.get("question", "")),
                            *[
                                str(value)
                                for value in (question.get("options") or {}).values()
                            ],
                        ]
                        for doc_id in self._docs_named_in_option(
                            domain,
                            candidate_docs,
                            text,
                        )
                    ],
                    len(candidate_docs),
                )
                option_blob = "\n".join(
                    str(value) for value in (question.get("options") or {}).values()
                )
                quantitative_scenarios = len(
                    re.findall(
                        r"\d+(?:\.\d+)?(?:万元|元|周岁|岁|个保单年度|年)",
                        option_blob,
                    )
                )
                # 数值假设场景最适合让每条政策公式独立绑定各自证据。
                # 条款列表比较和普通单文档条款题则更需要一个等价术语的联合视图。
                prefer_option_adjudication = (
                    quantitative_scenarios >= 4
                    or bool(re.search(r"保单贷款|借款", str(question.get("question", ""))))
                    or bool(
                        re.search(
                            r"保障触发条件|触发条件",
                            str(question.get("question", "")),
                        )
                    )
                    or bool(
                        re.search(
                            r"等.{0,8}(?:风险|免责)|明确(?:列明|约定)",
                            str(question.get("question", "")),
                        )
                    )
                )
            if domain == "regulatory" and domain in joint_domains:
                stem = str(question.get("question", ""))
                domain_concepts = [
                    term
                    for term in DOMAIN_TERMS.get(domain, [])
                    if term in self.index.question_text(question)
                ]
                prefer_option_adjudication = not (
                    "同时" in stem
                    or (
                        re.search(r"20\d{2}年\d{1,2}月\d{1,2}日", stem)
                        and len(domain_concepts) >= 3
                    )
                )
            if domain == "financial_reports":
                prefer_option_adjudication = "现金分红" in str(
                    question.get("question", "")
                )
                financial_option_blob = "\n".join(
                    str(value)
                    for value in (question.get("options") or {}).values()
                )
                year_span = re.search(
                    r"(20\d{2})\s*[-—–至到]\s*(20\d{2})",
                    self.index.question_text(question),
                )
                financial_temporal_risk = bool(
                    (
                        len(
                            set(
                                re.findall(
                                    r"20\d{2}",
                                    self.index.question_text(question),
                                )
                            )
                        )
                        >= 3
                        or (
                            year_span
                            and int(year_span.group(2)) - int(year_span.group(1)) >= 2
                        )
                    )
                    and re.search(
                        r"连续|先.+后|下降|上升|回升|回落",
                        financial_option_blob,
                    )
                )
                financial_special_metric_risk = (
                    "EBITDA" in financial_option_blob.upper()
                )
                if (
                    financial_special_metric_risk
                    or financial_temporal_risk
                ):
                    prefer_option_adjudication = True
            if domain == "research" and domain in joint_domains:
                research_text = self.index.question_text(question)
                if re.search(
                    r"分位|品牌化|渠道品牌|产品品牌",
                    research_text,
                    flags=re.I,
                ):
                    prefer_option_adjudication = True
            force_joint_domains = {
                str(item)
                for item in self.profile.get("force_joint_domains", [])
            }
            financial_dividend_risk = bool(
                domain == "financial_reports"
                and re.search(
                    r"现金分红|派息|派现",
                    str(question.get("question", "")),
                )
            )
            financial_structural_risk = bool(
                domain == "financial_reports"
                and (
                    re.search(
                        r"分地区|境外收入|境内收入|"
                        r"合并.{0,20}母公司|母公司.{0,20}合并",
                        self.index.question_text(question),
                    )
                    or sum(
                        term in self.index.question_text(question)
                        for term in (
                            "现金分红",
                            "经营活动现金流",
                            "每股收益",
                            "净利润",
                            "营业收入",
                            "净资产收益率",
                        )
                    )
                    >= 3
                )
            )
            financial_multi_metric_comparison = bool(
                domain == "financial_reports"
                and re.search(
                    r"(?:比较数据|指标比较|以下哪些判断)",
                    str(question.get("question", "")),
                )
                and sum(
                    term in self.index.question_text(question)
                    for term in (
                        "EBITDA",
                        "每股收益",
                        "经营活动产生的现金流量净额",
                        "净资产收益率",
                        "营业收入",
                        "净利润",
                    )
                )
                >= 3
            )
            if (
                domain in force_joint_domains
                and not financial_dividend_risk
                and not financial_structural_risk
                and not financial_multi_metric_comparison
                and not financial_temporal_risk
                and not financial_special_metric_risk
            ):
                prefer_option_adjudication = False
            if financial_multi_metric_comparison:
                prefer_option_adjudication = True
            unconditional_joint_domains = {
                str(item)
                for item in self.profile.get(
                    "force_joint_unconditional_domains",
                    [],
                )
            }
            if domain in unconditional_joint_domains:
                prefer_option_adjudication = False
            if (
                self.profile.get("compact_option_adjudication", False)
                and answer_format in {"mcq", "multi", "tf"}
                and isinstance(question.get("options"), dict)
                and question.get("options")
                and prefer_option_adjudication
            ):
                self._record_reasoning_allocation(
                    question,
                    "compact_option_adjudication",
                    discovery,
                    prefer_option_adjudication=True,
                    financial_temporal_risk=financial_temporal_risk,
                    financial_special_metric_risk=financial_special_metric_risk,
                    financial_dividend_risk=financial_dividend_risk,
                    financial_structural_risk=financial_structural_risk,
                    financial_multi_metric_comparison=(
                        financial_multi_metric_comparison
                    ),
                )
                return self._compact_option_answer(
                    question,
                    discovery,
                    plan,
                    hits,
                    quality,
                    repair_steps,
                    usage,
                )
            self._record_reasoning_allocation(
                question,
                "compact_joint_answer",
                discovery,
                prefer_option_adjudication=prefer_option_adjudication,
                forced_joint=domain in force_joint_domains,
                unconditional_joint=domain in unconditional_joint_domains,
                financial_temporal_risk=financial_temporal_risk,
                financial_special_metric_risk=financial_special_metric_risk,
                financial_dividend_risk=financial_dividend_risk,
                financial_structural_risk=financial_structural_risk,
                financial_multi_metric_comparison=(
                    financial_multi_metric_comparison
                ),
            )
            return self._compact_joint_answer(
                question,
                discovery,
                plan,
                hits,
                quality,
                repair_steps,
                usage,
                compact_calc_results,
            )

        calc_results: List[Dict[str, object]] = []
        if self.profile.get("python_tool", True) and self._needs_calculation(question):
            calc_result, calc_usage = self._run_calculation_tool(question, hits)
            usage.add(calc_usage)
            if calc_result:
                calc_results.append(calc_result)
            self._event(
                "calculation_completed",
                qid,
                result=calc_result,
                usage=calc_usage.to_dict(),
            )

        prompt = self._answer_prompt(question, discovery, plan, hits, calc_results, compact=False)
        self._record_reasoning_allocation(
            question,
            "primary_answer",
            discovery,
            compact_mode=False,
            calculation_result_count=len(calc_results),
        )
        self._event(
            "prompt_routed",
            qid,
            route="qwen_primary_answer",
            prompt_chars=len(prompt),
            prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            evidence_count=min(len(hits), int(self.profile.get("prompt_pages", 18))),
            max_tokens=int(self.profile.get("answer_max_tokens", 6144)),
        )
        response = self.llm.chat([{"role": "user", "content": prompt}], max_tokens=int(self.profile.get("answer_max_tokens", 6144)))
        usage.add(response.usage)
        parsed = self._parse_answer(response.content, answer_format)
        judgments = parsed.get("option_judgments")
        has_unknown = (
            isinstance(judgments, dict)
            and any(
                isinstance(value, dict)
                and str(value.get("verdict", "")).lower() == "unknown"
                for value in judgments.values()
            )
        )
        if (
            domain
            in {
                str(item)
                for item in self.profile.get(
                    "compact_joint_unknown_fallback_domains",
                    [],
                )
            }
            and has_unknown
        ):
            self._event(
                "compact_joint_unknown_fallback",
                qid,
                route="independent_option_adjudication",
                joint_usage=response.usage.to_dict(),
            )
            return self._compact_option_answer(
                question,
                discovery,
                plan,
                context_hits,
                quality,
                repair_steps,
                usage,
                calc_results,
            )
        answer = normalize_answer(str(parsed.get("answer", "")), answer_format)
        judgment_answer = self._answer_from_judgments(parsed, answer_format, question)
        if judgment_answer:
            answer = judgment_answer
        reasoning_answer = self._answer_from_reasoning(str(parsed.get("reasoning", "")), answer_format)
        if reasoning_answer:
            answer = reasoning_answer
        if answer_format == "calc":
            answer = normalize_calculation_answer(answer, str(question.get("question", "")))
        reasoning = str(parsed.get("reasoning", "") or response.content[:1000])
        self._event(
            "primary_answer_completed",
            qid,
            api_answer=parsed.get("answer", ""),
            normalized_answer=answer,
            api_reasoning=reasoning,
            confidence=parsed.get("confidence", ""),
            option_judgments=parsed.get("option_judgments", {}),
            usage=response.usage.to_dict(),
            finish_reason=response.finish_reason,
        )

        reactions = []
        needs_verification = bool(
            self.profile.get("verify", True)
            and not (
                self.profile.get("option_adjudication", False)
                and answer_format in {"mcq", "multi", "tf"}
            )
            and self._needs_reaction(question, answer, parsed, quality)
        )
        self._event(
            "verification_routed",
            qid,
            enabled=bool(self.profile.get("verify", True)),
            selected=needs_verification,
        )
        if needs_verification:
            repair_prompt = self._answer_prompt(question, discovery, plan, hits, calc_results, compact=False)
            repair_prompt += (
                "\n\n【二次核查任务】\n"
                f"上一轮答案为：{answer}\n"
                "请逐项核对证据是否足以支持答案。若证据不足，不要猜测；若需要计算，请使用已给出的计算结果。"
                "先在内部完成核对，只输出一次最终 JSON，不要在输出中反复修改结论。"
                "reasoning 不超过180个汉字，answer 与 option_judgments 必须一致。"
            )
            self._event(
                "prompt_routed",
                qid,
                route="qwen_answer_verification",
                prompt_chars=len(repair_prompt),
                prompt_sha256=hashlib.sha256(repair_prompt.encode("utf-8")).hexdigest(),
                max_tokens=int(self.profile.get("repair_max_tokens", 4096)),
            )
            repaired = self.llm.chat(
                [{"role": "user", "content": repair_prompt}],
                max_tokens=int(self.profile.get("repair_max_tokens", 4096)),
            )
            usage.add(repaired.usage)
            second = self._parse_answer(repaired.content, answer_format)
            second_answer = normalize_answer(str(second.get("answer", "")), answer_format)
            second_judgment_answer = self._answer_from_judgments(second, answer_format, question)
            if second_judgment_answer:
                second_answer = second_judgment_answer
            second_reasoning_answer = self._answer_from_reasoning(
                str(second.get("reasoning", "")),
                answer_format,
            )
            if second_reasoning_answer:
                second_answer = second_reasoning_answer
            if answer_format == "calc":
                second_answer = normalize_calculation_answer(second_answer, str(question.get("question", "")))
            if second_answer:
                reactions.append({"type": "verification", "before": answer, "after": second_answer})
                answer = second_answer
                reasoning = str(second.get("reasoning", reasoning))
            self._event(
                "verification_completed",
                qid,
                api_answer=second.get("answer", ""),
                normalized_answer=second_answer,
                api_reasoning=str(second.get("reasoning", "")),
                usage=repaired.usage.to_dict(),
                finish_reason=repaired.finish_reason,
            )

        option_adjudication: List[Dict[str, object]] = []
        if (
            self.profile.get("option_adjudication", False)
            and answer_format in {"mcq", "multi", "tf"}
            and isinstance(question.get("options"), dict)
            and question.get("options")
        ):
            adjudicated_answer, adjudicated_reasoning, adjudication_usage, option_adjudication = (
                self._adjudicate_options(question, discovery, hits, parsed)
            )
            usage.add(adjudication_usage)
            if adjudicated_answer:
                reactions.append(
                    {
                        "type": "option_adjudication",
                        "before": answer,
                        "after": adjudicated_answer,
                    }
                )
                answer = adjudicated_answer
                reasoning = adjudicated_reasoning

        if answer_format == "calc":
            arithmetic_answer = self._answer_from_explicit_arithmetic(
                reasoning,
                str(question.get("question", "")),
            )
            if arithmetic_answer and arithmetic_answer != answer:
                reactions.append(
                    {
                        "type": "explicit_arithmetic_recheck",
                        "before": answer,
                        "after": arithmetic_answer,
                    }
                )
                answer = arithmetic_answer

        return AnswerResult(
            qid=qid,
            answer=answer,
            reasoning=reasoning,
            usage=usage,
            metadata={
                "doc_discovery": discovery,
                "query_plan": plan.to_dict(),
                "query_quality": quality,
                "query_repair_steps": repair_steps,
                "evidence": [h.to_dict(chars=700) for h in hits[: int(self.profile.get("prompt_pages", 20))]],
                "calculation_results": calc_results,
                "reactions": reactions,
                "option_adjudication": option_adjudication,
                "answer_analysis": {
                    "confidence": parsed.get("confidence", ""),
                    "option_judgments": parsed.get("option_judgments", {}),
                },
            },
        )
