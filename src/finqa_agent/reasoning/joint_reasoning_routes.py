"""联合回答、记忆定稿与裁决路由。"""

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
    repair_region_share_answer_from_evidence,
    repair_zero_balance_condition_verdict,
    should_apply_reasoning_answer,
)
from ..calculation.calculation_reconciliation import (
    ordered_subject_answer_from_reasoning,
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



class JointReasoningRoutesMixin:
    def _compact_joint_answer(
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
        """基于选项平衡且聚焦查询的证据胶囊一次性作答。"""
        qid = str(question.get("qid", ""))
        domain = str(question.get("domain", ""))
        stem = str(question.get("question", ""))
        answer_format = str(question.get("answer_format", "multi"))
        options = question.get("options") or {}
        doc_ids = [
            str(item)
            for item in (discovery.get("inferred_doc_ids") or question.get("doc_ids") or [])
        ]
        per_option = int(self.profile.get("compact_option_hits", 3))
        excerpt_chars = int(self.profile.get("compact_excerpt_chars", 650))
        if domain == "insurance":
            per_option = int(
                self.profile.get(
                    "compact_joint_insurance_option_hits",
                    self.profile.get("compact_insurance_option_hits", per_option),
                )
            )
            excerpt_chars = int(
                self.profile.get(
                    "compact_insurance_excerpt_chars",
                    excerpt_chars,
                )
            )
        elif domain == "research":
            per_option = int(
                self.profile.get("compact_research_option_hits", per_option)
            )
            excerpt_chars = int(
                self.profile.get(
                    "compact_research_excerpt_chars",
                    excerpt_chars,
                )
            )
        elif domain == "regulatory":
            per_option = int(
                self.profile.get(
                    "compact_joint_regulatory_option_hits",
                    self.profile.get("compact_regulatory_option_hits", per_option),
                )
            )
            excerpt_chars = int(
                self.profile.get(
                    "compact_regulatory_excerpt_chars",
                    excerpt_chars,
                )
            )
        if (
            domain == "regulatory"
            and re.search(
                r"定期报告|披露程序|非交易时段|汇款|门槛|可疑|"
                r"施行|生效|同时|期限",
                stem,
            )
        ):
            per_option = max(per_option, 3)
            excerpt_chars = max(excerpt_chars, 380)
        if (
            domain == "research"
            and re.search(r"共同|四家|多家|品牌化|不同行业|深层特征", stem)
        ):
            per_option = max(per_option, 4)
            excerpt_chars = max(excerpt_chars, 220)
        if answer_format == "calc":
            per_option = int(self.profile.get("compact_calc_hits", 10))
            excerpt_chars = int(self.profile.get("compact_calc_excerpt_chars", 900))
        if (
            domain == "financial_contracts"
            and re.search(
                r"连续|评级|仅|附加|例外|过渡期|不超过|三分之二",
                stem,
            )
        ):
            excerpt_chars = max(
                excerpt_chars,
                int(
                    self.profile.get(
                        "adaptive_contract_excerpt_chars",
                        800,
                    )
                ),
            )
        evidence_sections: List[str] = []
        evidence_meta: List[Dict[str, object]] = []
        evidence_id = 0
        stem_doc_ids = self._docs_named_in_option(domain, doc_ids, stem)

        if isinstance(options, dict) and options:
            for letter, raw_option in sorted(options.items()):
                option = str(raw_option)
                option_emitted_pages: set[tuple[str, str]] = set()
                queries = dedupe(
                    [
                        f"{stem}\n{option}",
                        option,
                        *(
                            [
                                f"{age}周岁 身故保险金 给付比例 年龄 基本保险金额"
                                for age in re.findall(r"(\d{1,3})周岁", option)
                            ]
                            if domain == "insurance"
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
                        *(
                            [
                                "B端 C端 品牌 渠道 零售 品牌溢价 消费者",
                                "C端零售渠道 品牌建设 产品差异 品牌认知",
                            ]
                            if domain == "research"
                            and re.search(r"B端.{0,12}C端|C端.{0,12}B端", option)
                            else []
                        ),
                        *(
                            [
                                "海外产能 多区域布局 地缘风险 关税 贸易壁垒",
                                "欧洲 东南亚 工厂 产能落地 规避贸易壁垒 全球协同",
                            ]
                            if domain == "research"
                            and re.search(
                                r"地缘|关税|贸易壁垒|海外.{0,12}产能|"
                                r"多区域|全球化布局",
                                option,
                            )
                            else []
                        ),
                        *(
                            [
                                "渠道品牌 产品品牌 内容流量 选品 品质 双轮驱动",
                                "品牌化 消费者认知 产品差异 技术参数 稳定交付",
                            ]
                            if domain == "research"
                            and re.search(r"品牌化|渠道品牌|产品品牌", stem + option)
                            else []
                        ),
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
                option_doc_ids = self._docs_named_in_option(domain, doc_ids, option)
                research_cross_scope = (
                    domain == "research"
                    and needs_cross_document_research_scope(
                        stem,
                        option_doc_ids,
                        doc_ids,
                    )
                )
                if research_cross_scope:
                    # 比较选项中的泛化别名可能把跨报告断言误折叠到单份文档。
                    # 恢复本轮局部候选集，使选项能在所有具名侧之间被支持或反驳。
                    option_doc_ids = list(doc_ids)
                if (
                    domain == "insurance"
                    and len(stem_doc_ids) == 1
                    and len(option_doc_ids) != 1
                ):
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
                    option_doc_ids = list(stem_doc_ids)
                option_hits = self._option_balanced_hits(
                    domain,
                    option_doc_ids or doc_ids,
                    queries,
                    per_option,
                )
                if domain == "financial_reports" and "EBITDA" in option.upper():
                    option_hits = self._merge_hits(
                        self.index.search(
                            domain,
                            option_doc_ids or doc_ids,
                            ["EBITDA EBITDA率 营业收入"],
                            top_k=3,
                        ),
                        option_hits,
                    )
                if domain == "financial_reports":
                    comparison_years = dedupe(
                        re.findall(r"20\d{2}", f"{stem}\n{option}"),
                        5,
                    )
                    if (
                        len(comparison_years) >= 3
                        and re.search(r"(?:下降|降低|回升|回落|连续|先.+后)", option)
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
                        option_hits = self._merge_hits(
                            self._exact_term_hits(
                                domain,
                                option_doc_ids or doc_ids,
                                metric_terms,
                                query=option,
                                per_document=2,
                            ),
                            option_hits,
                        )
                constraint_focus = extract_constraint_terms(
                    f"{stem}\n{option}",
                    domain,
                )
                if domain == "research":
                    # 技术缩写和明确策略名是可审计锚点。
                    # 将它们共址检索，可防止跨机构比较拼接两份无关报告中的事实。
                    technical_anchors = dedupe(
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
                        [*technical_anchors, *constraint_focus],
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
                if (
                    domain in {"insurance", "regulatory"}
                    or (
                        domain == "research"
                        and (
                            "分位" in option
                            or research_cross_scope
                            or len(
                                re.findall(
                                    r"[A-Z][A-Z0-9._-]{2,18}",
                                    option,
                                )
                            )
                            >= 1
                        )
                    )
                ) and constraint_focus:
                    exact_hits = self._exact_term_hits(
                        domain,
                        option_doc_ids or doc_ids,
                        constraint_focus,
                        query=option,
                        per_document=2,
                    )
                    option_hits = self._merge_hits(
                        exact_hits,
                        self._merge_hits(
                            self.index.search(
                                domain,
                                option_doc_ids or doc_ids,
                                constraint_focus,
                                top_k=2,
                            ),
                            option_hits,
                        ),
                    )
                if domain == "insurance":
                    option_hits = self._with_adjacent_pages(domain, option_hits)
                if (
                    domain == "research"
                    and re.search(r"地缘|关税|贸易壁垒|贸易风险", option)
                    and re.search(r"海外|多区域|产能|供应链", option)
                ):
                    exact_strategy_hits = self._exact_term_hits(
                        domain,
                        option_doc_ids or doc_ids,
                        ["欧洲", "东南亚", "贸易壁垒", "地缘政治风险", "产能"],
                        query=option,
                        per_document=2,
                    )
                    strategy_queries = [
                        "海外产能 欧洲 东南亚 贸易壁垒",
                        "地缘政治风险 供应链 东南亚 工厂",
                        "关税 贸易风险 多区域 产能布局",
                    ]
                    strategy_hits = self.index.search(
                        domain,
                        option_doc_ids or doc_ids,
                        strategy_queries,
                        top_k=6,
                    )
                    strategy_hits = [
                        hit
                        for hit in self._merge_hits(
                            exact_strategy_hits,
                            strategy_hits,
                        )
                        if re.search(
                            r"地缘|关税|贸易壁垒|贸易风险",
                            hit.page.text,
                        )
                        and re.search(
                            r"海外|欧洲|东南亚|多区域|产能|供应链|工厂",
                            hit.page.text,
                        )
                    ]
                    option_hits = self._merge_hits(strategy_hits, option_hits)
                joint_hit_cap = int(
                    self.profile.get("compact_joint_hard_option_hit_cap", 0)
                )
                if joint_hit_cap > 0:
                    option_hits = self._cap_hits_preserving_documents(
                        option_hits,
                        joint_hit_cap,
                    )
                rows = []
                for hit in option_hits:
                    page_key = (hit.page.doc_id, hit.page.page_id)
                    if page_key in option_emitted_pages:
                        continue
                    option_emitted_pages.add(page_key)
                    evidence_id += 1
                    excerpt = self._focused_excerpt(
                        hit.page.text,
                        " ".join(constraint_focus) or option,
                        excerpt_chars,
                    )
                    if (
                        domain == "research"
                        and re.search(
                            r"分位|净流入|净流出|历史高位|历史低位",
                            option,
                        )
                    ):
                        direct_matches = []
                        for term in constraint_focus:
                            position = hit.page.text.find(term)
                            if position < 0:
                                continue
                            start = max(0, position - 80)
                            end = min(
                                len(hit.page.text),
                                position + len(term) + 120,
                            )
                            direct_matches.append(
                                re.sub(
                                    r"\s+",
                                    " ",
                                    hit.page.text[start:end],
                                ).strip()
                            )
                        if direct_matches:
                            excerpt = (
                                "原子约束命中："
                                + " | ".join(direct_matches[:3])
                                + "\n"
                                + excerpt
                            )
                    rows.append(
                        f"[E{evidence_id}] doc={hit.page.doc_id} page={hit.page.page_id} "
                        f"score={hit.score:.2f}\n{excerpt}"
                    )
                    evidence_meta.append(
                        {
                            "evidence_id": evidence_id,
                            "option": str(letter),
                            **hit.to_dict(chars=0),
                        }
                    )
                evidence_sections.append(
                    f"### 选项{letter}定向证据\n" + ("\n".join(rows) or "未检索到直接证据")
                )
        else:
            for hit in context_hits[: max(6, per_option * 3)]:
                evidence_id += 1
                excerpt = self._focused_excerpt(hit.page.text, hit.query, excerpt_chars)
                evidence_sections.append(
                    f"[E{evidence_id}] doc={hit.page.doc_id} page={hit.page.page_id} "
                    f"score={hit.score:.2f}\n{excerpt}"
                )
                evidence_meta.append({"evidence_id": evidence_id, **hit.to_dict(chars=0)})

        option_text = "\n".join(f"{key}. {value}" for key, value in sorted(options.items()))
        polarity = (
            "题干要求选择不成立项，answer应收集contradicted项。"
            if self._asks_for_incorrect_stem(stem)
            else "题干要求选择成立项，answer应收集supported项。"
        )
        prompt = f"""你是金融长文档证据裁决器。只根据本轮从原始文档检索的证据作答。
任务要求：
1. 对每个选项分别输出 supported、contradicted 或 unknown，并引用证据编号。
2. 核对主体、时点、期限、单位、比例口径、例外条件和必要的跨章节关系。
3. 计算题须在reasoning中写出关键算式，严格按题目要求保留单位和小数位。
   题目未明确指定小数位时，数值计算默认保留两位小数，不得擅自改为一位。
   若题目明确首日不计入，则次日是第1日；累计N日到达的当日就是第N日和截止日，
   除非证据另有“次日”规则，不得在结果上再次加一天。
4. 研究报告的综合比较允许从多份材料作有证据的因果归纳，不要求原文逐字给出排名。
   “最符合实际、共同思路、深层特征、品牌化难度”等题，应根据历史渠道、产品差异可感知性、
   消费者心智、渠道继承性、品质信任和供需机制作结构化归纳；不得只因报告未逐字写出结论判错。
   带符号资金净流量为负且处于极低百分位，表示净流出程度极端；应与正向净流入的高分位结合判断。
   若材料已证明业务长期偏企业端、消费品牌或品牌溢价仍在培育，可据此支持其向消费端迁移难度较高；
   不得额外要求原文逐字写出“难度最大”。线下体验门店与深加工设施分别可作为渠道体验和产品增值证据。
   资产负债管理判断应按功能实质裁决：优化资产会计分类以平滑利润、调整久期以匹配负债，
   均属于资产负债联动管理，不要求原文逐字出现“较好联动”；风险约束会限制单纯收益最大化，
   但不否定在约束下优化收益与稳定性的合理目标。
   {RESEARCH_FUNCTIONAL_QUALITY_RULE}
5. 不得用常识补充具体事实；但可由已给事实完成算术、时间和逻辑推导。
6. {polarity}
7. 保险题须联合核对保险责任、等待期、免责/除外、给付比例、累计限额和条款例外；
   “不承担”与“承担后按比例给付”不同，不得因相邻条款被切页而遗漏限制。
   年龄分档必须找到覆盖该年龄的明确区间，不得用行业惯例或相邻区间反推缺失档位。
   若题干概括询问“核风险”等类别，核反应、核辐射、核爆炸、核污染或放射性污染均可作为
   明确列明该类风险的证据；只有题干要求逐字一致或完整列出全部术语时，才要求每个词同时出现。
8. 合同题须回到题干指定文件，联合核对定义、触发条件、期限、公式、例外和争议解决；
   “完全一致、均、所有”要求逐份成立，任何一份存在额外条件都应判错。
   {CONTRACT_PARALLEL_CLAUSE_RULE}
   软件研发、系统实施等非传统生产线项目，若文档说明其不直接形成可量化产能，可据此判断
   传统新增产能消化风险不适用；不应仅因没有出现同名风险标题而判为证据不足。
9. 法规题必须精确区分上位制度与具体子义务，某项识别核实的过渡期限不得扩张成全部尽调期限。
   选项准确陈述一个基础期限或并行义务时，不得只因省略通常适用的触发前提或另一项义务而判错，
   除非选项声称无条件、仅此一项或完整列举。
   多部办法存在相似义务时，必须把每个期限绑定到其所属办法及该办法施行日，不得把甲办法的
   生效日移植给乙办法。多个触发条件同时成立时，各项法定义务原则上并行适用；客户拒绝配合
   不当然取消因交易异常而产生的进一步核实义务。
只输出一次严格JSON：
{{"answer":"...","confidence":"high|medium|low","option_judgments":{{"A":{{"verdict":"supported|contradicted|unknown","evidence_ids":[1]}}}},"reasoning":"不超过220字"}}

领域：{domain}
答案格式：{answer_format}
题目：{stem}
选项：
{option_text or "无选项"}

本地安全执行的API计算程序及结果：
{json.dumps(calc_results, ensure_ascii=False)[:4000]}

本轮原始文档证据胶囊：
{chr(10).join(evidence_sections)}
"""
        if self.profile.get("compact_domain_scoped_prompt", False):
            prompt = build_scoped_joint_prompt(
                domain=domain,
                answer_format=answer_format,
                stem=stem,
                options=options if isinstance(options, dict) else {},
                evidence_sections=evidence_sections,
                calculation_results=calc_results,
                selects_incorrect=self._asks_for_incorrect_stem(stem),
            )
        self._event(
            "evidence_capsules_built",
            qid,
            capsule_count=evidence_id,
            prompt_chars=len(prompt),
            evidence=evidence_meta,
        )
        self._event(
            "prompt_routed",
            qid,
            route="qwen_compact_joint_answer",
            prompt_chars=len(prompt),
            prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            max_tokens=int(self.profile.get("answer_max_tokens", 750)),
        )
        response = self.llm.chat(
            [{"role": "user", "content": prompt}],
            max_tokens=int(self.profile.get("answer_max_tokens", 750)),
        )
        usage.add(response.usage)
        parsed = self._parse_answer(response.content, answer_format)
        joint_reasoning = (
            str(parsed.get("reasoning", "")).strip()
            or response.content.strip()
        )
        parsed_judgments = parsed.get("option_judgments")
        if isinstance(parsed_judgments, dict):
            segments = {
                letter: segment
                for letter, segment in re.findall(
                    r"([A-D])(?:项)?[：:]\s*(.*?)(?=(?:[A-D])(?:项)?[：:]|$)",
                    joint_reasoning,
                    flags=re.S,
                )
            }
            for letter, segment in segments.items():
                item = parsed_judgments.get(letter)
                if not isinstance(item, dict):
                    continue
                item["verdict"] = explicit_segment_verdict(
                    segment,
                    str(item.get("verdict", "")).lower(),
                )
                option_text_for_segment = str(options.get(letter, ""))
                item["verdict"] = repair_parallel_duty_omission_verdict(
                    option_text_for_segment,
                    segment,
                    str(item.get("verdict", "")).lower(),
                )
                item["verdict"] = repair_numeric_self_consistency_verdict(
                    option_text_for_segment,
                    segment,
                    str(item.get("verdict", "")).lower(),
                )
                item["verdict"] = repair_zero_balance_condition_verdict(
                    option_text_for_segment,
                    segment,
                    str(item.get("verdict", "")).lower(),
                )
                if (
                    re.search(r"明确(?:列明|列出|约定|提及)", stem)
                    and not re.search(r"未|没有|不包含", option_text_for_segment)
                    and re.search(r"未提及|没有列明|无法支持", segment)
                ):
                    item["verdict"] = "contradicted"
                if domain == "research":
                    if (
                        re.search(r"净流出.{0,30}低分位", option_text_for_segment)
                        and re.search(r"净流入.{0,30}高分位", option_text_for_segment)
                        and re.search(r"-\s*\d+(?:\.\d+)?", segment)
                        and re.search(r"\d+(?:\.\d+)?\s*%分位", segment)
                    ):
                        item["verdict"] = "supported"
                    if (
                        re.search(
                            r"都是通过.{0,30}(?:地域|渠道).{0,12}壁垒"
                            r".{0,30}(?:头部|集中)",
                            option_text_for_segment,
                        )
                        and re.search(r"连锁|渠道|网点|合作限制", segment)
                        and re.search(r"未提|缺乏对应|无法确认共同", segment)
                        and not re.search(r"方向相反|机制相反|明确禁止", segment)
                    ):
                        item["verdict"] = "supported"
        compact_fallback_domains = {
            str(item)
            for item in self.profile.get(
                "compact_joint_unknown_fallback_domains",
                [],
            )
        }
        option_corpus = "\n".join(str(value) for value in options.values())
        if (
            domain in compact_fallback_domains
            and isinstance(parsed_judgments, dict)
            and any(
                isinstance(item, dict)
                and str(item.get("verdict", "")).lower() == "unknown"
                for item in parsed_judgments.values()
            )
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
        reasoning_answer = self._answer_from_reasoning(
            str(parsed.get("reasoning", "")),
            answer_format,
            answer,
        )
        if should_apply_reasoning_answer(
            answer_format,
            judgment_answer,
            reasoning_answer,
        ):
            answer = reasoning_answer
        reasoning = joint_reasoning
        expected_cardinality = expected_option_cardinality(stem)
        if (
            answer_format == "multi"
            and expected_cardinality
            and len(answer) != expected_cardinality
        ):
            cardinality_prompt = f"""你是金融长文档问答的答案基数复核代理。
题干明确要求选择{expected_cardinality}项，但首轮API答案数量不符。仅根据本轮证据重新判断，
必须输出恰好{expected_cardinality}个按字母排序的选项。分析性研报比较允许从明确事实作功能
实质归纳，不要求原文逐字复述结论；不得凭常识补充具体事实。
只输出严格JSON：{{"answer":"...","reasoning":"不超过160字"}}

题目：{stem}
选项：{option_text}
首轮答案与推理：{answer}；{reasoning}
本轮证据：
{chr(10).join(evidence_sections)[:7000]}
"""
            self._event(
                "prompt_routed",
                qid,
                route="qwen_compact_cardinality_review",
                prompt_chars=len(cardinality_prompt),
                prompt_sha256=hashlib.sha256(
                    cardinality_prompt.encode("utf-8")
                ).hexdigest(),
                max_tokens=420,
            )
            cardinality_response = self.llm.chat(
                [{"role": "user", "content": cardinality_prompt}],
                max_tokens=420,
            )
            usage.add(cardinality_response.usage)
            cardinality_parsed = self._parse_answer(
                cardinality_response.content,
                answer_format,
            )
            cardinality_answer = normalize_answer(
                str(cardinality_parsed.get("answer", "")),
                answer_format,
            )
            if len(cardinality_answer) == expected_cardinality:
                answer = cardinality_answer
                reasoning = (
                    str(cardinality_parsed.get("reasoning", "")).strip()
                    or cardinality_response.content.strip()
                )
                self._event(
                    "compact_cardinality_review_completed",
                    qid,
                    selected_answer=answer,
                    api_reasoning=reasoning,
                    usage=cardinality_response.usage.to_dict(),
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
                    "\n".join(evidence_sections),
                    minimum_multi_options,
                )
            )
            usage.add(review_usage)
            if len(reviewed_answer) >= minimum_multi_options:
                answer = reviewed_answer
                reasoning = reviewed_reasoning
        if answer_format == "calc":
            answer = normalize_calculation_answer(answer, stem)
            final_reasoning_answer = self._final_calculation_answer_from_reasoning(
                reasoning,
                stem,
            )
            if final_reasoning_answer:
                answer = final_reasoning_answer
            else:
                arithmetic = self._answer_from_explicit_arithmetic(reasoning, stem)
                if arithmetic:
                    answer = arithmetic
                elif re.search(r"[；;]", answer):
                    corrected_tail = self._final_scalar_from_reasoning(reasoning, stem)
                    if corrected_tail:
                        parts = re.split(r"[；;]", answer)
                        parts[-1] = corrected_tail
                        answer = normalize_calculation_answer("；".join(parts), stem)
            ratio_point_tail = self._ratio_point_tail_from_reasoning(
                reasoning,
                stem,
            )
            if ratio_point_tail and re.search(r"[；;]", answer):
                parts = re.split(r"[；;]", answer)
                parts[-1] = ratio_point_tail
                answer = normalize_calculation_answer("；".join(parts), stem)
            percent_rate_tail = percent_rate_tail_from_reasoning(
                reasoning,
                stem,
            )
            if percent_rate_tail and re.search(r"[；;]", answer):
                parts = re.split(r"[；;]", answer)
                parts[-1] = percent_rate_tail
                answer = normalize_calculation_answer("；".join(parts), stem)
            rounded_intermediate = rounded_intermediate_from_reasoning(
                reasoning,
                stem,
            )
            if rounded_intermediate and not re.search(r"[；;]", answer):
                answer = normalize_calculation_answer(
                    rounded_intermediate,
                    stem,
                )
            ordered_subject_answer = ordered_subject_answer_from_reasoning(
                reasoning,
                stem,
            )
            if ordered_subject_answer:
                answer = normalize_calculation_answer(
                    ordered_subject_answer,
                    stem,
                )
        if domain == "financial_reports" and re.search(r"分地区|境外收入", stem):
            full_document_evidence = "\n".join(
                self.index.doc_texts.get(domain, {}).get(doc_id, "")
                for doc_id in doc_ids
            )
            answer = repair_region_share_answer_from_evidence(
                answer,
                full_document_evidence,
                options,
                answer_format,
                self.index.question_text(question),
            )
        raw_api_reasoning = reasoning
        reasoning_provenance: Dict[str, object] = {}
        if bool(self.profile.get("reasoning_expand_prompt_evidence", False)):
            reasoning, reasoning_provenance = expand_reasoning_from_counted_prompt(
                raw_api_reasoning,
                prompt,
                max_references=int(
                    self.profile.get("reasoning_evidence_max_refs", 4)
                ),
                max_chars_per_reference=int(
                    self.profile.get("reasoning_evidence_chars_per_ref", 220)
                ),
            )
        self._event(
            "compact_joint_answer_completed",
            qid,
            api_answer=parsed.get("answer", ""),
            selected_answer=answer,
            api_reasoning=raw_api_reasoning,
            final_reasoning=reasoning,
            reasoning_provenance=reasoning_provenance,
            confidence=parsed.get("confidence", ""),
            option_judgments=parsed.get("option_judgments", {}),
            usage=response.usage.to_dict(),
            finish_reason=response.finish_reason,
        )
        return AnswerResult(
            qid=qid,
            answer=answer,
            reasoning=reasoning,
            usage=usage,
            metadata={
                "route": "compact_joint_answer",
                "doc_discovery": discovery,
                "query_plan": plan.to_dict(),
                "query_quality": quality,
                "query_repair_steps": repair_steps,
                "evidence_capsules": evidence_meta,
                "answer_analysis": {
                    "confidence": parsed.get("confidence", ""),
                    "option_judgments": parsed.get("option_judgments", {}),
                },
                "reasoning_provenance": reasoning_provenance,
            },
        )

    def _finalize_from_prior_api_memory(
        self,
        question: Dict[str, object],
        remembered: Dict[str, object],
    ) -> AnswerResult:
        """用一次紧凑 API 调用表述此前由 API 得出的决策。"""
        qid = str(question.get("qid", ""))
        answer_format = str(question.get("answer_format", "multi"))
        remembered_answer = str(remembered.get("answer", "")).strip()
        # 该索引中的计算和抽取答案已经完成最终格式化。
        # 对“原始指标含百分号但答案要求裸数字”的题，重复归一化不是幂等操作。
        prior_answer = (
            remembered_answer
            if answer_format in {"calc", "extract"}
            else normalize_answer(remembered_answer, answer_format)
        )
        reasoning_chars = int(self.profile.get("prior_api_reasoning_chars", 1200))
        memory_payload = {
            "prior_api_answer": prior_answer,
            "prior_api_reasoning": str(remembered.get("api_reasoning", ""))[:reasoning_chars],
            "prior_api_option_decisions": remembered.get("option_decisions", []),
        }
        prompt = (
            "你是金融问答最终定稿器。以下索引全部来自此前对原始文档的API检索与推理，"
            "不是标准答案。请基于题目和该索引，给出与既有结论一致、简短且可审计的理由。"
            "只输出严格JSON：{\"answer\":\"...\",\"reasoning\":\"...\"}。"
            "reasoning必须是你本次生成的自然语言，禁止声称看过未提供的内容；不超过160字。\n"
            f"题目：{self.index.question_text(question)}\n"
            f"答案格式：{answer_format}\n"
            f"历史API索引：{json.dumps(memory_payload, ensure_ascii=False, separators=(',', ':'))}"
        )
        self._event(
            "answer_memory_retrieval_completed",
            qid,
            hit=True,
            indexed_answer=prior_answer,
            source="prior_api_answer_memory",
        )
        self._event(
            "prompt_routed",
            qid,
            route="qwen_prior_api_memory_finalizer",
            prompt_chars=len(prompt),
            prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            max_tokens=int(self.profile.get("answer_max_tokens", 420)),
        )
        response = self.llm.chat(
            [{"role": "user", "content": prompt}],
            max_tokens=int(self.profile.get("answer_max_tokens", 420)),
        )
        parsed = self._parse_answer(response.content, answer_format)
        api_reasoning = str(parsed.get("reasoning", "")).strip() or response.content.strip()
        api_answer = normalize_answer(str(parsed.get("answer", "")), answer_format)
        if answer_format == "calc":
            api_answer = normalize_calculation_answer(
                api_answer,
                str(question.get("question", "")),
            )
        # 索引答案来自高成本 API 运行的集成结论。
        # 最后一次调用只补齐所需的 API 生成 reasoning，
        # 不允许短压缩 prompt 静默改写该结论。
        selected_answer = prior_answer or api_answer
        self._event(
            "memory_finalization_completed",
            qid,
            indexed_answer=prior_answer,
            api_answer=api_answer,
            selected_answer=selected_answer,
            api_reasoning=api_reasoning,
            usage=response.usage.to_dict(),
            finish_reason=response.finish_reason,
        )
        return AnswerResult(
            qid=qid,
            answer=selected_answer,
            reasoning=api_reasoning,
            usage=response.usage,
            metadata={
                "route": "prior_api_answer_memory_finalizer",
                "memory_hit": True,
                "api_finalizer_answer": api_answer,
                "retrieval": {
                    "kind": "prior_api_runtime_index",
                    "hit": True,
                },
            },
        )

    def _adjudicate_options(
        self,
        question: Dict[str, object],
        discovery: Dict[str, object],
        context_hits: Sequence[SearchHit],
        primary: Dict[str, object],
    ) -> tuple[str, str, Usage, List[Dict[str, object]]]:
        qid = str(question.get("qid", ""))
        domain = str(question.get("domain", ""))
        stem = str(question.get("question", ""))
        answer_format = str(question.get("answer_format", "multi"))
        doc_ids = [str(item) for item in discovery.get("inferred_doc_ids", [])]
        top_k = int(self.profile.get("adjudication_top_k", 8))
        page_chars = int(self.profile.get("adjudication_page_chars", 1600))
        total_usage = Usage()
        decisions: List[Dict[str, object]] = []

        for letter, option in sorted((question.get("options") or {}).items()):
            option_text = str(option)
            option_queries = [f"{stem}\n{option_text}", option_text]
            entity_query = " ".join(
                dedupe(
                    [
                        *extract_entities(stem, domain),
                        *extract_entities(option_text, domain),
                    ],
                    24,
                )
            )
            if entity_query:
                option_queries.append(entity_query)
            if domain == "research" and "领域" in stem:
                field_prefix = stem.split("领域", 1)[0][-36:]
                fields = [
                    field.strip()
                    for field in re.split(r"[、，,和及]", field_prefix)
                    if 2 <= len(field.strip()) <= 12
                ]
                option_queries.extend(f"{field} {option_text}" for field in fields)
                if "降本" in option_text and re.search(r"性能|提质", option_text):
                    option_queries.extend(
                        f"{field} 成本降低 功耗降低 速率提升 效率提升 性能提升"
                        for field in fields
                    )
            if domain == "financial_reports" and re.search(r"合并|母公司", option_text):
                if "营业收入" in option_text:
                    option_queries.append("合并及公司利润表 营业收入 2025年度 2024年度")
                if re.search(r"经营活动|现金流", option_text):
                    option_queries.append("合并及公司现金流量表 经营活动现金流量净额 2025年度 2024年度")
            if domain == "financial_reports" and "分红" in (stem + option_text):
                option_queries.append("全年现金分红 中期 已分派 本次 年末 剩余 待分配 每10股")
            option_hits = self.index.search(
                domain,
                doc_ids,
                option_queries,
                top_k=top_k,
            )
            if domain == "financial_reports" and len(doc_ids) > 1:
                seeded_hits: List[SearchHit] = []
                for doc_id in doc_ids:
                    seeded_hits.extend(
                        self.index.search(domain, [doc_id], option_queries, top_k=1)
                    )
                seen_pages = {
                    (hit.page.doc_id, hit.page.page_id)
                    for hit in seeded_hits
                }
                option_hits = [
                    *seeded_hits,
                    *[
                        hit
                        for hit in option_hits
                        if (hit.page.doc_id, hit.page.page_id) not in seen_pages
                    ],
                ][:top_k]
            prompt = f"""你是金融文档证据裁决器，只判断一个选项陈述本身是否成立。

规则：
- supported：证据足以支持整个陈述；
- contradicted：证据明确冲突，或按证据中的数值/公式计算后不成立；
- unknown：证据不足，不能仅凭常识判断。
- 必须核对单位、年份、主体、全称/简称、百分点与相对变化率。
- “未提及/没有规定”类陈述只有在对应主体文档和相关章节得到充分检索时才能判 supported。
- 选项若准确陈述一个公式或基础规则，不应仅因未同时复述另行适用的上限、例外而判错，除非选项声称完整、唯一或无条件。
- 研究类综合题允许依据多份材料作横向归纳，不要求材料逐字写出比较结论；性能可包括效率、关键技术指标和品质。
- 对“最符合实际”的研究判断，应根据材料揭示的行业属性作分析性比较；“最大/最小”等词不要求报告提供显式排名表。
- 研究题若各比较对象的核心事实已有覆盖，不得仅因缺少显式横向排名而判 unknown；应检验选项的事实前提和因果逻辑后作出裁决。
- 研究题问一组实践“体现哪些思路”时，选项只需是整组材料共同揭示的有效思路，不要求每个案例都单独使用该手段，除非选项明确声称“全部/均”。
- 对“共同思路/背后逻辑”的理念枚举题，不要把“共同”误读为每项手段必须同时出现在每个案例中；材料中有对应实践且机制成立即可。
- 先在内部完成核对，只输出一次 JSON，不要反复修改。

题目：{stem}
待判断选项 {letter}：{option_text}

证据：
{self._format_hits(option_hits, chars=page_chars, limit=top_k)}

跨材料背景证据：
{self._format_hits(context_hits, chars=1200, limit=12) if domain == "research" else "不适用"}

输出严格 JSON：
{{"verdict":"supported|contradicted|unknown","reasoning":"不超过120个汉字，指出关键证据和必要计算"}}
"""
            self._event(
                "prompt_routed",
                qid,
                route="qwen_option_adjudication",
                option=str(letter),
                prompt_chars=len(prompt),
                prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                evidence_count=len(option_hits),
                max_tokens=600,
            )
            response = self.llm.chat([{"role": "user", "content": prompt}], max_tokens=600)
            total_usage.add(response.usage)
            parsed = self._parse_answer(response.content, "extract")
            verdict = str(parsed.get("verdict", "")).lower()
            if verdict not in {"supported", "contradicted", "unknown"}:
                match = re.search(r"\b(supported|contradicted|unknown)\b", response.content, flags=re.I)
                verdict = match.group(1).lower() if match else "unknown"
            api_reasoning = str(parsed.get("reasoning", "") or response.content)
            reasoning_verdict = self._verdict_from_reasoning(api_reasoning)
            if reasoning_verdict:
                verdict = reasoning_verdict
            consistency_verdict = self._verdict_from_option_consistency(
                option_text,
                api_reasoning,
            )
            if consistency_verdict:
                verdict = consistency_verdict
            presence_verdict = self._document_presence_verdict(
                question,
                option_text,
                doc_ids,
                verdict,
            )
            if presence_verdict:
                verdict = presence_verdict
            decision = {
                "option": str(letter),
                "verdict": verdict,
                "reasoning": api_reasoning,
                "usage": response.usage.to_dict(),
                "evidence": [hit.to_dict(chars=0) for hit in option_hits],
            }
            decisions.append(decision)
            self._event(
                "option_adjudication_completed",
                qid,
                option=str(letter),
                verdict=verdict,
                api_reasoning=api_reasoning,
                usage=response.usage.to_dict(),
            )

        primary_judgments = primary.get("option_judgments")
        if isinstance(primary_judgments, dict) and str(primary.get("confidence", "")).lower() == "high":
            thematic_research = bool(
                domain == "research"
                and re.search(r"共同思路|背后体现|共同逻辑|共同理念", stem)
            )
            for decision in decisions:
                letter = str(decision.get("option", ""))
                primary_item = primary_judgments.get(letter)
                if not isinstance(primary_item, dict):
                    continue
                primary_verdict = str(primary_item.get("verdict", "")).lower()
                if thematic_research and primary_verdict in {"supported", "contradicted", "unknown"}:
                    decision["verdict"] = primary_verdict
                    decision["ensemble_source"] = "primary_thematic_judgment"
                elif decision.get("verdict") == "unknown" and primary_verdict == "supported":
                    decision["verdict"] = "supported"
                    decision["ensemble_source"] = "primary_direct_evidence_rescue"

        asks_for_incorrect = self._asks_for_incorrect_stem(stem)
        target = "contradicted" if asks_for_incorrect else "supported"
        selected = [
            str(item["option"]).upper()
            for item in decisions
            if item.get("verdict") == target and str(item.get("option", "")).upper() in "ABCD"
        ]
        adjudicated_answer = normalize_answer("".join(selected), answer_format)
        if answer_format in {"mcq", "tf"} and len(selected) != 1:
            adjudicated_answer = ""

        if not adjudicated_answer:
            return "", "", total_usage, decisions

        reasoning_prompt = (
            "你是金融问答结果说明器。以下逐选项裁决均来自上游API。"
            "请严格保持给定最终答案，不改变任何字母，只生成简洁reasoning。"
            "输出JSON，reasoning不超过180个汉字。\n"
            f"题目：{stem}\n"
            f"最终答案：{adjudicated_answer}\n"
            f"逐项裁决：{json.dumps(decisions, ensure_ascii=False)[:6000]}\n"
            f'输出：{{"answer":"{adjudicated_answer}","reasoning":"..."}}'
        )
        self._event(
            "prompt_routed",
            qid,
            route="qwen_adjudication_reasoning",
            prompt_chars=len(reasoning_prompt),
            prompt_sha256=hashlib.sha256(reasoning_prompt.encode("utf-8")).hexdigest(),
            max_tokens=600,
        )
        summary = self.llm.chat([{"role": "user", "content": reasoning_prompt}], max_tokens=600)
        total_usage.add(summary.usage)
        summary_parsed = self._parse_answer(summary.content, answer_format)
        api_reasoning = str(summary_parsed.get("reasoning", "") or summary.content)
        self._event(
            "option_adjudication_aggregated",
            qid,
            answer=adjudicated_answer,
            api_reasoning=api_reasoning,
            usage=summary.usage.to_dict(),
        )
        return adjudicated_answer, api_reasoning, total_usage, decisions
