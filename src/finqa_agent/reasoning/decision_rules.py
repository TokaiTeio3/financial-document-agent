"""确定性的答案解析、校准与裁决规则。"""

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



class DecisionRulesMixin:
    def _answer_from_explicit_arithmetic(self, reasoning: str, question: str) -> str:
        matches = re.findall(
            r"(?:合计|总计)[^0-9]{0,12}"
            r"((?:\d+(?:\.\d+)?\s*[+\-*/×÷]\s*)+\d+(?:\.\d+)?)"
            r"\s*[=＝]\s*[-+]?\d+(?:\.\d+)?",
            reasoning,
        )
        if not matches:
            if re.search(r"答案格式.{0,80}[；;]", question):
                return ""
            ratio_matches = re.findall(
                r"([0-9().\s+\-*/×÷]{5,})\s*(?:≈|=|约为)\s*"
                r"[-+]?\d+(?:\.\d+)?\s*(%)",
                reasoning,
            )
            if not ratio_matches:
                return ""
            expression, suffix = ratio_matches[-1]
            expression = expression.strip().replace("×", "*").replace("÷", "/")
            if not re.search(r"[+\-*/]", expression):
                return ""
            result = self.executor.execute(f"answer = ({expression}) * 100")
            if not result.ok or result.answer == "":
                return ""
            return normalize_calculation_answer(f"{result.answer}{suffix}", question)
        expression = matches[-1].replace("×", "*").replace("÷", "/")
        result = self.executor.execute(f"answer = {expression}")
        if not result.ok or result.answer == "":
            return ""
        return normalize_calculation_answer(result.answer, question)

    @staticmethod
    def _final_calculation_answer_from_reasoning(reasoning: str, question: str) -> str:
        """优先采用 API 的明确最终结论，而非较早的草算公式。"""
        if re.search(r"[；;]", question):
            calculated_percentages = re.findall(
                r"(?:≈|=)\s*(\d+(?:\.\d+)?)\s*%",
                reasoning,
            )
            point_gains = re.findall(
                r"(\d+(?:\.\d+)?)\s*个百分点",
                reasoning,
            )
            if calculated_percentages and point_gains:
                return normalize_calculation_answer(
                    f"{calculated_percentages[0]}；{point_gains[-1]}",
                    question,
                )
        marker_pattern = (
            r"(?:最终(?:评价计分|计分|结果|答案)?|正确答案(?:应)?为|"
            r"(?:确认)?结果(?:应)?为|答案为|"
            r"保留.{0,12}?(?:为|得)|更正为|修正为|应为|输出|因此|所以|故)"
        )
        markers = list(re.finditer(marker_pattern, reasoning))
        if not markers:
            if re.search(r"答案格式.{0,100}[；;]", question):
                growth = re.findall(
                    r"同比增幅\s*(?:=|≈|约为).{0,80}?"
                    r"(\d+(?:\.\d+)?)\s*%",
                    reasoning,
                )
                point_gain = re.findall(
                    r"(?:提高|增加).{0,24}?(\d+(?:\.\d+)?)\s*个百分点",
                    reasoning,
                )
                if growth and point_gain:
                    return normalize_calculation_answer(
                        f"{growth[-1]}；{point_gain[-1]}",
                        question,
                    )
            return ""
        marker = markers[-1]
        tail = reasoning[marker.start():].strip()
        # 最终结论通常在下一个句子边界前结束。
        # 只抽取该句可以避免后续证据引用或被否决的草稿值覆盖结论。
        tail = re.split(r"[。；\n]", tail, maxsplit=1)[0][:240]

        dates = re.findall(
            r"(?:20\d{2}年\d{1,2}月\d{1,2}日|20\d{2}[-/.]\d{1,2}[-/.]\d{1,2})",
            tail,
        )
        if dates:
            return normalize_calculation_answer(dates[-1], question)

        if re.search(r"答案格式.{0,100}[；;]", question):
            compounds = re.findall(
                r"[-+]?\d+(?:\.\d+)?(?:%|元|万元|亿元|天|日|分|倍)?"
                r"(?:\s*[；;]\s*[-+]?\d+(?:\.\d+)?(?:%|元|万元|亿元|天|日|分|倍)?)+",
                tail,
            )
            if compounds:
                return normalize_calculation_answer(compounds[-1], question)
            return ""

        numbers = re.findall(
            r"[-+]?\d+(?:\.\d+)?(?:%|元|万元|亿元|天|日|分|倍)?",
            tail,
        )
        if not numbers:
            return ""
        return normalize_calculation_answer(numbers[-1], question)

    @staticmethod
    def _final_scalar_from_reasoning(reasoning: str, question: str) -> str:
        """为复合答案的最后一个字段提取修正后的最终标量。"""
        markers = list(
            re.finditer(
                r"(?:更正为|修正为|应为|重新(?:核算|计算).{0,12}(?:为|得)|"
                r"精确计算.{0,8}(?:为|得)|保留.{0,12}(?:为|得))",
                reasoning,
            )
        )
        if not markers:
            return ""
        for marker in reversed(markers):
            tail = reasoning[marker.start():]
            tail = re.split(r"[。\n]", tail, maxsplit=1)[0][:160]
            numbers = re.findall(
                r"[-+]?\d+(?:\.\d+)?(?:%|元|万元|亿元|天|日|分|倍)?",
                tail,
            )
            if numbers:
                return normalize_calculation_answer(numbers[0], question)
        return ""

    @staticmethod
    def _ratio_point_tail_from_reasoning(reasoning: str, question: str) -> str:
        return ratio_point_tail_from_reasoning(reasoning, question)

    @staticmethod
    def _verdict_from_reasoning(reasoning: str) -> str:
        """解决 API 裁决字段与其最终语句相矛盾的问题。"""
        signals: List[tuple[int, int, str]] = []
        patterns = {
            "supported": (
                r"\bsupported\b|(?:判定|判|应|修正|更正)(?:为|成)\s*supported\b|"
                r"应判\s*supported\b|"
                r"陈述成立|表述成立|选项[A-D]?正确|表述正确|准确无误|"
                r"(?:实际)?数据.{0,6}支持(?:该|此)?(?:判断|选项|表述)|计算(?:结果)?无误|"
                r"(?:选项[A-D]?|陈述|表述).{0,6}(?:准确陈述|陈述准确|表述准确)|"
                r"与(?:证据|原文|规定).{0,8}(?:一致|吻合)|"
                r"(?:表述|选项).{0,8}(?:证据逻辑|证据|原文).{0,6}一致"
            ),
            "contradicted": (
                r"\bcontradicted\b|(?:判定|判|应|修正|更正)(?:为|成)\s*contradicted\b|"
                r"应判\s*contradicted\b|"
                r"陈述错误|表述错误|选项[A-D]?错误|不成立|"
                r"(?:故|因此).{0,8}(?:矛盾|冲突|不符|错误)|"
                r"未出现选项所述|证据未支持(?:其)?存在|"
                r"(?:标题|名称|表述).{0,16}与原文不一致|"
                r"(?:数值|日期|主体|单位|公式|排序|比例|降幅|增幅)"
                r".{0,8}(?:错误|冲突|不符)|"
                r"与(?:证据|原文|规定|数据|事实).{0,8}(?:矛盾|冲突|不符)|"
                r"与事实相反|与实际相反|严重失实|并非.{0,12}一致|不一致"
            ),
            "unknown": (
                r"无法判断|无法确认|无法核实|无法核对|无法验证|无法证明|不能证明|未能证明|"
                r"证据不足|关键事实缺证|不能确定|不足以判断|"
                r"无法(?:判断|确认|核实).{0,12}(?:陈述|选项|表述)?(?:成立|正确)"
            ),
        }
        priorities = {"supported": 1, "unknown": 2, "contradicted": 3}
        for verdict, pattern in patterns.items():
            for match in re.finditer(pattern, reasoning):
                if verdict == "supported" and re.search(
                    r"不(?:一致|吻合|成立|正确)|(?:矛盾|冲突|不符|错误|有误)|"
                    r"是否正确|无法(?:判断|确认|核实|验证)|证据不足|关键事实缺证|"
                    r"(?<![A-Za-z])contradicted(?![A-Za-z])",
                    match.group(0),
                ):
                    continue
                if verdict == "contradicted" and re.search(
                    r"(?:并|且)?无(?:任何)?(?:矛盾|冲突|不符|错误)|"
                    r"(?:没有|不存在)(?:矛盾|冲突|不符|错误)",
                    match.group(0),
                ):
                    continue
                signals.append((match.end(), priorities[verdict], verdict))
        return max(signals, default=(-1, -1, ""))[2]

    @staticmethod
    def _verdict_from_option_consistency(option_text: str, reasoning: str) -> str:
        if (
            re.search(
                r"最高.{0,16}(?:借款|贷款).{0,20}现金价值.{0,12}80%",
                option_text,
            )
            and not re.search(r"仅|恰好|无条件|不扣除", option_text)
            and re.search(
                r"现金价值.{0,20}扣除.{0,20}(?:借款|利息).{0,20}80%",
                reasoning,
            )
            and not re.search(
                r"比例.{0,8}(?:错误|不同)|80%.{0,8}(?:错误|不同)",
                reasoning,
            )
        ):
            return "supported"
        if (
            re.search(
                r"占.{0,20}\d+(?:\.\d+)?%.{0,30}"
                r"(?:披露|年报).{0,20}\d+(?:\.\d+)?%.{0,16}(?:一致|接近)",
                option_text,
            )
            and re.search(r"(?:数学|计算逻辑).{0,12}自洽", reasoning)
            and re.search(r"缺(?:少|乏).{0,16}(?:原始|分子|具体)数据", reasoning)
            and not re.search(
                r"(?:口径|年度|主体|单位).{0,8}(?:错误|冲突|不符)",
                reasoning,
            )
        ):
            return "supported"
        if (
            re.search(r"占.{0,16}约\s*\d+(?:\.\d+)?%", option_text)
            and re.search(
                r"披露.{0,16}\d+(?:\.\d+)?%.{0,16}(?:一致|近似|自洽)",
                option_text,
            )
            and re.search(
                r"(?:计算比例|计算结果).{0,20}\d+(?:\.\d+)?%"
                r".{0,30}(?:披露|年报).{0,20}\d+(?:\.\d+)?%"
                r".{0,20}(?:一致|近似|自洽)",
                reasoning,
            )
            and not re.search(
                r"(?:口径|年度|主体|单位).{0,8}(?:错误|冲突|不符)",
                reasoning,
            )
        ):
            return "supported"
        if (
            re.search(r"并(?:遵守|符合|履行)", option_text)
            and re.search(
                r"(?:后半句|另一项|同时).{0,20}(?:未提及|未规定|无直接依据)|"
                r"(?:未提及|未规定).{0,20}(?:增设|添加|另一)",
                reasoning,
            )
            and not re.search(r"明确(?:禁止|排除)|与.{0,16}(?:相反|不能并行)", reasoning)
        ):
            return "supported"
        if re.search(
            r"计算结果.{0,30}(?:与|和).{0,20}披露值.{0,20}"
            r"(?:一致|吻合|自洽)|"
            r"与披露值.{0,20}(?:逻辑)?自洽",
            reasoning,
        ) and not re.search(
            r"(?:主体|年度|口径|单位|公式).{0,10}(?:错误|冲突|不符)",
            reasoning,
        ):
            return "supported"
        if (
            "约" in option_text
            and re.search(r"(?:0\.0?1|尾差|四舍五入).{0,20}(?:偏差|差异|不精确|判错)", reasoning)
            and re.search(r"吻合|相符|支持|成立|可接受|实际", reasoning)
        ):
            return "supported"
        if (
            "连续" in option_text
            and re.search(r"连续下降", reasoning)
            and re.search(r"连续上升", reasoning)
            and not re.search(
                r"(?:该主体|选项所述主体).{0,12}(?:数据缺失|无法验证)|"
                r"(?:方向|年份|数值).{0,8}(?:错误|冲突|不符)",
                reasoning,
            )
        ):
            return "supported"
        if (
            "约" in option_text
            and re.search(r"尾差|四舍五入|相差\s*0\.0?1|存在.{0,8}差", reasoning)
            and re.search(r"方向.{0,4}对|数值.{0,4}对|实际支持|基本吻合", reasoning)
        ):
            return "supported"
        if (
            re.search(r"无.{0,18}(?:欠款|借款|利息|欠交)", option_text)
            and re.search(r"数值结果相同|余额等于|扣除额为0|无需扣除", reasoning)
            and re.search(r"遗漏|未复述|未同时", reasoning)
        ):
            return "supported"
        if (
            re.search(r"无.{0,22}(?:欠款|借款|利息|欠交)", option_text)
            and re.search(
                r"现金价值.{0,30}扣除.{0,30}(?:欠款|借款|利息|欠交).{0,30}80%",
                reasoning,
            )
            and not re.search(r"(?:比例|数值).{0,8}(?:不同|错误|不符)", reasoning)
        ):
            return "supported"
        if (
            re.search(
                r"现金价值.{0,35}扣除.{0,35}(?:欠交|借款|利息).{0,35}80%",
                option_text,
            )
            and re.search(
                r"现金价值.{0,35}扣除.{0,35}(?:欠交|借款|利息).{0,35}80%",
                reasoning,
            )
            and not re.search(r"(?:比例|数值).{0,8}(?:不同|错误|不符)", reasoning)
        ):
            return "supported"
        if (
            not re.search(r"仅|只需|无需|唯一|完整|全部流程|无条件", option_text)
            and re.search(
                r"遗漏了?(?:另一|其他|法定|同时).{0,16}(?:义务|步骤|流程)|"
                r"未同时(?:说明|复述|提及).{0,16}(?:义务|步骤|条件)|"
                r"未提及.{0,16}(?:反馈|核实|报告).{0,8}义务",
                reasoning,
            )
            and not re.search(
                r"(?:主体|时点|数值|方向|适用范围).{0,8}(?:错误|冲突)",
                reasoning,
            )
        ):
            return "supported"
        if (
            "至少" in option_text
            and re.search(
                r"遗漏.{0,12}(?:起算|触发|前提)|"
                r"并非无条件|仅.{0,16}(?:结束|终止|发生)后",
                reasoning,
            )
            and not re.search(r"(?:期限|年限|数值).{0,8}(?:错误|不符|冲突)", reasoning)
        ):
            return "supported"
        if re.search(
            r"(?:差额|计算结果|实际值|实际).{0,80}(?:与选项|不等于|并非).{0,30}"
            r"(?:不符|矛盾|错误)",
            reasoning,
        ):
            return "contradicted"
        universal_claim = bool(
            re.search(
                r"^(?:无论|不论).{0,24}(?:都|均)|"
                r"(?:无论|不论)是否|"
                r"一律|"
                r"任何.{0,12}(?:都|均)|"
                r"全部.{0,12}(?:都|均)",
                option_text,
            )
        )
        conditional_refutation = bool(
            re.search(
                r"并非.{0,20}(?:无论|均|一律)|"
                r"(?:仅|只)(?:在|有|对).{0,30}(?:时|情况下|才)|"
                r"只有.{0,30}才|"
                r"未达.{0,30}(?:若|仅在|只有)|"
                r"低于.{0,30}(?:若|仅在|只有)",
                reasoning,
            )
        )
        if universal_claim and conditional_refutation:
            return "contradicted"
        if "排序" not in option_text:
            return ""
        option_match = re.search(r"排序(?:为|是)?\s*[:：]\s*([^。；]+)", option_text)
        reasoning_match = re.search(
            r"(?:正确)?排序(?:应)?为\s*[:：]?\s*([^。；]+)",
            reasoning,
        )
        if not option_match or not reasoning_match:
            return ""

        def sequence(value: str) -> List[str]:
            value = re.split(r"[，,]?(?:但|然而|选项|实际)", value, maxsplit=1)[0]
            parts = re.split(r"[<>＜＞、，,]", value)
            cleaned = []
            for part in parts:
                item = re.sub(r"[（(].*?[）)]", "", part)
                item = re.sub(r"[-+]?\d+(?:\.\d+)?(?:元|%|倍)?", "", item)
                item = re.sub(r"^(?:依次为|从高到低|由高到低)", "", item)
                item = re.sub(r"\s+", "", item).strip("：:。；")
                if item:
                    cleaned.append(item)
            return cleaned

        option_sequence = sequence(option_match.group(1))
        reasoning_sequence = sequence(reasoning_match.group(1))
        if len(option_sequence) >= 2 and len(reasoning_sequence) >= 2:
            def is_subsequence(shorter: str, longer: str) -> bool:
                cursor = 0
                for char in longer:
                    if cursor < len(shorter) and shorter[cursor] == char:
                        cursor += 1
                return cursor == len(shorter)

            def same_item(left: str, right: str) -> bool:
                if left == right:
                    return True
                shorter, longer = sorted((left, right), key=len)
                return len(shorter) >= 2 and is_subsequence(shorter, longer)

            if len(option_sequence) != len(reasoning_sequence):
                return "contradicted"
            return (
                "supported"
                if all(
                    same_item(left, right)
                    for left, right in zip(option_sequence, reasoning_sequence)
                )
                else "contradicted"
            )
        return ""

    def _document_presence_verdict(
        self,
        question: Dict[str, object],
        option_text: str,
        doc_ids: Sequence[str],
        current_verdict: str,
    ) -> str:
        stem = str(question.get("question", ""))
        negative_presence = bool(
            re.search(
                r"未.{0,24}(?:提及|列明|包含|规定)|没有.{0,24}(?:提及|列明|包含|规定)",
                option_text,
            )
            and re.search(r"原文|明确(?:列明|提及|包含|规定)", stem)
        )
        if not negative_presence and (
            current_verdict != "unknown"
            or not re.search(r"明确(?:列明|提及|包含|规定)", stem)
        ):
            return ""
        quoted = re.findall(r"[“\"]([^”\"]{2,60})[”\"]", stem)
        if any(len(compact(subject)) > 12 for subject in quoted):
            # 长引用条款包含多个必要条件。
            # 单个原子条件的字面出现不足以解决未知判定。
            return ""
        if quoted:
            subjects = quoted
        else:
            subject_match = re.search(r"关于(.{2,50}?)[，,。]", stem)
            subjects = [subject_match.group(1)] if subject_match else []
        topic_terms = dedupe(
            [
                part.strip("“”\"' 等相关风险免责或除外责任")
                for subject in subjects
                for part in re.split(r"[、，,；;/／或和及]", subject)
            ],
            30,
        )
        topic_terms = [term for term in topic_terms if len(compact(term)) >= 2]
        if not topic_terms:
            return ""

        domain = str(question.get("domain", ""))
        option_compact = canonical(option_text)
        option_core = re.sub(
            r"(?:养老年金保险|终身寿险|重大疾病保险|重疾险|医疗保险|"
            r"责任保险|商业保险|财产保险|保险).*$",
            "",
            option_compact,
        )
        core_bigrams = {
            option_core[index:index + 2]
            for index in range(max(0, len(option_core) - 1))
        }
        matched_by_binding = self._docs_named_in_option(domain, doc_ids, option_text)
        if not matched_by_binding:
            # 显式存在性问题可能提到词法发现未覆盖的文档。
            # 这里用该领域全部文档构造的别名重新绑定，不注入实体或答案。
            matched_by_binding = self._docs_named_in_option(domain, [], option_text)
        if not matched_by_binding:
            return ""
        texts = [
            canonical(self.index.doc_texts.get(domain, {}).get(doc_id, ""))
            for doc_id in dedupe(matched_by_binding)
        ]
        mentioned = any(
            canonical(term) in text
            for term in topic_terms
            for text in texts
        )
        if negative_presence:
            return "contradicted" if mentioned else "supported"
        return "supported" if mentioned else "contradicted"

    @staticmethod
    def _asks_for_incorrect_stem(stem: str) -> bool:
        """识别题目要求的正反极性，不把背景否定词误当作指令。"""
        return bool(
            re.search(
                r"(?:下列|以下|其中|哪些).{0,12}(?:说法|表述|选项)?.{0,6}"
                r"(?:错误|不正确|不符合|不一致|有误|不能成立|不属于)",
                stem,
            )
            or re.search(
                r"(?:错误|不正确|不符合|不一致|有误|不能成立|不属于)(?:的)?是",
                stem,
            )
        )

    @staticmethod
    def _parse_answer(raw: str, answer_format: str) -> Dict[str, object]:
        candidates = [raw]
        candidates.extend(re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", raw, flags=re.S | re.I))
        if "{" in raw and "}" in raw:
            candidates.append(raw[raw.find("{"): raw.rfind("}") + 1])
        for candidate in candidates:
            try:
                data = json.loads(candidate)
            except Exception:
                continue
            if isinstance(data, dict):
                return data
        # 被截断的模型回复可能只有 JSON 前部字段而没有右花括号。
        # 只恢复显式 answer 字段；若解析自由文本中的所有 A-D，会把截断静默变成 ABCD。
        partial_answer = ""
        answer_match = re.search(r'["\']answer["\']\s*:\s*["\']([^"\']*)', raw, flags=re.I)
        if answer_match:
            partial_answer = normalize_answer(answer_match.group(1), answer_format)
        return {
            "answer": partial_answer,
            "reasoning": raw,
            "_parse_failed": True,
        }

    @staticmethod
    def _answer_from_judgments(
        parsed: Dict[str, object],
        answer_format: str,
        question: Dict[str, object] | None = None,
    ) -> str:
        if answer_format not in {"mcq", "multi", "tf"}:
            return ""
        judgments = parsed.get("option_judgments")
        if not isinstance(judgments, dict):
            return ""
        stem = str((question or {}).get("question", ""))
        asks_for_incorrect = DecisionRulesMixin._asks_for_incorrect_stem(stem)
        target_verdict = "contradicted" if asks_for_incorrect else "supported"
        supported = [
            str(key).upper()
            for key, value in sorted(judgments.items())
            if isinstance(value, dict) and str(value.get("verdict", "")).lower() == target_verdict
        ]
        supported = [key for key in supported if key in "ABCD"]
        if answer_format == "multi" and supported:
            return "".join(supported)
        if answer_format in {"mcq", "tf"} and len(supported) == 1:
            return supported[0]
        return ""

    @staticmethod
    def _answer_from_reasoning(
        reasoning: str,
        answer_format: str,
        declared_answer: str = "",
    ) -> str:
        if answer_format not in {"mcq", "multi", "tf"}:
            return ""
        matches = re.findall(
            r"(?:最终|综上|结论|因此|故)[^。；，,\n]{0,16}?"
            r"(?:答案|选项|说法)?(?:应为|为|是|选择)\s*[:：]?\s*([A-D]{1,4})",
            reasoning.upper(),
        )
        matches.extend(
            re.findall(
                r"(?:最终|综上|结论|因此|故)\s*(?:仅|应选|选择)?\s*"
                r"([A-D]{1,4})\s*(?:项)?(?:成立|正确|应选)",
                reasoning.upper(),
            )
        )
        matches.extend(
            re.findall(
                r"(?:成立项|正确项|应选项|最终答案)\s*(?:应为|为|是|包括)?\s*"
                r"[:：]?\s*([A-D]{2,4}|[A-D](?:\s*[、，,]\s*[A-D]){1,3}|[A-D])",
                reasoning.upper(),
            )
        )
        matches.extend(
            re.findall(
                r"([A-D](?:\s*[、，,]\s*[A-D]){1,3})\s*"
                r"(?:均|都).{0,8}?(?:SUPPORTED|成立|支持|正确|应选)",
                reasoning.upper(),
            )
        )
        matches.extend(
            re.findall(
                r"(?:应全选|应选|答案应为|正确答案(?:应)?为)"
                r"\s*[:：]?\s*([A-D]{2,4})",
                reasoning.upper(),
            )
        )
        if not matches:
            return ""
        candidate = normalize_answer(
            re.sub(r"[、，,\s]", "", matches[-1]),
            answer_format,
        )
        if answer_format == "multi":
            return candidate
        return candidate[:1]

    @staticmethod
    def _needs_calculation(question: Dict[str, object]) -> bool:
        text = str(question.get("question", "")) + "\n" + "\n".join(str(v) for v in (question.get("options") or {}).values())
        return str(question.get("answer_format", "")) == "calc" or any(term in text for term in CALC_TERMS)

    @staticmethod
    def _needs_reaction(question: Dict[str, object], answer: str, parsed: Dict[str, object], quality: Dict[str, object]) -> bool:
        if parsed.get("_parse_failed"):
            return True
        if not answer:
            return True
        if quality.get("needs_repair"):
            return True
        if str(parsed.get("confidence", "")).lower() == "low":
            return True
        judgments = parsed.get("option_judgments")
        if isinstance(judgments, dict) and any(
            isinstance(value, dict) and value.get("verdict") == "unknown"
            for value in judgments.values()
        ):
            return True
        reasoning = str(parsed.get("reasoning", ""))
        return len(reasoning) < 12
