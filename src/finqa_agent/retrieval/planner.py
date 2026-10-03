from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List

from ..core.entities import CALC_TERMS, DOMAIN_TERMS, NEGATIVE_TERMS, dedupe, extract_entities
from ..runtime.llm import QwenClient, Usage


@dataclass
class QueryPlan:
    qid: str
    domain: str
    useful_entities: List[str] = field(default_factory=list)
    global_queries: List[str] = field(default_factory=list)
    option_queries: Dict[str, List[str]] = field(default_factory=dict)
    negative_queries: List[str] = field(default_factory=list)
    calculation_queries: List[str] = field(default_factory=list)
    must_cover_terms: List[str] = field(default_factory=list)
    source: str = "fallback"
    raw_response: str = ""
    usage: Usage = field(default_factory=Usage)

    def to_dict(self) -> Dict[str, object]:
        return {
            "qid": self.qid,
            "domain": self.domain,
            "useful_entities": self.useful_entities,
            "global_queries": self.global_queries,
            "option_queries": self.option_queries,
            "negative_queries": self.negative_queries,
            "calculation_queries": self.calculation_queries,
            "must_cover_terms": self.must_cover_terms,
            "source": self.source,
            "llm_usage": self.usage.to_dict(),
        }

    def all_queries(self) -> List[str]:
        values: List[str] = []
        values.extend(self.global_queries)
        for queries in self.option_queries.values():
            values.extend(queries)
        values.extend(self.negative_queries)
        values.extend(self.calculation_queries)
        return dedupe(values, 100)


class QueryPlanner:
    def __init__(self, llm: QwenClient, max_tokens: int = 1200) -> None:
        self.llm = llm
        self.max_tokens = max_tokens

    def build(self, question: Dict[str, object]) -> QueryPlan:
        fallback = self._fallback(question)
        prompt = self._prompt(question, fallback)
        try:
            response = self.llm.chat([{"role": "user", "content": prompt}], max_tokens=self.max_tokens)
            data = self._extract_json(response.content)
            plan = QueryPlan(
                qid=str(question.get("qid", "")),
                domain=str(question.get("domain", "")),
                useful_entities=self._list(data.get("useful_entities"), 40) or fallback.useful_entities,
                global_queries=self._list(data.get("global_queries"), 10) or fallback.global_queries,
                option_queries=self._option_queries(data.get("option_queries"), question),
                negative_queries=self._list(data.get("negative_queries"), 8),
                calculation_queries=self._list(data.get("calculation_queries"), 8),
                must_cover_terms=self._list(data.get("must_cover_terms"), 40) or fallback.must_cover_terms,
                source="qwen",
                raw_response=response.content,
                usage=response.usage,
            )
            if not any(plan.option_queries.values()):
                plan.option_queries = fallback.option_queries
            return plan
        except Exception:
            return fallback

    def build_local(self, question: Dict[str, object]) -> QueryPlan:
        """仅使用题目中出现的实体构建确定性计划。"""
        return self._fallback(question)

    def _prompt(self, question: Dict[str, object], fallback: QueryPlan) -> str:
        domain = str(question.get("domain", ""))
        options = question.get("options") or {}
        option_text = "\n".join(f"{k}. {v}" for k, v in sorted(options.items()))
        domain_terms = DOMAIN_TERMS.get(domain, [])
        return f"""你是金融长文档问答系统的检索 query 规划器。
你的任务只包括：
1. 从当前题目中识别有用实体、产品、法规名、公司名、年份、数值、指标、条款词。
2. 生成用于本地 BM25/关键词检索的 query。
3. 不判断答案，不输出正确选项，不写结论。

要求：
- query 要尽量短，优先保留“实体/主体 + 指标/条款 + 年份/数值/限制条件”。
- 对每个选项生成 2-4 条 option query。
- 如果题目含否定、除外、不得、无需、不承担等，要生成 negative_queries。
- 如果题目需要计算，要生成 calculation_queries，用于召回公式、比例、金额、表格依据。
- 允许参考领域通用词，但最终 query 必须基于当前题目。
- 输出严格 JSON，不要 Markdown。

JSON 字段：
useful_entities, global_queries, option_queries, negative_queries, calculation_queries, must_cover_terms

qid: {question.get("qid", "")}
domain: {domain}
answer_format: {question.get("answer_format", "")}
question: {question.get("question", "")}
options:
{option_text}

领域通用检索词:
{json.dumps(domain_terms, ensure_ascii=False)}

本地代码抽取到的候选词，仅作参考:
{json.dumps(fallback.useful_entities, ensure_ascii=False)}
"""

    def _fallback(self, question: Dict[str, object]) -> QueryPlan:
        domain = str(question.get("domain", ""))
        qid = str(question.get("qid", ""))
        options = question.get("options") or {}
        qtext = str(question.get("question", ""))
        all_text = "\n".join([qtext, *[str(v) for v in options.values()]])
        entities = extract_entities(all_text, domain)
        domain_hits = [t for t in DOMAIN_TERMS.get(domain, []) if t in all_text]
        numbers = re.findall(r"20\d{2}|\d+(?:\.\d+)?\s*(?:亿元|万元|元|%|日|个月|年|倍)?", all_text)
        focus = dedupe([*entities, *domain_hits, *numbers], 40)
        question_focus = re.sub(
            r"(根据|依据|关于|下列|以下|说法|选项|正确的是|错误的是|不正确的是|符合的是|不符合的是|请选择|请问)",
            " ",
            qtext,
        )
        question_focus = re.sub(r"\s+", " ", question_focus).strip()
        global_queries = [
            question_focus[:240],
            " ".join(focus[:12]),
            " ".join(dedupe([*domain_hits, *numbers], 16)),
        ]
        option_queries: Dict[str, List[str]] = {}
        for key, value in sorted(options.items()):
            option_text = re.sub(r"\s+", " ", str(value)).strip()
            opt_entities = extract_entities(str(value), domain)
            opt_numbers = re.findall(r"20\d{2}|\d+(?:\.\d+)?\s*(?:亿元|万元|元|%|日|个月|年|倍)?", str(value))
            option_queries[str(key)] = dedupe([
                f"{question_focus[:120]} {option_text[:180]}".strip(),
                option_text[:240],
                " ".join(dedupe([*opt_entities, *domain_hits[:6], *opt_numbers], 12)),
                " ".join(dedupe([*entities[:6], *opt_entities[:6], *opt_numbers], 12)),
            ])
        negative = [f"{' '.join(focus[:8])} {term}" for term in NEGATIVE_TERMS if term in all_text]
        calc = []
        if any(term in all_text for term in CALC_TERMS) or len(numbers) >= 2:
            calc = [" ".join(dedupe([*focus[:10], "公式", "比例", "金额", "计算"], 16))]
        return QueryPlan(
            qid=qid,
            domain=domain,
            useful_entities=focus,
            global_queries=[q for q in dedupe(global_queries) if q],
            option_queries=option_queries,
            negative_queries=dedupe(negative, 8),
            calculation_queries=dedupe(calc, 8),
            must_cover_terms=focus,
            source="fallback",
        )

    @staticmethod
    def _extract_json(text: str) -> Dict[str, object]:
        candidates = [text]
        candidates.extend(re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S | re.I))
        if "{" in text and "}" in text:
            candidates.append(text[text.find("{"): text.rfind("}") + 1])
        for candidate in candidates:
            try:
                data = json.loads(candidate)
            except Exception:
                continue
            if isinstance(data, dict):
                return data
        raise ValueError("query planner did not return JSON")

    @staticmethod
    def _list(value: object, limit: int) -> List[str]:
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            return []
        return dedupe(value, limit)

    def _option_queries(self, value: object, question: Dict[str, object]) -> Dict[str, List[str]]:
        options = question.get("options") or {}
        result = {str(k): [] for k in options}
        if not isinstance(value, dict):
            return result
        for key in result:
            result[key] = self._list(value.get(key), 5)
        return result
