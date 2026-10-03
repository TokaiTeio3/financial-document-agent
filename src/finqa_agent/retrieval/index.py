from __future__ import annotations

import json
import math
import re
from collections import Counter
from difflib import SequenceMatcher
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from ..core.entities import (
    DOMAIN_TERMS,
    GENERIC_TERMS,
    compact,
    dedupe,
    extract_entities,
    repair_mojibake,
)

try:
    from opencc import OpenCC
except ImportError:  # pragma: no cover - 依赖由 requirements 安装
    OpenCC = None


_T2S = OpenCC("t2s") if OpenCC is not None else None


def canonical(text: str) -> str:
    value = compact(text)
    return _T2S.convert(value) if _T2S is not None else value


@dataclass
class Page:
    domain: str
    doc_id: str
    page_id: str
    text: str
    path: str
    aliases: List[str] = field(default_factory=list)


@dataclass
class SearchHit:
    page: Page
    score: float
    query: str

    def to_dict(self, chars: int = 500) -> Dict[str, object]:
        return {
            "domain": self.page.domain,
            "doc_id": self.page.doc_id,
            "page_id": self.page.page_id,
            "score": round(self.score, 4),
            "query": self.query,
            "text": self.page.text[:chars],
        }


class DocumentIndex:
    def __init__(self, docs_dir: str | Path, cache_dir: str | Path = "runtime_index") -> None:
        self.docs_dir = Path(docs_dir)
        self.cache_dir = Path(cache_dir)
        self.pages_by_domain: Dict[str, List[Page]] = {}
        self.doc_aliases: Dict[str, Dict[str, List[str]]] = {}
        self.doc_title_aliases: Dict[str, Dict[str, List[str]]] = {}
        self.doc_texts: Dict[str, Dict[str, str]] = {}
        self._token_cache: Dict[tuple[str, str, str], List[str]] = {}
        self._compact_cache: Dict[tuple[str, str, str], str] = {}

    def load(self) -> None:
        self.pages_by_domain.clear()
        self.doc_aliases.clear()
        self.doc_title_aliases.clear()
        self.doc_texts.clear()
        self._token_cache.clear()
        self._compact_cache.clear()
        for domain_dir in sorted(p for p in self.docs_dir.iterdir() if p.is_dir()):
            domain = domain_dir.name
            pages: List[Page] = []
            aliases: Dict[str, List[str]] = {}
            title_aliases: Dict[str, List[str]] = {}
            doc_texts: Dict[str, str] = {}
            for path in sorted(domain_dir.glob("*.md")):
                doc_id = path.stem
                text = repair_mojibake(
                    path.read_text(encoding="utf-8", errors="ignore")
                )
                doc_texts[doc_id] = text
                doc_titles = self._extract_title_aliases(text, domain)
                doc_aliases = self._extract_doc_aliases(text, domain, doc_id, doc_titles)
                aliases[doc_id] = doc_aliases
                title_aliases[doc_id] = doc_titles
                pages.extend(self._split_doc(domain, doc_id, path, text, doc_aliases))
            self.pages_by_domain[domain] = pages
            self.doc_aliases[domain] = aliases
            self.doc_title_aliases[domain] = title_aliases
            self.doc_texts[domain] = doc_texts
        self._persist_doc_index()

    def _persist_doc_index(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            domain: {
                doc_id: aliases
                for doc_id, aliases in sorted(doc_aliases.items())
            }
            for domain, doc_aliases in sorted(self.doc_aliases.items())
        }
        (self.cache_dir / "document_entity_index.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _extract_title_aliases(self, text: str, domain: str) -> List[str]:
        page_markers = list(re.finditer(r"(?mi)^<!--\s*page:\s*\d+(?:;.*?)?-->\s*$", text))
        first_page_end = page_markers[1].start() if len(page_markers) > 1 else min(len(text), 5000)
        head = text[:first_page_end]
        title_scope = text[:12000] if domain == "insurance" else head
        heading_aliases = [
            match.strip()
            for match in re.findall(r"(?m)^#{1,3}\s+([^\n]{2,100})$", title_scope)
            if not re.fullmatch(r"Page\s*\d+", match.strip(), flags=re.I)
        ]
        declared_short_names = re.findall(
            r"(?:股票|证券|公司)?简称\s*[:：]\s*[*` ]*([\u4e00-\u9fffA-Za-z0-9·]{2,20})",
            text[:5000],
        )
        declared_short_names = [
            re.split(r"(?:股票|证券)?代码", name, maxsplit=1)[0]
            for name in declared_short_names
        ]
        company_aliases = re.findall(
            r"[\u4e00-\u9fffA-Za-z0-9（）()·]{2,36}"
            r"(?:股份有限公司|有限责任公司|集团有限公司|控股有限公司|股份公司|有限公司)",
            text[:16000],
        )
        if domain == "financial_reports" and company_aliases:
            company_aliases = [alias for alias, _ in Counter(company_aliases).most_common(2)]
        # 保险产品名通常出现在通用标题下方。
        # 仅在保险领域把这些文档派生实体作为标题别名；
        # 报告正文实体对公司路由来说过宽。
        insurance_aliases = extract_entities(head, domain) if domain == "insurance" else []
        base_aliases = dedupe(
            [*declared_short_names, *heading_aliases, *company_aliases, *insurance_aliases],
            80,
        )
        derived_aliases: List[str] = []
        corporate_suffixes = (
            "股份有限公司",
            "有限责任公司",
            "集团有限公司",
            "控股有限公司",
            "股份公司",
            "有限公司",
        )
        for alias in base_aliases:
            compact_alias = compact(alias)
            for suffix in corporate_suffixes:
                suffix_match = re.search(rf"([\u4e00-\u9fffA-Za-z0-9·]{{2,40}}){re.escape(suffix)}", compact_alias)
                if suffix_match:
                    full_company = suffix_match.group(1) + suffix
                    derived_aliases.extend([full_company, suffix_match.group(1)])
            if domain == "insurance" and "保险" in compact_alias:
                product_alias = re.sub(
                    r"(?:专属商业)?(?:养老)?(?:年金|医疗|疾病|责任|两全|终身寿)?保险.*$",
                    "",
                    compact_alias,
                )
                if 3 <= len(product_alias) <= 24:
                    derived_aliases.append(product_alias)
        return dedupe([*derived_aliases, *base_aliases], 100)

    def _extract_doc_aliases(
        self,
        text: str,
        domain: str,
        doc_id: str,
        title_aliases: Sequence[str],
    ) -> List[str]:
        head = text[:12000]
        aliases = [*title_aliases, *extract_entities(head, domain)]
        if domain == "financial_reports":
            aliases.extend(re.findall(r"[\u4e00-\u9fff]{2,16}(?:股份有限公司|集团|银行)", head[:4000]))
        if domain == "insurance":
            aliases.extend(re.findall(r"[\u4e00-\u9fffA-Za-z0-9（）()·]{2,40}(?:保险|险|金生|添盈|e生保)", head[:6000]))
        return dedupe([doc_id, *aliases], 100)

    def _split_doc(self, domain: str, doc_id: str, path: Path, text: str, aliases: List[str]) -> List[Page]:
        page_markers = list(
            re.finditer(
                r"(?mi)^(?:#\s*page[_ -]*(\d+)\s*|<!--\s*page:\s*(\d+)(?:;.*?)?-->)\s*$",
                text,
            )
        )
        if page_markers:
            pages: List[Page] = []
            for i, marker in enumerate(page_markers):
                start = marker.end()
                end = page_markers[i + 1].start() if i + 1 < len(page_markers) else len(text)
                body = text[start:end].strip()
                if body:
                    page_number = marker.group(1) or marker.group(2)
                    pages.append(Page(domain, doc_id, f"page_{int(page_number):04d}", body, str(path), aliases))
            if pages:
                return pages
        chunks = []
        size = 2400
        overlap = 300
        pos = 0
        idx = 1
        while pos < len(text):
            chunk = text[pos: pos + size].strip()
            if chunk:
                chunks.append(Page(domain, doc_id, f"chunk_{idx:04d}", chunk, str(path), aliases))
                idx += 1
            pos += size - overlap
        return chunks

    def locate_docs(self, question: Dict[str, object], max_docs: int = 6) -> Dict[str, object]:
        domain = str(question.get("domain", ""))
        qtext = self.question_text(question)
        q_compact = compact(qtext)
        candidates = []
        q_entities = extract_entities(qtext, domain)
        lexical_terms = dedupe(
            [
                *q_entities,
                *re.findall(r"[\u4e00-\u9fff]{3,12}|[A-Za-z][A-Za-z0-9._%-]{2,20}", qtext),
            ],
            100,
        )
        lexical_terms = [
            term
            for term in lexical_terms
            if term not in DOMAIN_TERMS.get(domain, []) and len(compact(term)) >= 3
        ]
        for doc_id, aliases in self.doc_aliases.get(domain, {}).items():
            score = 0.0
            reasons: List[str] = []
            strong_aliases: List[str] = []
            q_compact_text = canonical(qtext)
            generic_aliases = {
                *GENERIC_TERMS,
                *DOMAIN_TERMS.get(domain, []),
                "年度报告",
                "年年度报告",
                "管理办法",
                "阅读指引",
                "保险条款",
                "募集说明书",
                "主营业务毛利率",
                "现金价值",
                "解除合同",
                "保险责任",
                "保险期间",
                "保险费",
                "投保人",
                "被保险人",
                "犹豫期",
            }
            for title_alias in self.doc_title_aliases.get(domain, {}).get(doc_id, []):
                ctitle = canonical(title_alias)
                semantic_title = re.sub(r"20\d{2}年?", "", ctitle)
                semantic_title = re.sub(r"[\dQ.*_\-\s]+", "", semantic_title)
                if (
                    len(ctitle) < 3
                    or len(semantic_title) < 2
                    or ctitle in generic_aliases
                    or semantic_title in generic_aliases
                ):
                    continue
                if ctitle in q_compact_text:
                    strong_aliases.append(title_alias)
                    continue
                block = SequenceMatcher(None, ctitle, q_compact_text, autojunk=False).find_longest_match()
                common = ctitle[block.a: block.a + block.size]
                common_core = re.sub(r"20\d{2}年?", "", common)
                common_core = re.sub(r"[\dQ.\-（）()【】\[\]\s]+", "", common_core)
                if (
                    domain in {"insurance", "financial_reports"}
                    and block.size >= (4 if domain == "insurance" else 3)
                    and len(common_core) >= (4 if domain == "insurance" else 3)
                    and common not in generic_aliases
                    and common_core not in generic_aliases
                ):
                    strong_aliases.append(common)
            for alias in aliases:
                if alias and canonical(alias) in q_compact_text:
                    is_date_like = bool(re.fullmatch(r"[\d年月日Q.\-]+", alias))
                    calias = compact(alias)
                    semantic_alias = re.sub(r"20\d{2}年?", "", calias)
                    semantic_alias = re.sub(r"[\dQ.*_\-\s]+", "", semantic_alias)
                    if is_date_like:
                        score += 2.0
                    else:
                        score += 80.0 if len(alias) >= 4 else 25.0
                    reasons.append(f"alias:{alias}")
            if strong_aliases:
                score += 120.0 * len(strong_aliases)
                reasons.append(f"title_match:{len(strong_aliases)}")
            for term in q_entities:
                if term in aliases:
                    score += 35.0
                    reasons.append(f"entity:{term}")
            raw_doc = self.doc_texts.get(domain, {}).get(doc_id, "")
            head = compact(raw_doc[:24000])
            full_compact = compact(raw_doc)
            for term in q_entities + DOMAIN_TERMS.get(domain, []):
                if term and compact(term) in head:
                    score += 8.0
            exact_body_matches = 0
            for term in lexical_terms:
                cterm = compact(term)
                if not cterm:
                    continue
                if cterm in head:
                    score += 16.0 + min(16.0, len(cterm))
                    exact_body_matches += 1
                elif len(cterm) >= 4 and cterm in full_compact:
                    score += 5.0 + min(8.0, len(cterm) / 2)
                    exact_body_matches += 1
            if exact_body_matches:
                reasons.append(f"body_terms:{exact_body_matches}")
            years = set(re.findall(r"20\d{2}", qtext))
            doc_years = set(re.findall(r"20\d{2}", doc_id))
            if years and doc_years and years.isdisjoint(doc_years):
                score -= 30.0
            if doc_id in q_compact:
                score += 100.0
            if score > 0:
                candidates.append(
                    {
                        "doc_id": doc_id,
                        "score": score,
                        "reasons": dedupe(reasons, 8),
                        "strong_alias": bool(strong_aliases),
                    }
                )
        candidates.sort(key=lambda x: x["score"], reverse=True)

        discovery_queries = [str(question.get("question", ""))]
        discovery_queries.extend(str(value) for value in (question.get("options") or {}).values())
        discovery_hits = self.search(
            domain,
            [],
            [query for query in discovery_queries if query],
            top_k=max(30, max_docs * 6),
        )
        lexical_doc_ids = dedupe([hit.page.doc_id for hit in discovery_hits], max_docs * 3)
        strong_doc_ids = [str(item["doc_id"]) for item in candidates if item.get("strong_alias")]
        question_years = set(re.findall(r"20\d{2}", qtext))
        if question_years:
            year_matched = [
                doc_id
                for doc_id in strong_doc_ids
                if not set(re.findall(r"20\d{2}", doc_id))
                or not question_years.isdisjoint(set(re.findall(r"20\d{2}", doc_id)))
            ]
            if year_matched:
                strong_doc_ids = year_matched
        if strong_doc_ids:
            inferred = dedupe(strong_doc_ids, max_docs)
        elif lexical_doc_ids:
            inferred = lexical_doc_ids[:max_docs]
        elif candidates:
            best = float(candidates[0]["score"])
            threshold = max(12.0, best * 0.18)
            inferred = [c["doc_id"] for c in candidates if float(c["score"]) >= threshold][:max_docs]
        else:
            inferred = []
        return {
            "domain": domain,
            "inferred_doc_ids": inferred,
            "candidates": candidates[:12],
            "method": "entity_lexical_doc_discovery",
            "locatable": bool(inferred),
        }

    def search(
        self,
        domain: str,
        doc_ids: Sequence[str],
        queries: Sequence[str],
        top_k: int = 20,
    ) -> List[SearchHit]:
        pages = self.pages_by_domain.get(domain, [])
        if doc_ids:
            allowed = {str(x) for x in doc_ids}
            pages = [p for p in pages if p.doc_id in allowed]
        if not pages:
            return []
        tokenized = [self._page_tokens(p) for p in pages]
        df: Dict[str, int] = {}
        for toks in tokenized:
            for tok in set(toks):
                df[tok] = df.get(tok, 0) + 1
        avgdl = sum(len(t) for t in tokenized) / max(1, len(tokenized))
        hits_by_query: List[List[SearchHit]] = []
        for query in dedupe(queries, 80):
            q_tokens = self._tokens(query)
            q_terms = [t for t in extract_entities(query, domain) + q_tokens if len(t) >= 2]
            query_hits: List[SearchHit] = []
            for page, toks in zip(pages, tokenized):
                score = self._bm25(q_tokens, toks, df, len(pages), avgdl)
                page_compact = self._page_compact(page)
                for term in q_terms:
                    cterm = compact(term)
                    if cterm and cterm in page_compact:
                        score += 4.0 + min(8.0, len(cterm) / 2)
                if score > 0:
                    query_hits.append(SearchHit(page, score, query))
            query_hits.sort(key=lambda h: h.score, reverse=True)
            if query_hits:
                floor = max(0.5, query_hits[0].score * 0.12)
                hits_by_query.append([hit for hit in query_hits if hit.score >= floor][:8])

        selected: List[SearchHit] = []
        seen: set[tuple[str, str]] = set()
        rank = 0
        while len(selected) < top_k and any(rank < len(group) for group in hits_by_query):
            for group in hits_by_query:
                if rank >= len(group):
                    continue
                hit = group[rank]
                key = (hit.page.doc_id, hit.page.page_id)
                if key not in seen:
                    seen.add(key)
                    selected.append(hit)
                    if len(selected) >= top_k:
                        break
            rank += 1
        return selected

    @staticmethod
    def _page_key(page: Page) -> tuple[str, str, str]:
        return page.domain, page.doc_id, page.page_id

    def _page_tokens(self, page: Page) -> List[str]:
        key = self._page_key(page)
        if key not in self._token_cache:
            self._token_cache[key] = self._tokens(page.text)
        return self._token_cache[key]

    def _page_compact(self, page: Page) -> str:
        key = self._page_key(page)
        if key not in self._compact_cache:
            self._compact_cache[key] = compact(page.text)
        return self._compact_cache[key]

    @staticmethod
    def question_text(question: Dict[str, object]) -> str:
        options = question.get("options") or {}
        option_text = "\n".join(f"{k}. {v}" for k, v in sorted(options.items()))
        return f"{question.get('question', '')}\n{option_text}"

    @staticmethod
    def _tokens(text: str) -> List[str]:
        tokens = re.findall(r"[A-Za-z0-9._%-]+|[\u4e00-\u9fff]{2,8}", str(text or ""))
        char_bigrams = [
            str(text)[i:i + 2]
            for i in range(max(0, len(str(text)) - 1))
            if re.match(r"[\u4e00-\u9fff]{2}", str(text)[i:i + 2])
        ]
        return [t.lower() for t in tokens + char_bigrams if t.strip()]

    @staticmethod
    def _bm25(query_tokens: Sequence[str], doc_tokens: Sequence[str], df: Dict[str, int], n_docs: int, avgdl: float) -> float:
        if not query_tokens or not doc_tokens:
            return 0.0
        freq: Dict[str, int] = {}
        for tok in doc_tokens:
            freq[tok] = freq.get(tok, 0) + 1
        k1 = 1.5
        b = 0.75
        dl = len(doc_tokens)
        score = 0.0
        for tok in query_tokens:
            f = freq.get(tok.lower(), 0)
            if not f:
                continue
            idf = math.log(1 + (n_docs - df.get(tok.lower(), 0) + 0.5) / (df.get(tok.lower(), 0) + 0.5))
            score += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / max(1.0, avgdl)))
        return score
