from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable


_LATIN_OR_NUMBER = re.compile(r"[A-Za-z]+(?:[-_.][A-Za-z0-9]+)*|\d+(?:\.\d+)?%?")
_CJK = re.compile(r"[\u3400-\u9fff]+")
_STOPWORDS = {
    "根据", "关于", "哪些", "下列", "其中", "什么", "如何", "是否", "以及", "进行",
    "公司", "本公司", "报告", "问题", "内容", "相关", "分别", "说明", "给出", "请问",
}


def tokenize(text: str) -> list[str]:
    """Tokenize Chinese without an embedding model or mandatory segmenter."""
    text = text.lower()
    tokens = _LATIN_OR_NUMBER.findall(text)
    for run in _CJK.findall(text):
        if len(run) <= 2:
            tokens.append(run)
        else:
            tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
            tokens.extend(run[i : i + 3] for i in range(len(run) - 2))
    return [token for token in tokens if token and token not in _STOPWORDS]


def query_anchors(question: str) -> list[str]:
    candidates: list[str] = []
    candidates.extend(_LATIN_OR_NUMBER.findall(question))
    for run in _CJK.findall(question):
        parts = re.split(r"(?:根据|关于|下列|哪些|什么|如何|是否|以及|的|中|和|与|较|请)", run)
        candidates.extend(part for part in parts if 2 <= len(part) <= 18)
        candidates.extend(run[i : i + 4] for i in range(max(0, len(run) - 3)))
    unique = sorted({item.lower() for item in candidates if item.strip()}, key=len, reverse=True)
    return unique[:24]


@dataclass(slots=True)
class BM25Document:
    text: str
    title: str
    tokens: list[str]


class BM25Index:
    def __init__(self, documents: Iterable[tuple[str, str]], k1: float = 1.5, b: float = 0.75):
        self.documents = [BM25Document(text, title, tokenize(f"{title} {text}")) for text, title in documents]
        self.k1 = k1
        self.b = b
        self.avgdl = sum(len(doc.tokens) for doc in self.documents) / max(1, len(self.documents))
        self.term_freqs = [Counter(doc.tokens) for doc in self.documents]
        document_frequency: Counter[str] = Counter()
        for doc in self.documents:
            document_frequency.update(set(doc.tokens))
        total = len(self.documents)
        self.idf = {
            term: math.log(1 + (total - freq + 0.5) / (freq + 0.5))
            for term, freq in document_frequency.items()
        }

    def scores(self, query: str) -> list[float]:
        query_tokens = tokenize(query)
        anchors = query_anchors(query)
        output: list[float] = []
        for doc, frequencies in zip(self.documents, self.term_freqs, strict=True):
            length = max(1, len(doc.tokens))
            score = 0.0
            for term in query_tokens:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                numerator = frequency * (self.k1 + 1)
                denominator = frequency + self.k1 * (1 - self.b + self.b * length / max(1, self.avgdl))
                score += self.idf.get(term, 0.0) * numerator / denominator

            lowered_text = doc.text.lower()
            lowered_title = doc.title.lower()
            for anchor in anchors:
                if anchor in lowered_title:
                    # Title/entity matches should dominate generic financial phrases in the body.
                    score += 25.0 + min(len(anchor), 12)
                elif anchor in lowered_text:
                    score += 0.8 + min(len(anchor), 12) * 0.12
            output.append(score)
        return output
