"""证据表提取与 reasoning 引用辅助工具。

提取器直接处理发送给 API 的实际消息，因此不会改变检索、Prompt、答案或 Token
用量；它只把 Prompt 中的证据标记转换为结构化审计表。
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, Iterable, List, Mapping, Sequence


_EVIDENCE_HEADER = re.compile(
    r"(?m)^\[(?P<prefix>E?)(?P<number>\d+)\]\s+"
    r"(?P<header>[^\n]*)"
)


def _metadata(header: str) -> Dict[str, object]:
    def value(*names: str) -> str:
        pattern = r"(?:^|\s)(?:" + "|".join(map(re.escape, names)) + r")=([^\s]+)"
        match = re.search(pattern, header)
        return match.group(1).strip() if match else ""

    score_text = value("score")
    try:
        score: float | None = float(score_text)
    except ValueError:
        score = None
    return {
        "domain": value("domain"),
        "doc_id": value("doc_id", "doc"),
        "page_id": value("page"),
        "score": score,
    }


def extract_evidence_table(
    messages: Sequence[Mapping[str, Any]],
) -> List[Dict[str, object]]:
    """从实际 API 消息载荷中提取证据区块。"""
    text = "\n".join(str(message.get("content", "")) for message in messages)
    matches = list(_EVIDENCE_HEADER.finditer(text))
    rows: List[Dict[str, object]] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        excerpt = text[start:end].strip()
        # 如果这是本次调用中的最后一个编号证据块，
        # 则在下一个主要 prompt 区块前停止。
        excerpt = re.split(
            r"\n(?=(?:题目|选项|输出|任务要求|首轮理由|API生成程序|安全执行结果)[:：])",
            excerpt,
            maxsplit=1,
        )[0].strip()
        number = int(match.group("number"))
        source_prefix = match.group("prefix")
        source_label = f"{source_prefix}{number}" if source_prefix else str(number)
        prompt_reference = f"[{source_label}]"
        canonical_label = f"E{number}"
        header = match.group("header").strip()
        rows.append(
            {
                "evidence_id": number,
                "label": canonical_label,
                "source_label": source_label,
                "prompt_reference": prompt_reference,
                "header": header,
                **_metadata(header),
                "excerpt": excerpt,
                "excerpt_chars": len(excerpt),
                "excerpt_sha256": hashlib.sha256(
                    excerpt.encode("utf-8")
                ).hexdigest(),
            }
        )
    return rows


def infer_option_from_messages(
    messages: Sequence[Mapping[str, Any]],
) -> str:
    text = "\n".join(str(message.get("content", "")) for message in messages)
    match = re.search(r"(?:待判断选项|选项)\s*([A-D])\s*[：:]", text)
    return match.group(1) if match else ""


def reasoning_evidence_references(reasoning: str) -> List[str]:
    """返回 API reasoning 摘要引用的规范 E 标签。"""
    values: List[int] = []
    for pattern in (
        r"(?<![A-Za-z0-9])E(\d+)(?!\d)",
        r"(?:证据|依据)\s*\[(\d+)\]",
        r"(?<![A-Za-z0-9])\[(\d+)\]",
    ):
        values.extend(int(item) for item in re.findall(pattern, reasoning))
    seen: set[int] = set()
    return [
        f"E{value}"
        for value in values
        if not (value in seen or seen.add(value))
    ]


def table_fingerprint(rows: Iterable[Mapping[str, object]]) -> str:
    stable = "\n".join(
        "|".join(
            [
                str(row.get("label", "")),
                str(row.get("doc_id", "")),
                str(row.get("page_id", "")),
                str(row.get("excerpt_sha256", "")),
            ]
        )
        for row in rows
    )
    return hashlib.sha256(stable.encode("utf-8")).hexdigest()
