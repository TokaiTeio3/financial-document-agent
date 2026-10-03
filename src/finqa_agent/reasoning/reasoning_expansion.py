from __future__ import annotations

import hashlib
import re
from typing import Dict, List, Tuple


_CITATION_PATTERN = re.compile(r"(?<![A-Za-z])E(\d+)")
_PROMPT_BLOCK_PATTERN = re.compile(r"(?m)^\[E(\d+)\]\s+doc=")


def _bounded_prompt_block(block: str, max_chars: int) -> str:
    """返回 API Prompt 区块连续且便于阅读的前缀。"""
    value = block.rstrip()
    if len(value) <= max_chars:
        return value
    prefix = value[:max_chars]
    sentence_end = max(
        prefix.rfind(mark)
        for mark in ("。", "；", ";", "\n")
    )
    if sentence_end >= max(60, max_chars // 2):
        prefix = prefix[: sentence_end + 1]
    return prefix.rstrip()


def expand_reasoning_from_counted_prompt(
    api_reasoning: str,
    prompt: str,
    *,
    max_references: int = 4,
    max_chars_per_reference: int = 220,
) -> Tuple[str, Dict[str, object]]:
    """只使用已计入 Prompt Token 的文本追加引用证据。

    API reasoning 保持原样。每个追加片段都是实际发送给 API 的 Prompt 中的连续
    子串；本函数不做摘要、改写，也不注入文档或实体知识。
    """
    reasoning = str(api_reasoning).strip()
    prompt_text = str(prompt)
    cited_ids = list(
        dict.fromkeys(_CITATION_PATTERN.findall(reasoning))
    )[: max(0, int(max_references))]

    matches = list(_PROMPT_BLOCK_PATTERN.finditer(prompt_text))
    blocks: Dict[str, str] = {}
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(prompt_text)
        block = prompt_text[start:end]
        section_end = block.find("\n### ")
        if section_end >= 0:
            block = block[:section_end]
        blocks.setdefault(match.group(1), block)

    segments: List[str] = []
    segment_rows: List[Dict[str, object]] = []
    for evidence_id in cited_ids:
        block = blocks.get(evidence_id, "")
        if not block:
            continue
        segment = _bounded_prompt_block(
            block,
            max(80, int(max_chars_per_reference)),
        )
        if not segment or segment in reasoning:
            continue
        if segment not in prompt_text:
            raise RuntimeError("reasoning expansion segment is not prompt-exact")
        segments.append(segment)
        segment_rows.append(
            {
                "source": "counted_api_prompt",
                "evidence_id": int(evidence_id),
                "content": segment,
                "sha256": hashlib.sha256(segment.encode("utf-8")).hexdigest(),
            }
        )

    expanded = reasoning
    if segments:
        expanded = "\n".join([reasoning, *segments])
    provenance: Dict[str, object] = {
        "policy": "api_output_plus_exact_counted_prompt_segments",
        "api_reasoning": reasoning,
        "prompt_sha256": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
        "prompt_segments": segment_rows,
        "additional_api_tokens": 0,
    }
    return expanded, provenance
