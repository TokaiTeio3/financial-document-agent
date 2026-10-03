"""对 PDF 解析得到的 Markdown 执行损失感知文本规范化。

规范化器生成便于检索的派生文本，并作为流水线中的增量步骤：调用方应保留
pymupdf4llm 原始输出，并将规范化文本写入独立目录。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Dict, Iterable, List, Sequence, Tuple


_CJK = r"\u4e00-\u9fff"
_SENTENCE_END = tuple("。！？；;：:")
_DOT_LEADER_RE = re.compile(
    r"^(?P<title>.+?)\s*(?:\.|·|。|．|…|\u2026){6,}\s*(?P<page>\d{1,4})\s*$"
)


@dataclass
class NormalizationStats:
    toc_lines: int = 0
    boilerplate_lines: int = 0
    space_fixes: int = 0
    soft_line_joins: int = 0
    pages: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {
            "pages": self.pages,
            "toc_lines": self.toc_lines,
            "boilerplate_lines": self.boilerplate_lines,
            "space_fixes": self.space_fixes,
            "soft_line_joins": self.soft_line_joins,
        }


@dataclass
class NormalizedDocument:
    pages: Dict[str, str]
    stats: NormalizationStats = field(default_factory=NormalizationStats)
    boilerplate: List[str] = field(default_factory=list)


def normalize_document_pages(pages: Sequence[Tuple[str, str]]) -> NormalizedDocument:
    """规范化单份文档的全部页面，并识别重复样板文本。

    参数：
        pages：由 ``(page_name, raw_markdown)`` 组成的序列。

    返回：
        以 page_name 为页面文本键的 NormalizedDocument。
    """

    stats = NormalizationStats(pages=len(pages))
    boilerplate = _detect_boilerplate([text for _, text in pages])
    normalized: Dict[str, str] = {}
    for name, text in pages:
        normalized[name] = normalize_page_text(text, boilerplate, stats)
    return NormalizedDocument(
        pages=normalized,
        stats=stats,
        boilerplate=sorted(boilerplate),
    )


def normalize_page_text(
    text: str,
    boilerplate_lines: Iterable[str] | None = None,
    stats: NormalizationStats | None = None,
) -> str:
    """规范化单个 Markdown 页面以供检索。

    函数保持表格和图片占位符不变，重写目录点引导符，移除换行产生的无效空格，
    并把重复页眉页脚标记为已省略的样板文本。
    """

    stats = stats or NormalizationStats(pages=1)
    boilerplate = set(boilerplate_lines or [])
    logical_lines: List[str] = []

    for raw_line in text.splitlines():
        line = _normalize_inline_spaces(raw_line.rstrip(), stats).strip()
        if not line:
            logical_lines.append("")
            continue

        toc_line = _normalize_toc_line(line)
        if toc_line != line:
            stats.toc_lines += 1
            logical_lines.append(toc_line)
            continue

        key = _boilerplate_key(line)
        if key and key in boilerplate:
            stats.boilerplate_lines += 1
            logical_lines.append(f"[BOILERPLATE] {line}")
            continue

        logical_lines.append(line)

    return _repair_soft_line_breaks(logical_lines, stats).strip() + "\n"


def _normalize_inline_spaces(line: str, stats: NormalizationStats) -> str:
    original = line
    line = line.replace("\u3000", " ")
    line = re.sub(r"[ \t]+", " ", line)
    line = re.sub(fr"(?<=[{_CJK}])\s+(?=[{_CJK}])", "", line)
    line = re.sub(fr"(?<=[{_CJK}])\s+(?=[，。！？；：、）】》])", "", line)
    line = re.sub(fr"(?<=[（【《])\s+(?=[{_CJK}A-Za-z0-9])", "", line)
    line = re.sub(fr"(?<=[{_CJK}])\s+(?=\d)", "", line)
    line = re.sub(fr"(?<=\d)\s+(?=[{_CJK}])", "", line)
    line = re.sub(fr"(?<=[A-Za-z])\s+(?=[{_CJK}])", "", line)
    line = re.sub(fr"(?<=[{_CJK}])\s+(?=[A-Za-z])", "", line)
    if line != original:
        stats.space_fixes += 1
    return line


def _normalize_toc_line(line: str) -> str:
    match = _DOT_LEADER_RE.match(line)
    if not match:
        return line
    title = match.group("title").strip()
    page = match.group("page").strip()
    if len(title) < 2:
        return line
    return f"目录项: {title} | 页码: {page}"


def _detect_boilerplate(texts: Sequence[str]) -> set[str]:
    if len(texts) < 3:
        return set()

    counts: Dict[str, int] = {}
    for text in texts:
        seen_on_page = set()
        for line in text.splitlines():
            key = _boilerplate_key(_normalize_inline_spaces(line.strip(), NormalizationStats()))
            if key:
                seen_on_page.add(key)
        for key in seen_on_page:
            counts[key] = counts.get(key, 0) + 1

    min_count = max(3, int(len(texts) * 0.3))
    return {key for key, count in counts.items() if count >= min_count}


def _boilerplate_key(line: str) -> str:
    line = line.strip()
    if not line:
        return ""
    if line.startswith("|") or line.startswith("#"):
        return ""
    if line.startswith("**==> picture"):
        return ""
    if line.startswith("<!-- page:") or line.startswith("目录项:"):
        return ""
    if re.fullmatch(r"(?:第\s*)?\d{1,4}\s*(?:页|/|／\s*\d{1,4})?", line):
        return line
    compact = re.sub(r"\s+", "", line)
    if not (4 <= len(compact) <= 80):
        return ""
    if _DOT_LEADER_RE.match(line):
        return ""
    # 重复出现的法条或表格标题可能有实际意义；
    # 不要把带强章节前缀的行标记为页眉页脚。
    if re.match(r"^(第[一二三四五六七八九十百千万\d]+[章节条]|表\s*\d|图\s*\d)", compact):
        return ""
    return compact


def _repair_soft_line_breaks(lines: Sequence[str], stats: NormalizationStats) -> str:
    repaired: List[str] = []
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if not line:
            repaired.append("")
            idx += 1
            continue

        current = line
        while idx + 1 < len(lines):
            nxt = lines[idx + 1]
            if not _should_join_soft_break(current, nxt):
                break
            joiner = "" if _cjk_boundary(current, nxt) else " "
            current = current.rstrip() + joiner + nxt.lstrip()
            stats.soft_line_joins += 1
            idx += 1
        repaired.append(current)
        idx += 1

    text = "\n".join(repaired)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _should_join_soft_break(left: str, right: str) -> bool:
    left = left.rstrip()
    right = right.lstrip()
    if not left or not right:
        return False
    protected_prefixes = ("#", "|", "-", "*", ">", "<!--", "[BOILERPLATE", "目录项:")
    if left.startswith(protected_prefixes) or right.startswith(protected_prefixes):
        return False
    if left.endswith(_SENTENCE_END):
        return False
    if re.match(r"^\d+[.)、]", right):
        return False
    if len(left) < 8 or len(right) < 4:
        return False
    return True


def _cjk_boundary(left: str, right: str) -> bool:
    return bool(re.search(fr"[{_CJK}]$", left) and re.search(fr"^[{_CJK}]", right))
