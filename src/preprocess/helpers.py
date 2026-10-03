"""预处理流水线使用的小型无依赖辅助函数。"""

from __future__ import annotations

from html import unescape
from html.parser import HTMLParser
from pathlib import Path
import re
from typing import Iterable


def read_text_with_fallback(
    file_path: Path,
    encodings: Iterable[str] = ("utf-8-sig", "utf-8", "gb18030"),
) -> str:
    last_error: UnicodeDecodeError | None = None
    for encoding in encodings:
        try:
            return Path(file_path).read_text(encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    return Path(file_path).read_text(encoding="utf-8")


class _HTMLTextExtractor(HTMLParser):
    _skip_tags = {"script", "style", "noscript"}
    _block_tags = {
        "article", "br", "div", "h1", "h2", "h3", "h4", "h5", "h6",
        "li", "p", "section", "table", "td", "th", "tr",
    }

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        tag = tag.lower()
        if tag in self._skip_tags:
            self._skip_depth += 1
        elif tag in self._block_tags:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self._skip_tags and self._skip_depth:
            self._skip_depth -= 1
        elif tag in self._block_tags:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            text = unescape(data).strip()
            if text:
                self._parts.append(text)

    def text(self) -> str:
        raw = " ".join(self._parts)
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in raw.splitlines()]
        return "\n".join(line for line in lines if line)


def html_to_text(html: str) -> str:
    parser = _HTMLTextExtractor()
    parser.feed(html)
    parser.close()
    return parser.text()


def canonical_doc_id(doc_id: str) -> str:
    match = re.match(r"(strict_v3_\d{3})", str(doc_id))
    return match.group(1) if match else str(doc_id)
