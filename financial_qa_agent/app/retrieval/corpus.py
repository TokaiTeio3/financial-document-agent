from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass
from pathlib import Path

from app.retrieval.bm25 import BM25Index, query_anchors
from app.schemas import Domain, SearchHit


_PAGE = re.compile(r"<!--\s*page:\s*(\d+)", re.IGNORECASE)
_HEADING = re.compile(r"^#{1,4}\s+(.+)$", re.MULTILINE)
_COMPANY_NAME = re.compile(r"([\u3400-\u9fff]{2,30}?(?:股份有限公司|集团有限公司|有限责任公司|有限公司))")
_GENERIC_HEADINGS = {"目录", "重要提示", "致股东", "董事长致辞", "公司简介", "中国建筑精神"}


@dataclass(slots=True)
class Chunk:
    domain: Domain
    path: Path
    document_id: str
    chunk_id: str
    title: str
    page: int | None
    text: str


def _clean(text: str) -> str:
    text = re.sub(r"\[BOILERPLATE\]\s*", "", text)
    text = re.sub(r"\*\*==>.*?<==\*\*", "", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def chunk_file(
    path: Path,
    domain: Domain,
    chunk_size: int,
    overlap: int,
    page_overlap: int = 240,
) -> list[Chunk]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    headings = [item.strip("* ") for item in _HEADING.findall(raw)]
    company_match = _COMPANY_NAME.search(raw[:12000])
    title = company_match.group(1) if company_match else next(
        (
            item for item in headings
            if not re.fullmatch(r"Page\s+\d+", item, re.IGNORECASE)
            and item not in _GENERIC_HEADINGS
            and len(item) >= 4
        ),
        path.stem,
    )
    parts = [_clean(part) for part in re.split(r"(?=<!--\s*page:\s*\d+)", raw)]
    parts = [part for part in parts if part]
    chunks: list[Chunk] = []
    serial = 0
    step = max(1, chunk_size - overlap)
    for page_index, part in enumerate(parts):
        page_match = _PAGE.search(part)
        page = int(page_match.group(1)) if page_match else None
        previous_tail = parts[page_index - 1][-page_overlap:] if page_overlap and page_index > 0 else ""
        next_head = parts[page_index + 1][:page_overlap] if page_overlap and page_index + 1 < len(parts) else ""
        expanded = "\n".join(filter(None, (
            f"[上一页重叠]\n{previous_tail}" if previous_tail else "",
            f"[当前页]\n{part}",
            f"[下一页重叠]\n{next_head}" if next_head else "",
        )))
        cursor = 0
        while cursor < len(expanded):
            piece = _clean(expanded[cursor : cursor + chunk_size])
            if piece:
                chunks.append(Chunk(domain, path, path.stem, f"{path.stem}:{serial}", title, page, piece))
                serial += 1
            cursor += step
    return chunks


def chunk_to_record(chunk: Chunk, data_root: Path) -> dict:
    return {
        "domain": chunk.domain,
        "source": str(chunk.path.relative_to(data_root)).replace("\\", "/"),
        "document_id": chunk.document_id,
        "chunk_id": chunk.chunk_id,
        "title": chunk.title,
        "page": chunk.page,
        "text": chunk.text,
    }


def load_chunk_records(index_file: Path, data_root: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    with index_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            chunks.append(Chunk(
                domain=item["domain"],
                path=data_root / item["source"],
                document_id=item["document_id"],
                chunk_id=item["chunk_id"],
                title=item["title"],
                page=item.get("page"),
                text=item["text"],
            ))
    return chunks


class DomainCorpus:
    def __init__(
        self,
        domain: Domain,
        root: Path,
        chunk_size: int,
        overlap: int,
        page_overlap: int = 240,
        index_file: Path | None = None,
    ):
        self.domain = domain
        self.root = root
        if index_file and index_file.is_file():
            self.chunks = load_chunk_records(index_file, root.parent)
        else:
            self.chunks: list[Chunk] = []
            for path in sorted(root.glob("*.md")):
                self.chunks.extend(chunk_file(path, domain, chunk_size, overlap, page_overlap))
        self.index = BM25Index((chunk.text, f"{chunk.title} {chunk.document_id}") for chunk in self.chunks)

        # First-stage document index prevents many high-scoring chunks from one document
        # from crowding all other named entities out of the final context.
        grouped: dict[str, list[Chunk]] = {}
        for chunk in self.chunks:
            grouped.setdefault(chunk.document_id, []).append(chunk)
        self.document_ids = list(grouped)
        self.document_index = BM25Index(
            (
                "\n".join(chunk.text[:1200] for chunk in chunks[:4]),
                f"{chunks[0].title} {document_id}",
            )
            for document_id, chunks in grouped.items()
        )

    def search(self, query: str, top_k: int = 6, document_ids: list[str] | None = None) -> list[SearchHit]:
        allowed = {item.lower() for item in document_ids or []}
        anchors = query_anchors(query)
        document_scores = dict(zip(self.document_ids, self.document_index.scores(query), strict=True))
        combined_scores = [
            chunk_score + 1.8 * document_scores.get(chunk.document_id, 0.0)
            for chunk, chunk_score in zip(self.chunks, self.index.scores(query), strict=True)
        ]
        ranked = sorted(enumerate(combined_scores), key=lambda pair: pair[1], reverse=True)
        hits: list[SearchHit] = []
        hits_per_document: dict[str, int] = {}
        per_document_limit = top_k if allowed else max(2, min(3, top_k // 3 or 2))
        for index, score in ranked:
            chunk = self.chunks[index]
            if allowed and chunk.document_id.lower() not in allowed:
                continue
            if hits_per_document.get(chunk.document_id, 0) >= per_document_limit:
                continue
            if score <= 0 and hits:
                continue
            matched = [anchor for anchor in anchors if anchor in chunk.text.lower() or anchor in chunk.title.lower()]
            hits.append(SearchHit(
                domain=self.domain,
                document_id=chunk.document_id,
                source=str(chunk.path.relative_to(self.root.parent)).replace("\\", "/"),
                chunk_id=chunk.chunk_id,
                page=chunk.page,
                score=round(score, 4),
                matched_anchors=matched[:8],
                content=chunk.text[:1400],
            ))
            hits_per_document[chunk.document_id] = hits_per_document.get(chunk.document_id, 0) + 1
            if len(hits) >= top_k:
                break
        return hits


class CorpusRegistry:
    DOMAINS: tuple[Domain, ...] = (
        "financial_reports", "financial_contracts", "insurance", "regulatory", "research"
    )

    def __init__(
        self,
        data_root: Path,
        chunk_size: int = 900,
        overlap: int = 120,
        page_overlap: int = 240,
        index_root: Path | None = None,
    ):
        self.data_root = data_root
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.page_overlap = page_overlap
        self.index_root = index_root
        self.use_prebuilt_index = False
        if index_root and (index_root / "manifest.json").is_file():
            try:
                manifest = json.loads((index_root / "manifest.json").read_text(encoding="utf-8"))
                self.use_prebuilt_index = (
                    manifest.get("version") == 1
                    and manifest.get("data_root") == str(data_root.resolve())
                    and manifest.get("chunk_size") == chunk_size
                    and manifest.get("chunk_overlap") == overlap
                    and manifest.get("page_overlap") == page_overlap
                )
            except (OSError, json.JSONDecodeError):
                self.use_prebuilt_index = False
        self._corpora: dict[Domain, DomainCorpus] = {}
        self._lock = threading.Lock()

    def get(self, domain: Domain) -> DomainCorpus:
        if domain not in self._corpora:
            with self._lock:
                if domain not in self._corpora:
                    root = self.data_root / domain
                    if not root.is_dir():
                        raise FileNotFoundError(f"领域数据目录不存在: {root}")
                    index_file = (
                        self.index_root / f"{domain}.jsonl"
                        if self.index_root and self.use_prebuilt_index
                        else None
                    )
                    self._corpora[domain] = DomainCorpus(
                        domain, root, self.chunk_size, self.overlap, self.page_overlap, index_file
                    )
        return self._corpora[domain]

    def stats(self) -> dict[str, dict[str, int]]:
        return {
            domain: {"documents": len(list((self.data_root / domain).glob("*.md"))), "chunks": len(corpus.chunks)}
            for domain in self.DOMAINS
            if (corpus := self._corpora.get(domain)) is not None
        }
