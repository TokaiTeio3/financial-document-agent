from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.config import Settings
from app.retrieval.corpus import CorpusRegistry, chunk_file, chunk_to_record


def main() -> None:
    parser = argparse.ArgumentParser(description="Build page-aware BM25 source chunks with bidirectional overlap.")
    parser.add_argument("--domain", action="append", choices=CorpusRegistry.DOMAINS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--page-overlap", type=int)
    args = parser.parse_args()

    settings = Settings.from_env()
    output = (args.output or settings.index_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    domains = args.domain or list(CorpusRegistry.DOMAINS)
    page_overlap = args.page_overlap if args.page_overlap is not None else settings.page_overlap
    counts: dict[str, int] = {}

    for domain in domains:
        target = output / f"{domain}.jsonl"
        temporary = output / f".{domain}.jsonl.tmp"
        count = 0
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for path in sorted((settings.data_root / domain).glob("*.md")):
                for chunk in chunk_file(
                    path,
                    domain,
                    settings.chunk_size,
                    settings.chunk_overlap,
                    page_overlap,
                ):
                    handle.write(json.dumps(chunk_to_record(chunk, settings.data_root), ensure_ascii=False) + "\n")
                    count += 1
        temporary.replace(target)
        counts[domain] = count
        print(f"{domain}: {count} chunks -> {target}")

    manifest = {
        "version": 1,
        "data_root": str(settings.data_root.resolve()),
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "page_overlap": page_overlap,
        "domains": counts,
        "overlap_strategy": "previous page tail + current page + next page head; sliding chunk overlap",
    }
    temporary_manifest = output / ".manifest.json.tmp"
    temporary_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary_manifest.replace(output / "manifest.json")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
