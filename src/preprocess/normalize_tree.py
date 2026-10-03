"""规范化分页 Markdown，并构建检索使用的合并文档。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

from .text_normalizer import normalize_document_pages


def _read_pages(doc_dir: Path) -> List[Tuple[str, str]]:
    return [
        (path.name, path.read_text(encoding="utf-8"))
        for path in sorted(doc_dir.glob("page_*.md"))
    ]


def normalize_tree(
    input_dir: Path,
    normalized_dir: Path,
    merged_dir: Path,
) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    for domain_dir in sorted(path for path in input_dir.iterdir() if path.is_dir()):
        for doc_dir in sorted(path for path in domain_dir.iterdir() if path.is_dir()):
            pages = _read_pages(doc_dir)
            if not pages:
                continue
            normalized = normalize_document_pages(pages)
            output_doc = normalized_dir / domain_dir.name / doc_dir.name
            output_doc.mkdir(parents=True, exist_ok=True)
            merged_parts: List[str] = []
            for page_name, text in normalized.pages.items():
                (output_doc / page_name).write_text(text, encoding="utf-8")
                page_number = int(page_name[5:9])
                merged_parts.append(
                    f"<!-- page: {page_number}; source: {page_name} -->\n\n"
                    f"## Page {page_number}\n\n{text.strip()}\n"
                )
            merged_domain = merged_dir / domain_dir.name
            merged_domain.mkdir(parents=True, exist_ok=True)
            merged_path = merged_domain / f"{doc_dir.name}.md"
            merged_path.write_text(
                "\n\n".join(merged_parts).strip() + "\n",
                encoding="utf-8",
            )
            results.append(
                {
                    "domain": domain_dir.name,
                    "doc_id": doc_dir.name,
                    "pages": len(pages),
                    "merged_output": str(merged_path),
                    "stats": normalized.stats.to_dict(),
                }
            )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize parsed pages and build merged retrieval documents."
    )
    parser.add_argument("--input-dir", default="data/processed_pymupdf4llm")
    parser.add_argument(
        "--normalized-dir",
        default="data/processed_pymupdf4llm_normalized",
    )
    parser.add_argument(
        "--merged-dir",
        default="data/processed_pymupdf4llm_merged_normalized",
    )
    parser.add_argument(
        "--report",
        default="output/preprocess/normalization_report.json",
    )
    args = parser.parse_args()
    results = normalize_tree(
        Path(args.input_dir),
        Path(args.normalized_dir),
        Path(args.merged_dir),
    )
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"normalized docs={len(results)}, "
        f"pages={sum(int(item['pages']) for item in results)}"
    )
    print(f"merged retrieval data: {args.merged_dir}")
    print(f"report: {report}")


if __name__ == "__main__":
    main()
