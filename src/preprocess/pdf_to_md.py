#!/usr/bin/env python3
"""
PDF 批量转 Markdown 脚本（基于 pymupdf4llm）

用法示例：
    python -m src.preprocess.pdf_to_md \
        --input-dir design-draft/data/raw_dataset/raw \
        --output-dir data/processed_pymupdf4llm \
        --workers 4

输入目录结构（与 raw_dataset 一致）：
    input-dir/
    ├── insurance/
    │   ├── 1.pdf
    │   ├── 2.pdf
    │   └── ...
    ├── regulatory/
    └── ...

输出目录结构（与已有解析数据对齐）：
    output-dir/
    ├── insurance/
    │   ├── 1/
    │   │   ├── page_0001.md
    │   │   ├── page_0002.md
    │   │   └── ...
    │   ├── 2/
    │   └── ...
    ├── regulatory/
    └── ...
"""

import argparse
import json
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.preprocess.helpers import canonical_doc_id, html_to_text, read_text_with_fallback
from src.preprocess.text_normalizer import normalize_document_pages


def find_pdf_files(input_dir: Path) -> list[Path]:
    """大小写无关地查找 PDF，覆盖 .pdf/.PDF 等后缀。"""
    return sorted(
        path
        for path in Path(input_dir).rglob("*")
        if path.is_file() and path.suffix.lower() == ".pdf"
    )


def _pdf_domain(pdf_path: Path) -> str:
    """监管附件 PDF 归入 regulatory 域，避免和题目 domain 脱节。"""
    parent = pdf_path.parent
    if parent.name == "attachments" and parent.parent.name == "regulatory":
        return "regulatory"
    return parent.name


def process_pdf(pdf_path: Path, output_dir: Path, normalized_output_dir: Path | None = None) -> dict:
    """处理单个 PDF，按页输出 Markdown。"""
    import pymupdf4llm

    domain = _pdf_domain(pdf_path)
    doc_id = pdf_path.stem

    out_dir = output_dir / domain / doc_id
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        chunks = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True)
        raw_pages = []
        for i, chunk in enumerate(chunks):
            page_num = i + 1
            md_path = out_dir / f"page_{page_num:04d}.md"
            text = chunk["text"]
            md_path.write_text(text, encoding="utf-8")
            raw_pages.append((md_path.name, text))

        if normalized_output_dir is not None:
            normalized_dir = normalized_output_dir / domain / doc_id
            normalized_dir.mkdir(parents=True, exist_ok=True)
            normalized = normalize_document_pages(raw_pages)
            for page_name, text in normalized.pages.items():
                (normalized_dir / page_name).write_text(text, encoding="utf-8")

        return {
            "pdf": str(pdf_path),
            "pages": len(chunks),
            "output": str(out_dir),
            "status": "ok",
        }
    except Exception as e:
        return {
            "pdf": str(pdf_path),
            "pages": 0,
            "output": str(out_dir),
            "status": "error",
            "error": str(e),
        }


def _write_single_page_doc(
    source_path: Path,
    output_dir: Path,
    domain: str,
    doc_id: str,
    text: str,
    normalized_output_dir: Path | None = None,
) -> dict:
    out_dir = output_dir / domain / doc_id
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "page_0001.md"
    md_path.write_text(text.strip() + "\n", encoding="utf-8")
    if normalized_output_dir is not None:
        normalized_dir = normalized_output_dir / domain / doc_id
        normalized_dir.mkdir(parents=True, exist_ok=True)
        normalized = normalize_document_pages([(md_path.name, text.strip() + "\n")])
        (normalized_dir / md_path.name).write_text(
            normalized.pages[md_path.name],
            encoding="utf-8",
        )
    return {
        "source": str(source_path),
        "pages": 1,
        "output": str(out_dir),
        "status": "ok",
    }


def process_text_file(
    file_path: Path,
    output_dir: Path,
    domain: str = "regulatory",
    normalized_output_dir: Path | None = None,
) -> dict:
    """处理监管 TXT 文件，写成单页 Markdown。"""
    try:
        text = read_text_with_fallback(file_path)
        doc_id = canonical_doc_id(file_path.stem)
        return _write_single_page_doc(file_path, output_dir, domain, doc_id, text, normalized_output_dir)
    except Exception as e:
        return {
            "source": str(file_path),
            "pages": 0,
            "output": "",
            "status": "error",
            "error": str(e),
        }


def process_html_file(
    file_path: Path,
    output_dir: Path,
    domain: str = "regulatory",
    normalized_output_dir: Path | None = None,
) -> dict:
    """处理监管 HTML 文件，抽取文本后写成单页 Markdown。"""
    try:
        html = read_text_with_fallback(file_path)
        text = html_to_text(html)
        return _write_single_page_doc(file_path, output_dir, domain, file_path.stem, text, normalized_output_dir)
    except Exception as e:
        return {
            "source": str(file_path),
            "pages": 0,
            "output": "",
            "status": "error",
            "error": str(e),
        }


def convert_regulatory_text_sources(
    raw_dir: Path,
    output_dir: Path,
    normalized_output_dir: Path | None = None,
) -> list[dict]:
    """将监管 TXT/HTML 原始材料转换到 processed_pymupdf4llm/regulatory。"""
    raw_dir = Path(raw_dir)
    regulatory_dir = raw_dir if raw_dir.name == "regulatory" else raw_dir / "regulatory"
    if not regulatory_dir.exists():
        return []

    results = []
    txt_dir = regulatory_dir / "txt"
    html_dir = regulatory_dir / "html"

    if txt_dir.exists():
        for file_path in sorted(txt_dir.glob("*.txt")):
            results.append(process_text_file(file_path, output_dir, normalized_output_dir=normalized_output_dir))

    if html_dir.exists():
        for file_path in sorted(html_dir.glob("*.html")):
            results.append(process_html_file(file_path, output_dir, normalized_output_dir=normalized_output_dir))

    return results


def main():
    parser = argparse.ArgumentParser(description="PDF 批量转 Markdown")
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="输入 PDF 根目录（含各领域的子目录）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="输出 Markdown 根目录",
    )
    parser.add_argument(
        "--normalized-output-dir",
        type=str,
        default=None,
        help="Optional retrieval-normalized Markdown output directory; raw output is still preserved.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="并发处理线程数（默认 4）",
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="可选：将处理结果写入 JSON 文件",
    )
    parser.add_argument(
        "--include-regulatory-text",
        action="store_true",
        help="同时转换监管 TXT/HTML 原始材料",
    )
    parser.add_argument(
        "--skip-pdf",
        action="store_true",
        help="跳过 PDF 转换，仅执行其他指定转换",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    normalized_output_dir = Path(args.normalized_output_dir) if args.normalized_output_dir else None

    if not input_dir.exists():
        print(f"错误: 输入目录不存在: {input_dir}", file=sys.stderr)
        return

    results = []

    if not args.skip_pdf:
        pdfs = find_pdf_files(input_dir)
        print(f"发现 {len(pdfs)} 个 PDF 文件")

        if not pdfs:
            print("没有找到 PDF 文件，请检查输入目录。", file=sys.stderr)
        else:
            ok_count = 0
            err_count = 0

            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                future_to_pdf = {
                    executor.submit(process_pdf, pdf, output_dir, normalized_output_dir): pdf
                    for pdf in pdfs
                }
                for future in as_completed(future_to_pdf):
                    result = future.result()
                    results.append(result)
                    if result["status"] == "ok":
                        ok_count += 1
                        print(
                            f"[OK] {result['pdf']} → {result['pages']} 页 → {result['output']}"
                        )
                    else:
                        err_count += 1
                        print(
                            f"[ERR] {result['pdf']}: {result.get('error', 'unknown')}",
                            file=sys.stderr,
                        )

            print(
                f"\nPDF 转换完成: {ok_count}/{len(pdfs)} 成功, {err_count}/{len(pdfs)} 失败"
            )

    if args.include_regulatory_text:
        text_results = convert_regulatory_text_sources(input_dir, output_dir, normalized_output_dir)
        results.extend(text_results)
        ok_count = sum(1 for r in text_results if r["status"] == "ok")
        err_count = len(text_results) - ok_count
        print(
            f"监管 TXT/HTML 转换完成: {ok_count}/{len(text_results)} 成功, {err_count}/{len(text_results)} 失败"
        )

    if args.json:
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(results, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"结果已保存: {json_path}")


if __name__ == "__main__":
    main()
