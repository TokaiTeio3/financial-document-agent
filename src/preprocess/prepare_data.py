#!/usr/bin/env python3
"""
数据准备自动化脚本

读者操作流程：
1. 从赛题页面下载 public_dataset_a.zip，放到项目根目录 data/ 下
2. 运行本脚本：python -m src.preprocess.prepare_data
3. 脚本自动完成：解压 ZIP → 提取 PDF → 用 pymupdf4llm 转为 Markdown

输出结构：
    data/
    ├── raw_dataset/           # 解压后的原始数据（含 questions/ 和 raw/）
    │   ├── questions/
    │   └── raw/
    │       ├── insurance/
    │       ├── regulatory/
    │       └── ...
    └── processed_pymupdf4llm/  # 解析后的 Markdown
        ├── insurance/
        ├── regulatory/
        └── ...
"""

import argparse
import zipfile
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


def unzip_dataset(zip_path: Path, extract_to: Path) -> None:
    """安全解压数据集 ZIP，拒绝越界路径并跳过过长文件名。"""
    if not zip_path.exists():
        print(f"错误: 找不到数据集压缩包: {zip_path}", file=sys.stderr)
        print("请从赛题页面下载 public_dataset_a.zip 并放到 data/ 目录下。", file=sys.stderr)
        return

    print(f"正在解压: {zip_path} → {extract_to}")
    extract_to.mkdir(parents=True, exist_ok=True)
    extraction_root = extract_to.resolve()

    skipped = 0
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            # 去掉 ZIP 根目录前缀（如 public_dataset_upload/）
            relative = member
            if "/" in relative:
                parts = relative.split("/", 1)
                if len(parts) == 2:
                    relative = parts[1]
                else:
                    continue
            if not relative:
                continue

            target = (extract_to / relative).resolve()
            if not target.is_relative_to(extraction_root):
                raise ValueError(f"ZIP member escapes extraction directory: {member}")
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                if member.endswith("/"):
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    with zf.open(member) as src, open(target, "wb") as dst:
                        dst.write(src.read())
            except OSError as e:
                if e.errno == 36:  # 文件名过长
                    skipped += 1
                    continue
                raise

    if skipped:
        print(f"解压完成，跳过 {skipped} 个文件名过长的文件（不影响 PDF 和题目）。")
    else:
        print("解压完成。")


def convert_all_pdfs(
    raw_dir: Path,
    output_dir: Path,
    workers: int = 4,
    normalized_output_dir: Path | None = None,
) -> None:
    """调用 pdf_to_md 批量转换所有 PDF。

    不设置固定的单文件超时：长募集说明书在 CPU 环境下可能需要数分钟，
    强行按时间跳过会造成不可复现的文档缺失。
    """
    from src.preprocess.pdf_to_md import find_pdf_files, process_pdf

    pdfs = find_pdf_files(raw_dir)
    if not pdfs:
        print(f"未在 {raw_dir} 下找到 PDF 文件，跳过转换。")
        return

    print(f"开始转换 {len(pdfs)} 个 PDF 文件（workers={workers}）...")

    ok_count = 0
    err_count = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(process_pdf, pdf, output_dir, normalized_output_dir): pdf
            for pdf in pdfs
        }
        for future in as_completed(futures):
            result = future.result()
            if result["status"] == "ok":
                ok_count += 1
                print(f"[OK] {result['pdf']} → {result['pages']} 页")
            else:
                err_count += 1
                print(
                    f"[ERR] {result['pdf']}: {result.get('error', 'unknown')}",
                    file=sys.stderr,
                )

    print(f"\n转换完成: {ok_count}/{len(pdfs)} 成功, {err_count}/{len(pdfs)} 失败")


def convert_regulatory_sources(
    raw_dir: Path,
    output_dir: Path,
    normalized_output_dir: Path | None = None,
) -> None:
    """补充转换监管 TXT/HTML 原始材料。"""
    from src.preprocess.pdf_to_md import convert_regulatory_text_sources

    results = convert_regulatory_text_sources(raw_dir, output_dir, normalized_output_dir)
    if not results:
        print(f"未在 {raw_dir} 下找到监管 TXT/HTML 文件，跳过转换。")
        return

    ok_count = sum(1 for r in results if r["status"] == "ok")
    err_count = len(results) - ok_count
    print(f"监管 TXT/HTML 转换完成: {ok_count}/{len(results)} 成功, {err_count}/{len(results)} 失败")
    for result in results:
        if result["status"] != "ok":
            print(
                f"[ERR] {result['source']}: {result.get('error', 'unknown')}",
                file=sys.stderr,
            )


def main():
    parser = argparse.ArgumentParser(description="数据准备自动化脚本")
    parser.add_argument(
        "--zip",
        type=str,
        default="data/public_dataset_a.zip",
        help="数据集压缩包路径（默认: data/public_dataset_a.zip）",
    )
    parser.add_argument(
        "--extract-to",
        type=str,
        default="data/raw_dataset",
        help="解压目标目录（默认: data/raw_dataset）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed_pymupdf4llm",
        help="Markdown 输出目录（默认: data/processed_pymupdf4llm）",
    )
    parser.add_argument(
        "--normalized-output-dir",
        type=str,
        default="data/processed_pymupdf4llm_normalized",
        help="规范化分页 Markdown 目录。",
    )
    parser.add_argument(
        "--merged-output-dir",
        type=str,
        default="data/processed_pymupdf4llm_merged_normalized",
        help="Agent 检索使用的规范化合并文档目录。",
    )
    parser.add_argument(
        "--report",
        type=str,
        default="output/preprocess/preprocess_report.json",
        help="预处理统计报告。",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="PDF 转换并发数（默认 4）",
    )
    parser.add_argument(
        "--skip-unzip",
        action="store_true",
        help="跳过解压步骤（如果已经解压过）",
    )
    parser.add_argument(
        "--skip-convert",
        action="store_true",
        help="跳过 PDF 转换步骤",
    )
    args = parser.parse_args()

    zip_path = Path(args.zip)
    extract_to = Path(args.extract_to)
    output_dir = Path(args.output_dir)
    normalized_output_dir = Path(args.normalized_output_dir) if args.normalized_output_dir else None
    merged_output_dir = Path(args.merged_output_dir)

    # 1. 解压
    if not args.skip_unzip:
        unzip_dataset(zip_path, extract_to)

    # 2. 转换 PDF
    if not args.skip_convert:
        raw_pdf_dir = extract_to / "raw"
        if raw_pdf_dir.exists():
            convert_all_pdfs(
                raw_pdf_dir,
                output_dir,
                workers=args.workers,
                normalized_output_dir=None,
            )
            convert_regulatory_sources(
                raw_pdf_dir,
                output_dir,
                normalized_output_dir=None,
            )
        else:
            # 有些压缩包的目录结构可能不同，直接搜全部 PDF
            convert_all_pdfs(
                extract_to,
                output_dir,
                workers=args.workers,
                normalized_output_dir=None,
            )
            convert_regulatory_sources(
                extract_to,
                output_dir,
                normalized_output_dir=None,
            )

        if normalized_output_dir is None:
            raise SystemExit("--normalized-output-dir cannot be empty in reproduction mode")
        from src.preprocess.normalize_tree import normalize_tree

        results = normalize_tree(output_dir, normalized_output_dir, merged_output_dir)
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        import json

        report_path.write_text(
            json.dumps(results, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            f"规范化完成: {len(results)} 份文档，"
            f"{sum(int(item['pages']) for item in results)} 页"
        )

    print("\n数据准备完成。")
    print(f"  原始数据: {extract_to}")
    print(f"  Markdown: {output_dir}")
    print(f"  检索数据: {merged_output_dir}")
    print(f"  处理报告: {args.report}")


if __name__ == "__main__":
    main()
