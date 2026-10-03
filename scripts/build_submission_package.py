"""按白名单装配可审计的竞赛提交目录。"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from scripts.build_evidence import build_evidence
from scripts.validate_submission_bundle import validate_bundle
from src.finqa_agent.runtime.observability import export_observability_artifacts


ROOT_FILES = (
    "README.md",
    "requirements.txt",
    "generate_answer.py",
    "generate_answer.sh",
    "preprocess.sh",
    "reproduce.sh",
    "run_preprocess.py",
    "run_reproduce.py",
)
SOURCE_TREES = (
    "src/finqa_agent",
    "src/preprocess",
)
PRODUCTION_SCRIPTS = (
    "scripts/build_evidence.py",
    "scripts/build_submission_package.py",
    "scripts/config_dashboard.py",
    "scripts/extract_evidence_tables.py",
    "scripts/validate_submission_readiness.py",
    "scripts/validate_submission_bundle.py",
)
METHOD_DOCUMENT = "金融长文本Agent的动态记忆压缩与高效问答挑战.北宇治吹奏部.pdf"


def copy_path(source: Path, destination: Path) -> None:
    if source.is_dir():
        shutil.copytree(
            source,
            destination,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"),
        )
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def resolve_dataset(path: Path) -> Path:
    if not path.is_dir():
        raise ValueError(
            "--official-input must be an extracted directory for final packaging"
        )
    nested = path / "raw_dataset"
    root = nested if nested.is_dir() else path
    if not (root / "raw").is_dir():
        raise ValueError(f"official raw document directory not found under: {root}")
    return root


def write_manifest(root: Path, metadata: dict[str, object]) -> None:
    files = []
    bookkeeping = {"SUBMISSION_MANIFEST.json", "SUBMISSION_AUDIT.json"}
    for path in sorted(
        item
        for item in root.rglob("*")
        if item.is_file() and item.name not in bookkeeping
    ):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size": path.stat().st_size,
                "sha256": digest,
            }
        )
    payload = {
        "schema_version": 1,
        "purpose": "Financial-QA cold-start reproduction submission",
        **metadata,
        "files": files,
    }
    (root / "SUBMISSION_MANIFEST.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--official-input", required=True, type=Path)
    parser.add_argument(
        "--reference-run",
        type=Path,
        default=Path("output/20260726_222454_701518"),
    )
    parser.add_argument(
        "--processed-data",
        type=Path,
        default=Path("data/processed_pymupdf4llm_merged_normalized"),
    )
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    dataset = resolve_dataset(args.official_input.resolve())
    reference = args.reference_run.resolve()

    for relative in ROOT_FILES + SOURCE_TREES + PRODUCTION_SCRIPTS:
        source = project / relative
        if not source.exists():
            raise SystemExit(f"required source is missing: {source}")
        copy_path(source, output / relative)
    method_document = project / METHOD_DOCUMENT
    if not method_document.is_file():
        method_document = project / "docs" / METHOD_DOCUMENT
    if not method_document.is_file():
        raise SystemExit(f"required method PDF is missing: {METHOD_DOCUMENT}")
    copy_path(method_document, output / "docs" / METHOD_DOCUMENT)
    copy_path(project / "config/config.ultra.yaml", output / "config/config.ultra.yaml")
    copy_path(args.processed_data.resolve(), output / "processed_data")
    copy_path(dataset / "raw", output / "data/raw_dataset/raw")
    copy_path(project / "data/raw_dataset/questions", output / "data/raw_dataset/questions")

    submission = reference / "submission_b_reproduction.csv"
    results = reference / "results_b_reproduction.json"
    log = reference / "run.log.jsonl"
    manifest = reference / "manifest.json"
    for path in (submission, results, log, manifest):
        if not path.is_file():
            raise SystemExit(f"reference-run artifact is missing: {path}")
    copy_path(submission, output / "answer.csv")
    copy_path(results, output / "logs/results_b_reproduction.json")
    copy_path(log, output / "logs/api_calls.jsonl")
    copy_path(manifest, output / "logs/manifest.json")
    evidence = build_evidence(log)
    (output / "evidence.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    observed = export_observability_artifacts(log, output / "logs")
    backfill = reference / "evidence_tables_backfill.json"
    if backfill.is_file():
        copy_path(backfill, output / "logs/evidence_tables.json")
        tables = json.loads(backfill.read_text(encoding="utf-8"))
        metrics_path = output / "logs/run_metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics["evidence_table_question_count"] = len(tables)
        metrics["evidence_table_count"] = sum(len(value) for value in tables.values())
        metrics["evidence_row_count"] = sum(
            len(table.get("rows", []))
            for value in tables.values()
            for table in value
        )
        metrics["evidence_table_source"] = "backfilled_from_exact_api_request_messages"
        metrics_path.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    metadata = {
        "reference_run": reference.name,
        "reference_manifest": json.loads(manifest.read_text(encoding="utf-8")),
        "observability": observed,
    }
    write_manifest(output, metadata)
    report = validate_bundle(output)
    (output / "SUBMISSION_AUDIT.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"staged submission: {output}")


if __name__ == "__main__":
    main()
