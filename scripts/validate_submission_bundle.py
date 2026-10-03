"""对已装配的竞赛提交目录执行快速失败审计。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path


REQUIRED_COLUMNS = [
    "qid",
    "answer_1",
    "answer_2",
    "answer_3",
    "answer_4",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "reasoning",
]
REQUIRED_PATHS = [
    "answer.csv",
    "evidence.json",
    "logs/api_calls.jsonl",
    "logs/evidence_tables.json",
    "logs/run_metrics.json",
    "logs/manifest.json",
    "processed_data",
    "data/raw_dataset/raw",
    "data/raw_dataset/questions",
    "src/finqa_agent",
    "src/preprocess",
    "scripts/build_evidence.py",
    "config/config.ultra.yaml",
    "generate_answer.py",
    "generate_answer.sh",
    "preprocess.sh",
    "reproduce.sh",
    "run_preprocess.py",
    "run_reproduce.py",
    "requirements.txt",
    "README.md",
    "docs/金融长文本Agent的动态记忆压缩与高效问答挑战.北宇治吹奏部.pdf",
]
FORBIDDEN_PARTS = {
    ".env",
    "ground_truth",
    "__pycache__",
    ".pytest_cache",
    "release_artifacts",
}
SECRET_PATTERN = re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{20,}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_answer(path: Path) -> dict[str, int]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or list(rows[0]) != REQUIRED_COLUMNS:
        raise ValueError("answer.csv columns do not match the official contract")
    summary = [row for row in rows if row["qid"] == "summary"]
    questions = [row for row in rows if row["qid"] != "summary"]
    if len(questions) != 100 or len(summary) != 1:
        raise ValueError("answer.csv must contain 100 questions and one summary row")
    if len({row["qid"] for row in questions}) != 100:
        raise ValueError("answer.csv contains duplicate qids")
    if any(not row["reasoning"].strip() for row in questions):
        raise ValueError("every question must contain non-empty reasoning")
    totals = {
        key: sum(int(row[key]) for row in questions)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }
    for key, value in totals.items():
        if int(summary[0][key]) != value:
            raise ValueError(f"summary {key} does not equal per-question sum")
    if totals["total_tokens"] != totals["prompt_tokens"] + totals["completion_tokens"]:
        raise ValueError("total_tokens must equal prompt_tokens + completion_tokens")
    return totals


def validate_bundle(root: Path) -> dict[str, object]:
    missing = [relative for relative in REQUIRED_PATHS if not (root / relative).exists()]
    if missing:
        raise ValueError(f"missing required paths: {', '.join(missing)}")
    files = [path for path in root.rglob("*") if path.is_file()]
    forbidden = [
        str(path.relative_to(root))
        for path in files
        if FORBIDDEN_PARTS.intersection(path.relative_to(root).parts)
    ]
    if forbidden:
        raise ValueError(f"forbidden files found: {', '.join(forbidden[:10])}")
    for path in files:
        if path.stat().st_size > 2_000_000:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if SECRET_PATTERN.search(text):
            raise ValueError(f"possible secret marker found in {path.relative_to(root)}")
    totals = validate_answer(root / "answer.csv")
    raw_files = list((root / "data/raw_dataset/raw").rglob("*"))
    if not any(path.is_file() for path in raw_files):
        raise ValueError("official raw input directory is empty")
    manifest_path = root / "SUBMISSION_MANIFEST.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for row in manifest.get("files", []):
            path = root / row["path"]
            if not path.is_file() or sha256(path) != row["sha256"]:
                raise ValueError(f"manifest hash mismatch: {row['path']}")
    return {
        "status": "ready",
        "file_count": len(files),
        "size_bytes": sum(path.stat().st_size for path in files),
        "raw_file_count": sum(path.is_file() for path in raw_files),
        **totals,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args()
    report = validate_bundle(args.bundle.resolve())
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
