"""Audit tracked publication files without printing potential secret values.

This is a conservative local check, not a replacement for a full secret scanner.
It intentionally excludes local ignored caches and scans only `git ls-files`.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_PARTS = {
    ".env", ".venv", "venv", "__pycache__", "ground_truth", "output",
    "runtime_index", "submission_materials", "release_artifacts", "docx_work",
}
FORBIDDEN_SUFFIXES = {".pdf", ".docx", ".zip", ".csv", ".jsonl", ".log", ".pem", ".key", ".pfx"}
PATTERNS = {
    "api-key-like": re.compile(r"(?<![\w-])sk-[A-Za-z0-9_-]{20,}"),
    "github-token-like": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})\b"),
    "private-key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "credential-url": re.compile(r"https?://[^\s/:]+:[^\s/@]+@"),
}


def audit(root: Path = ROOT) -> dict:
    result = subprocess.run(
        ["git", "ls-files", "-z"], cwd=root, check=True, capture_output=True,
    )
    names = [name for name in result.stdout.decode("utf-8").split("\0") if name]
    if not names:
        raise ValueError("No tracked files. Stage the curated release before auditing.")
    findings = []
    for name in names:
        path = root / name
        relative = Path(name)
        if path.is_symlink() or not path.is_file():
            findings.append({"path": name, "rule": "non-regular-file"})
            continue
        if FORBIDDEN_PARTS.intersection(relative.parts) or relative.suffix.lower() in FORBIDDEN_SUFFIXES:
            findings.append({"path": name, "rule": "private-artifact"})
        if relative.parts[0] in {"data", "processed_data"}:
            findings.append({"path": name, "rule": "non-fixture-data"})
        if path.stat().st_size > 1_000_000:
            findings.append({"path": name, "rule": "large-file"})
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            findings.append({"path": name, "rule": "unexpected-binary"})
            continue
        for rule, pattern in PATTERNS.items():
            if pattern.search(content):
                findings.append({"path": name, "rule": rule})
    return {"tracked_files": len(names), "findings": findings, "passed": not findings}


if __name__ == "__main__":
    report = audit()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["passed"] else 1)
