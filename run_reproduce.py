"""使用既有 processed_data 复现 B 榜性能并生成 answer.csv。"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from scripts.build_evidence import build_evidence


def run(command: list[str]) -> None:
    print("RUN", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, check=True)


def first_directory(root: Path, candidates: tuple[str, ...]) -> Path | None:
    for relative in candidates:
        path = root / relative
        if path.is_dir():
            return path
    return None


def resolve_performance_inputs(
    *,
    input_root: Path | None,
    questions_dir: Path | None,
    processed_data: Path | None,
) -> tuple[Path, Path]:
    if input_root is not None:
        root = input_root.resolve()
        if not root.is_dir():
            raise SystemExit(f"performance input root is not a directory: {root}")
        questions_dir = questions_dir or first_directory(
            root,
            (
                "data/raw_dataset/questions",
                "raw_dataset/questions",
                "questions",
            ),
        )
        processed_data = processed_data or first_directory(root, ("processed_data",))

    if questions_dir is None:
        raise SystemExit("provide --input or --questions-dir")
    if processed_data is None:
        raise SystemExit("provide --input or --processed-data")

    questions = questions_dir.resolve()
    processed = processed_data.resolve()
    if not questions.is_dir():
        raise SystemExit(f"questions directory not found: {questions}")
    if not processed.is_dir():
        raise SystemExit(f"processed_data directory not found: {processed}")
    return questions, processed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        help=(
            "性能复现包根目录；应包含 processed_data 和 "
            "data/raw_dataset/questions。可改用两个显式目录参数。"
        ),
    )
    parser.add_argument("--questions-dir", type=Path)
    parser.add_argument("--processed-data", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", default=Path("config/config.ultra.yaml"), type=Path)
    parser.add_argument("--workers", default=4, type=int)
    args = parser.parse_args()

    questions, processed = resolve_performance_inputs(
        input_root=args.input,
        questions_dir=args.questions_dir,
        processed_data=args.processed_data,
    )
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    runtime_index = output / "runtime_index"
    runtime_index.mkdir()

    run_dir = output / "run"
    run(
        [
            sys.executable,
            "-m",
            "src.finqa_agent.run",
            "--split",
            "B",
            "--profile",
            "reproduction",
            "--config",
            str(args.config.resolve()),
            "--questions-dir",
            str(questions),
            "--docs-dir",
            str(processed),
            "--runtime-index-dir",
            str(runtime_index),
            "--run-dir",
            str(run_dir),
            "--submission-output",
            "answer.csv",
            "--workers",
            str(args.workers),
        ]
    )

    shutil.copy2(run_dir / "answer.csv", output / "answer.csv")
    evidence = build_evidence(run_dir / "run.log.jsonl")
    (output / "evidence.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    shutil.copy2(run_dir / "run.log.jsonl", output / "api_calls.jsonl")
    shutil.copy2(run_dir / "evidence_tables.json", output / "evidence_tables.json")
    shutil.copy2(run_dir / "run_metrics.json", output / "run_metrics.json")
    print(f"answer: {output / 'answer.csv'}")
    print(f"evidence: {output / 'evidence.json'}")
    print(f"API log: {output / 'api_calls.jsonl'}")
    print(f"Evidence tables: {output / 'evidence_tables.json'}")
    print(f"Run metrics: {output / 'run_metrics.json'}")


if __name__ == "__main__":
    main()
