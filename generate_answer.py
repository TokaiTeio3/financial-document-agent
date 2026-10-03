"""从官方原始输入执行完整解题流程并生成最终 answer.csv。"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


FINAL_ARTIFACTS = (
    "answer.csv",
    "evidence.json",
    "api_calls.jsonl",
    "evidence_tables.json",
    "run_metrics.json",
)


def run(command: list[str]) -> None:
    print("RUN", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, check=True)


def resolve_dataset_root(input_path: Path) -> Path:
    for candidate in (
        input_path / "data" / "raw_dataset",
        input_path / "raw_dataset",
        input_path,
    ):
        if (candidate / "questions").is_dir():
            return candidate
    return input_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", default=Path("config/config.ultra.yaml"), type=Path)
    parser.add_argument("--workers", default=4, type=int)
    args = parser.parse_args()

    project = Path(__file__).resolve().parent
    input_path = args.input.resolve()
    output = args.output.resolve()
    if not input_path.exists():
        raise SystemExit(f"official input does not exist: {input_path}")
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    work = output / "work"
    preprocess_output = work / "preprocess"
    performance_output = work / "performance"

    run(
        [
            sys.executable,
            str(project / "run_preprocess.py"),
            "--input",
            str(input_path),
            "--output",
            str(preprocess_output),
            "--workers",
            str(args.workers),
        ]
    )

    if input_path.is_file():
        questions = preprocess_output / "raw_dataset" / "questions"
    else:
        questions = resolve_dataset_root(input_path) / "questions"
    processed = preprocess_output / "processed_data"
    if not questions.is_dir():
        raise SystemExit(f"questions directory not found after preprocessing: {questions}")
    if not processed.is_dir():
        raise SystemExit(f"processed_data not generated: {processed}")

    run(
        [
            sys.executable,
            str(project / "run_reproduce.py"),
            "--questions-dir",
            str(questions),
            "--processed-data",
            str(processed),
            "--output",
            str(performance_output),
            "--config",
            str(args.config.resolve()),
            "--workers",
            str(args.workers),
        ]
    )

    for name in FINAL_ARTIFACTS:
        source = performance_output / name
        if not source.is_file():
            raise SystemExit(f"performance artifact not generated: {source}")
        shutil.copy2(source, output / name)
    shutil.copytree(performance_output / "run", output / "run")
    shutil.copytree(performance_output / "runtime_index", output / "runtime_index")
    print(f"final answer: {output / 'answer.csv'}")


if __name__ == "__main__":
    main()
