"""从官方原始输入复现预处理并生成 processed_data。"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(command: list[str]) -> None:
    print("RUN", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, check=True)


def resolve_dataset_root(input_path: Path) -> Path:
    for candidate in (
        input_path / "data" / "raw_dataset",
        input_path / "raw_dataset",
        input_path,
    ):
        if (candidate / "raw").is_dir():
            return candidate
    return input_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", default=4, type=int)
    args = parser.parse_args()

    input_path = args.input.resolve()
    output = args.output.resolve()
    if not input_path.exists():
        raise SystemExit(f"official input does not exist: {input_path}")
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "-m",
        "src.preprocess.prepare_data",
    ]
    if input_path.is_file():
        extracted = output / "raw_dataset"
        command.extend(["--zip", str(input_path), "--extract-to", str(extracted)])
    else:
        dataset_root = resolve_dataset_root(input_path)
        if not (dataset_root / "raw").is_dir():
            raise SystemExit(f"raw document directory not found: {dataset_root / 'raw'}")
        command.extend(
            ["--skip-unzip", "--extract-to", str(dataset_root)]
        )

    command.extend(
        [
            "--output-dir",
            str(output / "parsed_pages"),
            "--normalized-output-dir",
            str(output / "normalized_pages"),
            "--merged-output-dir",
            str(output / "processed_data"),
            "--report",
            str(output / "preprocess_report.json"),
            "--workers",
            str(args.workers),
        ]
    )
    run(command)
    print(f"processed_data: {output / 'processed_data'}")
    print(f"report: {output / 'preprocess_report.json'}")


if __name__ == "__main__":
    main()
