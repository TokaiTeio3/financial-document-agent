from __future__ import annotations

import argparse

from .runtime.io import load_env
from .runtime.run_pipeline import FinancialQARunPipeline, RunOptions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clean B-list Financial QA Agent")
    parser.add_argument("--split", default="B", help="Question split, default B")
    parser.add_argument(
        "--profile",
        choices=[
            "accuracy",
            "long",
            "fast",
            "submission",
            "compressed",
            "lean",
            "reproduction",
        ],
        default="accuracy",
        help="accuracy=diagnostic ceiling; submission=enforce final token budget",
    )
    parser.add_argument("--config", default="config/config.yaml", help="config path")
    parser.add_argument("--questions-dir", default=None, help="override questions directory")
    parser.add_argument("--docs-dir", default=None, help="override merged document directory")
    parser.add_argument("--runtime-index-dir", default=None, help="override disposable index directory")
    parser.add_argument("--output", default=None, help="rich JSON output path")
    parser.add_argument("--submission-output", default=None, help="submission CSV path; default uses current timestamp")
    parser.add_argument("--run-dir", default=None, help="exact run directory; default output/<current timestamp>")
    parser.add_argument("--limit", type=int, default=None, help="run first N questions only")
    parser.add_argument("--workers", type=int, default=4, help="parallel workers")
    parser.add_argument("--qid", action="append", default=[], help="specific qid, repeatable or comma separated")
    parser.add_argument(
        "--option",
        choices=["A", "B", "C", "D"],
        default=None,
        help="micro-retry one option only; useful after a joint low-confidence answer",
    )
    return parser


def parse_qids(values: list[str]) -> set[str]:
    return {item.strip() for value in values for item in value.split(",") if item.strip()}


def main() -> None:
    args = build_parser().parse_args()
    load_env()
    pipeline = FinancialQARunPipeline(
        RunOptions(
            split=args.split,
            profile=args.profile,
            config=args.config,
            questions_dir=args.questions_dir,
            docs_dir=args.docs_dir,
            runtime_index_dir=args.runtime_index_dir,
            output=args.output,
            submission_output=args.submission_output,
            run_dir=args.run_dir,
            limit=args.limit,
            workers=args.workers,
            qids=parse_qids(args.qid),
            option=args.option,
        )
    )
    pipeline.run()


if __name__ == "__main__":
    main()
