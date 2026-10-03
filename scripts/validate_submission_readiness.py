from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.finqa_agent.runtime.io import load_config, load_questions


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Validate cold-start configuration and an optional completed result."
    )
    value.add_argument("--config", default="config/config.yaml")
    value.add_argument(
        "--profile",
        default="submission",
        help="Configuration profile to validate (default: submission).",
    )
    value.add_argument("--split", default="B")
    value.add_argument("--results", default=None)
    return value


def main() -> None:
    args = parser().parse_args()
    config = load_config(args.config)
    profiles = config.get("profiles", {})
    if args.profile not in profiles:
        raise RuntimeError(f"unknown profile: {args.profile}")
    profile = profiles[args.profile]
    agent = config.get("agent", {})
    data = config.get("data", {})
    questions = {
        str(question["qid"]): question
        for question in load_questions(data["questions_dir"], args.split, set())
    }

    memory_path = Path(
        str(data.get("prior_api_answer_memory", "runtime_index/prior_api_answer_memory.json"))
    )
    violations = []
    if bool(profile.get("use_prior_api_answer_memory", False)):
        violations.append(f"{args.profile}.use_prior_api_answer_memory must be false")
    if bool(agent.get("cross_round_memory", False)):
        violations.append("agent.cross_round_memory must be false")
    if bool(agent.get("runtime_entity_memory", False)):
        violations.append("agent.runtime_entity_memory must be false for the official run")
    if memory_path.exists():
        violations.append(f"cross-round answer memory exists: {memory_path}")

    total_tokens = 0
    missing: list[str] = []
    invalid: list[str] = []
    if args.results:
        result_path = Path(args.results)
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise RuntimeError("result JSON must be an object keyed by qid")
        missing = sorted(set(questions) - set(payload))
        invalid = sorted(
            qid
            for qid, row in payload.items()
            if qid in questions
            and (
                not isinstance(row, dict)
                or not str(row.get("answer", "")).strip()
                or not str(row.get("reasoning", "")).strip()
                or int(row.get("total_tokens", 0)) <= 0
            )
        )
        total_tokens = sum(
            int(row.get("total_tokens", 0))
            for row in payload.values()
            if isinstance(row, dict)
        )
        if missing:
            violations.append(f"missing qids: {','.join(missing)}")
        if invalid:
            violations.append(f"invalid results: {','.join(invalid)}")

    budget = int(profile.get("token_budget", 0))
    submission_limit = int(profile.get("submission_token_limit", 1000000))
    if args.results and submission_limit and total_tokens >= submission_limit:
        violations.append(
            f"result token total {total_tokens} must be below submission limit {submission_limit}"
        )

    print("official_mode=cold_start")
    print(f"profile={args.profile}")
    print(f"question_count={len(questions)}")
    print(f"answer_memory_exists={memory_path.exists()}")
    print(f"cross_round_memory={bool(agent.get('cross_round_memory', False))}")
    print(f"runtime_entity_memory={bool(agent.get('runtime_entity_memory', False))}")
    print(f"configured_token_budget={budget}")
    print(f"submission_token_limit={submission_limit}")
    if args.results:
        print(f"result_total_tokens={total_tokens}")
        print(f"coverage={len(questions) - len(missing)}/{len(questions)}")
        print(f"invalid_entries={len(invalid)}")
    if violations:
        raise RuntimeError("; ".join(violations))


if __name__ == "__main__":
    main()
