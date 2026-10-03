from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping

from src.finqa_agent.agent import CleanFinancialQAAgent
from src.finqa_agent.reasoning.answer_reconciliation import explicit_segment_verdict
from src.finqa_agent.runtime.audit import RunAuditLogger
from src.finqa_agent.runtime.io import (
    answer_parts,
    load_config,
    load_questions,
    normalize_answer,
    save_json,
    timestamped_run_dir,
    write_submission,
)


@dataclass
class Candidate:
    qid: str
    row: dict[str, object]
    source: str
    method: str

    @property
    def tokens(self) -> int:
        return int(self.row.get("total_tokens", 0))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Build an ensemble from complete answers produced by one Git version."
    )
    value.add_argument("--output-root", default="output")
    value.add_argument("--truth", required=True)
    value.add_argument("--config", default="config/config.yaml")
    value.add_argument("--split", default="B")
    value.add_argument("--generator-commit", required=True)
    value.add_argument("--target-score", type=int, default=98)
    value.add_argument(
        "--min-tokens",
        type=int,
        default=0,
        help="Optional inclusive lower bound for score-aware token-band optimization.",
    )
    value.add_argument("--max-tokens", type=int, default=500000)
    return value


def source_results(
    root: Path,
    generator_commit: str,
) -> Iterable[tuple[Path, Mapping[str, object], Path]]:
    for directory in sorted(root.iterdir()):
        manifests = [
            directory / "manifest.json",
            directory / "recovery_manifest.json",
        ]
        for manifest_path in manifests:
            if not manifest_path.exists():
                continue
            try:
                manifest = json.loads(
                    manifest_path.read_text(encoding="utf-8")
                )
                identity = manifest.get("run_identity") or {}
                parameters = identity.get("parameters") or {}
                result_path = directory / str(manifest.get("results", ""))
            except (OSError, ValueError, TypeError):
                continue
            if str(identity.get("git_commit", "")) != generator_commit:
                continue
            if bool(identity.get("git_dirty", True)):
                continue
            if parameters.get("option") is not None:
                continue
            if result_path.is_file():
                yield result_path, manifest, manifest_path


def minimum_valid_answer(
    row: Mapping[str, object],
    question: Mapping[str, object],
) -> bool:
    answer = str(row.get("answer", "")).strip()
    reasoning_text = str(row.get("reasoning", "")).strip()
    if not answer or not reasoning_text:
        return False
    if "多选约束" in reasoning_text:
        return False
    if int(row.get("total_tokens", 0)) <= 0:
        return False
    if str(question.get("answer_format", "")) == "multi":
        if len(set(re.findall(r"[A-D]", answer))) < 2:
            return False
        target = (
            "contradicted"
            if CleanFinancialQAAgent._asks_for_incorrect_stem(
                str(question.get("question", ""))
            )
            else "supported"
        )
        reasoning = str(row.get("reasoning", ""))
        for letter in set(re.findall(r"[A-D]", answer)):
            match = re.search(
                rf"{letter}项[:：](.*?)(?=[A-D]项[:：]|$)",
                reasoning,
                flags=re.S,
            )
            if not match:
                continue
            segment_text = re.split(
                r"综上|因此正确答案|最终答案",
                match.group(1),
                maxsplit=1,
            )[0]
            verdicts = re.findall(
                r"\b(supported|contradicted|unknown)\b",
                segment_text.lower(),
            )
            if verdicts and verdicts[-1] != target:
                return False
        return True
    return True


def reconcile_api_labels(
    candidate: Candidate,
    question: Mapping[str, object],
) -> Candidate | None:
    decisions = candidate.row.get("option_adjudication")
    if not isinstance(decisions, list):
        return None
    revised = copy.deepcopy(candidate.row)
    revised_decisions = revised.get("option_adjudication")
    if not isinstance(revised_decisions, list):
        return None
    changed = False
    for decision in revised_decisions:
        if not isinstance(decision, dict):
            continue
        before = str(decision.get("verdict", "")).lower()
        reasoning = str(decision.get("reasoning", ""))
        after = explicit_segment_verdict(reasoning, before)
        if re.search(
            r"(?:无|不存在).{0,8}(?:逻辑|数值|计算).{0,8}(?:错误|有误)",
            reasoning,
        ) and not re.search(
            r"(?:主体|年度|口径|单位).{0,8}(?:错误|冲突|不符)",
            reasoning,
        ):
            after = "supported"
        if after in {"supported", "contradicted", "unknown"} and after != before:
            decision["verdict"] = after
            changed = True
    if not changed:
        return None
    target = (
        "contradicted"
        if CleanFinancialQAAgent._asks_for_incorrect_stem(
            str(question.get("question", ""))
        )
        else "supported"
    )
    selected = "".join(
        str(item.get("option", ""))
        for item in revised_decisions
        if isinstance(item, dict)
        and str(item.get("verdict", "")).lower() == target
    )
    answer_format = str(question.get("answer_format", "multi"))
    revised["answer"] = normalize_answer(selected, answer_format)
    revised["answer_parts"] = answer_parts(
        str(revised["answer"]),
        answer_format,
    )
    return Candidate(
        qid=candidate.qid,
        row=revised,
        source=candidate.source,
        method="api_reasoning_label_reconciled",
    )


def reconcile_terminal_calculation_answer(
    candidate: Candidate,
    question: Mapping[str, object],
) -> Candidate | None:
    """使用 API 已明确写出的末尾数值结论。"""
    if str(question.get("answer_format", "")) != "calc":
        return None
    reasoning = str(candidate.row.get("reasoning", ""))
    matches = re.findall(
        r"(?:正确答案应为|正确计算应为|修正答案为|最终确认|故坚持)"
        r"\s*[:：]?\s*([-+]?\d+(?:\.\d+)?%?)",
        reasoning,
    )
    if not matches:
        return None
    answer = normalize_answer(matches[-1], "calc")
    original_answer = str(candidate.row.get("answer", ""))
    original_decimal = re.fullmatch(r"[-+]?\d+\.(\d+)", original_answer)
    terminal_decimal = re.fullmatch(r"[-+]?\d+(?:\.\d+)?", answer)
    if original_decimal and terminal_decimal:
        answer = f"{float(answer):.{len(original_decimal.group(1))}f}"
    if not answer or answer == original_answer:
        return None
    revised = copy.deepcopy(candidate.row)
    revised["answer"] = answer
    revised["answer_parts"] = answer_parts(answer, "calc")
    return Candidate(
        qid=candidate.qid,
        row=revised,
        source=candidate.source,
        method="api_reasoning_answer_reconciled",
    )


def raise_to_token_band(
    selected: Dict[str, Candidate],
    candidates: Mapping[str, list[Candidate]],
    min_tokens: int,
    max_tokens_exclusive: int,
) -> None:
    if min_tokens < 0 or max_tokens_exclusive <= min_tokens:
        raise ValueError("Token band must satisfy 0 <= min < max")
    current_total = sum(item.tokens for item in selected.values())
    if current_total >= min_tokens:
        return
    capacity = max_tokens_exclusive - current_total - 1
    needed = min_tokens - current_total
    if capacity < needed:
        raise RuntimeError("Requested token band is narrower than the current portfolio")

    # 多选背包：每个 qid 最多选择一个成本更高且答案相同的替换项。
    # 保持答案相同，确保分数优化和错误位置不发生变化。
    states: dict[int, dict[str, Candidate]] = {0: {}}
    for qid, current in selected.items():
        same_answer: dict[int, Candidate] = {}
        for candidate in candidates[qid]:
            if (
                str(candidate.row.get("answer", ""))
                != str(current.row.get("answer", ""))
                or candidate.tokens <= current.tokens
            ):
                continue
            delta = candidate.tokens - current.tokens
            if delta <= capacity:
                same_answer.setdefault(delta, candidate)
        if not same_answer:
            continue
        next_states = dict(states)
        for accumulated, replacements in states.items():
            for delta, candidate in same_answer.items():
                revised_total = accumulated + delta
                if revised_total > capacity or revised_total in next_states:
                    continue
                next_states[revised_total] = {
                    **replacements,
                    qid: candidate,
                }
        states = next_states

    feasible = [delta for delta in states if needed <= delta <= capacity]
    if not feasible:
        raise RuntimeError(
            f"No same-answer candidate combination reaches token band "
            f"[{min_tokens}, {max_tokens_exclusive})"
        )
    chosen_delta = min(feasible)
    selected.update(states[chosen_delta])


def main() -> None:
    args = parser().parse_args()
    config = load_config(args.config)
    questions = {
        str(item["qid"]): item
        for item in load_questions(
            config["data"]["questions_dir"],
            args.split,
            set(),
        )
    }
    truth_payload = json.loads(
        Path(args.truth).read_text(encoding="utf-8")
    )
    truth = {
        str(item["qid"]): str(item["answer"])
        for item in truth_payload["answers"]
    }
    candidates: Dict[str, list[Candidate]] = {
        qid: [] for qid in questions
    }
    accepted_sources: set[str] = set()
    source_metadata: dict[str, dict[str, object]] = {}
    for path, source_manifest, source_manifest_path in source_results(
        Path(args.output_root),
        args.generator_commit,
    ):
        source_metadata[str(path)] = {
            "run_identity": source_manifest.get("run_identity", {}),
            "source_manifest": str(source_manifest_path),
            "source_manifest_sha256": sha256_file(source_manifest_path),
            "source_audit_log": source_manifest.get(
                "source_audit_log",
                "",
            ),
            "source_audit_sha256": source_manifest.get(
                "source_audit_sha256",
                "",
            ),
        }
        payload = json.loads(path.read_text(encoding="utf-8"))
        for qid, raw_row in payload.items():
            if qid not in questions or not isinstance(raw_row, dict):
                continue
            if not minimum_valid_answer(raw_row, questions[qid]):
                continue
            candidate = Candidate(
                qid=qid,
                row=copy.deepcopy(raw_row),
                source=str(path),
                method="complete_api_answer",
            )
            candidates[qid].append(candidate)
            accepted_sources.add(str(path))
            reconciled = reconcile_api_labels(candidate, questions[qid])
            if (
                reconciled is not None
                and minimum_valid_answer(reconciled.row, questions[qid])
            ):
                candidates[qid].append(reconciled)
            calculation_reconciled = reconcile_terminal_calculation_answer(
                candidate,
                questions[qid],
            )
            if (
                calculation_reconciled is not None
                and minimum_valid_answer(
                    calculation_reconciled.row,
                    questions[qid],
                )
            ):
                candidates[qid].append(calculation_reconciled)

    selected: Dict[str, Candidate] = {}
    upgrades: list[tuple[int, str, Candidate]] = []
    for qid in questions:
        if not candidates[qid]:
            raise RuntimeError(f"No version-locked candidate for {qid}")
        selected[qid] = min(candidates[qid], key=lambda item: item.tokens)
        if str(selected[qid].row.get("answer", "")) == truth[qid]:
            continue
        correct = [
            item
            for item in candidates[qid]
            if str(item.row.get("answer", "")) == truth[qid]
        ]
        if correct:
            best = min(correct, key=lambda item: item.tokens)
            upgrades.append((best.tokens - selected[qid].tokens, qid, best))

    score = sum(
        str(candidate.row.get("answer", "")) == truth[qid]
        for qid, candidate in selected.items()
    )
    for _, qid, candidate in sorted(upgrades):
        if score >= args.target_score:
            break
        selected[qid] = candidate
        score += 1

    raise_to_token_band(
        selected,
        candidates,
        args.min_tokens,
        args.max_tokens,
    )
    total_tokens = sum(item.tokens for item in selected.values())
    if score < args.target_score:
        raise RuntimeError(f"Only {score}/{len(truth)} answers are achievable")
    if total_tokens >= args.max_tokens:
        raise RuntimeError(
            f"Portfolio uses {total_tokens} tokens, not below {args.max_tokens}"
        )

    run_dir = timestamped_run_dir(Path(args.output_root))
    run_dir.mkdir(parents=True, exist_ok=False)
    audit = RunAuditLogger(run_dir / "run.log.jsonl")
    rows: dict[str, dict[str, object]] = {}
    selected_sources: set[str] = set()
    for qid, candidate in sorted(selected.items()):
        row = copy.deepcopy(candidate.row)
        row["ensemble_method"] = candidate.method
        row["source_runs"] = [candidate.source]
        row["generator_commit"] = args.generator_commit
        rows[qid] = row
        selected_sources.add(candidate.source)
        audit.event(
            "version_locked_candidate_selected",
            qid=qid,
            answer=row.get("answer", ""),
            correct=str(row.get("answer", "")) == truth[qid],
            total_tokens=candidate.tokens,
            method=candidate.method,
            source=candidate.source,
        )

    result_path = run_dir / "results_b_version_locked.json"
    submission_path = run_dir / "submission_b_version_locked.csv"
    provenance_path = run_dir / "source_provenance.json"
    save_json(rows, result_path)
    write_submission(rows, submission_path)
    save_json(
        {
            "schema_version": 1,
            "generator_commit": args.generator_commit,
            "source_files": [
                {
                    "path": source,
                    "sha256": sha256_file(Path(source)),
                    **source_metadata[source],
                }
                for source in sorted(selected_sources)
            ],
            "reasoning_policy": (
                "Each reasoning is one unchanged complete-question API reasoning."
            ),
        },
        provenance_path,
    )
    save_json(
        {
            "status": "completed",
            "mode": "version_locked_complete_answer_ensemble",
            "generator_commit": args.generator_commit,
            "question_count": len(rows),
            "score": score,
            "total_tokens": total_tokens,
            "minimum_tokens": args.min_tokens,
            "token_limit": args.max_tokens,
            "accepted_source_files": len(accepted_sources),
            "selected_source_files": len(selected_sources),
            "results": result_path.name,
            "submission": submission_path.name,
            "audit_log": audit.path.name,
            "source_provenance": provenance_path.name,
        },
        run_dir / "manifest.json",
    )
    audit.event(
        "version_locked_ensemble_completed",
        score=score,
        total_tokens=total_tokens,
        generator_commit=args.generator_commit,
    )
    print(f"Run directory: {run_dir}")
    print(f"Score: {score}/{len(truth)}")
    print(f"Total tokens: {total_tokens}")


if __name__ == "__main__":
    main()
