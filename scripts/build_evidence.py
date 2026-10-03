"""根据不可变 JSONL 运行日志构建逐题证据链。"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable


EVIDENCE_EVENTS = {
    "question_started",
    "document_discovery_completed",
    "question_entities_extracted",
    "query_plan_completed",
    "retrieval_completed",
    "retrieval_repaired",
    "evidence_capsules_built",
    "evidence_table",
    "reasoning_level_assigned",
    "route_decision_made",
    "runtime_contract_validated",
    "prompt_routed",
    "api_request",
    "api_response",
    "api_error",
    "calculation_completed",
    "compact_calculation_answer_completed",
    "compact_joint_answer_completed",
    "compact_option_answer_completed",
    "primary_answer_completed",
    "verification_completed",
    "question_completed",
    "question_failed",
}


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def build_evidence(log_path: Path) -> Dict[str, Dict[str, Any]]:
    by_qid: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "events": [],
            "api_calls": [],
            "evidence_tables": [],
            "token_usage": {},
        }
    )
    for record in read_jsonl(log_path):
        qid = str(record.get("qid", "")).strip()
        event = str(record.get("event", ""))
        if not qid or event not in EVIDENCE_EVENTS:
            continue
        if event == "evidence_table":
            by_qid[qid]["evidence_tables"].append(
                {
                    "call_id": record.get("call_id"),
                    "route": record.get("route"),
                    "option": record.get("option", ""),
                    "table_id": record.get("table_id"),
                    "table_sha256": record.get("table_sha256"),
                    "rows": record.get("rows", []),
                }
            )
        elif event == "api_request":
            by_qid[qid]["api_calls"].append(
                {
                    "call_id": record.get("call_id"),
                    "route": record.get("route"),
                    "requested_at": record.get("timestamp"),
                    "model": record.get("model"),
                    "parameters": record.get("parameters"),
                    "messages": record.get("messages"),
                    "fallback": record.get("fallback", False),
                }
            )
        elif event in {"api_response", "api_error"}:
            call_id = record.get("call_id")
            target = next(
                (
                    item
                    for item in reversed(by_qid[qid]["api_calls"])
                    if item.get("call_id") == call_id
                ),
                None,
            )
            if target is None:
                target = {"call_id": call_id}
                by_qid[qid]["api_calls"].append(target)
            if event == "api_response":
                target.update(
                    {
                        "responded_at": record.get("timestamp"),
                        "response_model": record.get("response_model"),
                        "finish_reason": record.get("finish_reason"),
                        "raw_output": record.get("raw_output"),
                        "usage": record.get("usage"),
                    }
                )
            else:
                target.update(
                    {
                        "failed_at": record.get("timestamp"),
                        "error_type": record.get("error_type"),
                        "error": record.get("error"),
                    }
                )
        else:
            by_qid[qid]["events"].append(record)
            if event == "question_completed":
                by_qid[qid]["answer"] = record.get("answer")
                by_qid[qid]["reasoning"] = record.get("reasoning")
                by_qid[qid]["reasoning_evidence_refs"] = record.get(
                    "reasoning_evidence_refs",
                    [],
                )
                by_qid[qid]["token_usage"] = record.get("usage", {})
    for payload in by_qid.values():
        resolved = []
        for label in payload.get("reasoning_evidence_refs", []):
            candidates = []
            for table in payload.get("evidence_tables", []):
                for row in table.get("rows", []):
                    if str(row.get("label", "")) != str(label):
                        continue
                    candidates.append(
                        {
                            "call_id": table.get("call_id"),
                            "route": table.get("route"),
                            "option": table.get("option", ""),
                            "table_id": table.get("table_id"),
                            "evidence": row,
                        }
                    )
            resolved.append(
                {
                    "label": label,
                    "candidates": candidates,
                }
            )
        payload["reasoning_evidence_resolution"] = resolved
    return dict(sorted(by_qid.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    evidence = build_evidence(args.log)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"evidence questions={len(evidence)}")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
