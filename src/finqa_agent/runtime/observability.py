"""根据不可变运行日志构建生产可观测性产物。"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable

from .io import save_json


def read_events(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def export_observability_artifacts(
    log_path: Path,
    output_dir: Path,
) -> Dict[str, str]:
    event_counts: Counter[str] = Counter()
    route_counts: Counter[str] = Counter()
    reasoning_levels: Counter[str] = Counter()
    evidence_tables: Dict[str, list[Dict[str, object]]] = defaultdict(list)
    api_calls = 0
    api_errors = 0
    for event in read_events(log_path):
        name = str(event.get("event", ""))
        qid = str(event.get("qid", ""))
        event_counts[name] += 1
        if name == "api_response":
            api_calls += 1
            route_counts[str(event.get("route", "unclassified"))] += 1
        elif name == "api_error":
            api_errors += 1
        elif name == "reasoning_level_assigned":
            allocation = event.get("allocation") or {}
            reasoning_levels[str(allocation.get("level", "unknown"))] += 1
        elif name == "evidence_table":
            evidence_tables[qid].append(
                {
                    "call_id": event.get("call_id"),
                    "route": event.get("route"),
                    "option": event.get("option", ""),
                    "table_id": event.get("table_id"),
                    "table_sha256": event.get("table_sha256"),
                    "rows": event.get("rows", []),
                }
            )

    evidence_path = output_dir / "evidence_tables.json"
    metrics_path = output_dir / "run_metrics.json"
    save_json(dict(sorted(evidence_tables.items())), evidence_path)
    save_json(
        {
            "schema_version": 1,
            "api_success_count": api_calls,
            "api_error_count": api_errors,
            "event_counts": dict(sorted(event_counts.items())),
            "api_route_counts": dict(sorted(route_counts.items())),
            "reasoning_level_counts": dict(sorted(reasoning_levels.items())),
            "evidence_table_question_count": len(evidence_tables),
            "evidence_table_count": sum(
                len(tables) for tables in evidence_tables.values()
            ),
        },
        metrics_path,
    )
    return {
        "evidence_tables": evidence_path.name,
        "run_metrics": metrics_path.name,
    }
