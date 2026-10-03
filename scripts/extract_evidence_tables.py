"""从现有不可变 API 日志提取调用级 EvidenceTable。"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from src.finqa_agent.retrieval.evidence_trace import (
    extract_evidence_table,
    infer_option_from_messages,
    table_fingerprint,
)
from src.finqa_agent.runtime.io import save_json


def extract_from_log(path: Path) -> Dict[str, List[Dict[str, object]]]:
    tables: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            event = json.loads(line)
            if event.get("event") != "api_request":
                continue
            messages = event.get("messages") or []
            rows = extract_evidence_table(messages)
            if not rows:
                continue
            qid = str(event.get("qid", ""))
            call_id = event.get("call_id")
            route = str(event.get("route", "unclassified"))
            tables[qid].append(
                {
                    "call_id": call_id,
                    "route": route,
                    "option": infer_option_from_messages(messages),
                    "table_id": f"{qid}:{call_id}:{route}",
                    "table_sha256": table_fingerprint(rows),
                    "rows": rows,
                }
            )
    return dict(sorted(tables.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    tables = extract_from_log(args.log)
    save_json(tables, args.output)
    print(f"questions={len(tables)}")
    print(f"tables={sum(len(value) for value in tables.values())}")
    print(
        "rows="
        + str(
            sum(
                len(table.get("rows", []))
                for value in tables.values()
                for table in value
            )
        )
    )
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
