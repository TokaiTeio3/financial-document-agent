from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


class RunAuditLogger:
    """单次竞赛运行的线程安全 JSONL 审计轨迹。"""

    def __init__(self, path: str | Path, run_id: str = "") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self._lock = threading.Lock()
        self._sequence = 0

    def event(self, event: str, qid: str = "", **payload: Any) -> None:
        with self._lock:
            self._sequence += 1
            record: Dict[str, Any] = {
                "schema_version": 2,
                "run_id": self.run_id,
                "sequence": self._sequence,
                "timestamp": datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds"),
                "process_id": os.getpid(),
                "thread_name": threading.current_thread().name,
                "event": event,
                "qid": qid,
                **payload,
            }
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
