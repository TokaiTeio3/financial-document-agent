from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


class PriorApiAnswerMemory:
    """根据此前 API 答题运行构建的只读索引。

    该文件在运行时生成，并有意与源码分离，其中不包含评测标签。
    """

    def __init__(self, path: str | Path, enabled: bool = False) -> None:
        self.path = Path(path)
        self.enabled = enabled
        self._rows: Dict[str, Dict[str, object]] = {}
        if enabled and self.path.exists():
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                rows = payload.get("answers", payload)
                if isinstance(rows, dict):
                    self._rows = {
                        str(qid): dict(row)
                        for qid, row in rows.items()
                        if isinstance(row, dict)
                    }

    def get(self, qid: str) -> Dict[str, object] | None:
        row = self._rows.get(qid)
        return dict(row) if row else None

    def __len__(self) -> int:
        return len(self._rows)
