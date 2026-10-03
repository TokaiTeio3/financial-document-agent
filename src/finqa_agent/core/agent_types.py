"""金融问答 Agent 返回的公开结果类型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from ..runtime.llm import Usage


@dataclass
class AnswerResult:
    qid: str
    answer: str
    reasoning: str
    usage: Usage
    metadata: Dict[str, object] = field(default_factory=dict)
