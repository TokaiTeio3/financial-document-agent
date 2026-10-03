from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class GroundTruthItem(BaseModel):
    qid: str
    domain: str
    question: str
    type: str
    options: dict[str, str] = Field(default_factory=dict)
    answer: str
    answer_parts: list[str] = Field(default_factory=list)
    reasoning_and_evidence: str = ""
    source_materials: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def prompt(self) -> str:
        if not self.options:
            return self.question
        choices = "\n".join(f"{letter}. {text}" for letter, text in self.options.items())
        return f"{self.question}\n\n选项：\n{choices}\n\n请给出正确选项，并用文档证据说明。"

    @property
    def expected_document_ids(self) -> list[str]:
        output: list[str] = []
        for source in self.source_materials:
            stem = Path(str(source.get("file", ""))).stem
            # Regulatory txt sources carry a long Chinese title while processed files use the code.
            match = re.match(r"((?:strict_v3|csrc)_\d+(?:_att\d+)?)", stem, re.IGNORECASE)
            output.append((match.group(1) if match else stem).lower())
        return sorted(set(filter(None, output)))

    @property
    def expected_pages(self) -> list[int]:
        pages: list[int] = []
        for source in self.source_materials:
            pages.extend(int(page) for page in source.get("pages", []) if str(page).isdigit())
        return sorted(set(pages))


def repair_mojibake(value: Any) -> Any:
    if isinstance(value, dict):
        return {repair_mojibake(key): repair_mojibake(item) for key, item in value.items()}
    if isinstance(value, list):
        return [repair_mojibake(item) for item in value]
    if not isinstance(value, str):
        return value
    try:
        repaired = value.encode("latin-1").decode("gbk")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return value
    mojibake_markers = sum(value.count(char) for char in "¡££¬£º¼ÆÌâÑ¡¸ù¾Ý")
    return repaired if mojibake_markers or any("\u00c0" <= char <= "\u00ff" for char in value) else value


def load_ground_truth(path: Path) -> list[GroundTruthItem]:
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["answers"] if isinstance(payload, dict) else payload
    return [GroundTruthItem.model_validate(repair_mojibake(row)) for row in rows]


def normalized_answer(text: str) -> str:
    return re.sub(r"[\s,，。；;：:%％万元亿元人民币]", "", str(text)).upper()


def score_answer(item: GroundTruthItem, answer: dict[str, Any]) -> bool:
    if "选题" in item.type:
        predicted = "".join(sorted(str(letter).upper() for letter in answer.get("selected_options", [])))
        if not predicted:
            match = re.search(r"答案\s*[:：]?\s*([A-D]+)", str(answer.get("answer", "")).upper())
            predicted = "".join(sorted(match.group(1))) if match else ""
        return predicted == "".join(sorted(item.answer.upper()))
    candidate = str(answer.get("final_value") or answer.get("answer") or "")
    expected_parts = item.answer_parts or [item.answer]
    candidate_numbers = re.findall(r"-?\d+(?:\.\d+)?", candidate.replace(",", ""))
    expected_numbers = [
        number
        for part in expected_parts
        for number in re.findall(r"-?\d+(?:\.\d+)?", str(part).replace(",", ""))
    ]
    if expected_numbers and len(candidate_numbers) == len(expected_numbers):
        try:
            return [Decimal(number) for number in candidate_numbers] == [Decimal(number) for number in expected_numbers]
        except InvalidOperation:
            pass
    normalized_candidate = normalized_answer(candidate)
    return all(normalized_answer(part) in normalized_candidate for part in expected_parts)
