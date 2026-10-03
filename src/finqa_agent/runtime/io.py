from __future__ import annotations

import csv
import json
import os
import re
import tempfile
from decimal import Decimal, InvalidOperation
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml

from ..core.entities import repair_mojibake


def load_env(path: str | Path = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def load_config(path: str | Path = "config/config.yaml") -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def read_json_or_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    p = Path(path)
    if p.suffix.lower() == ".jsonl":
        rows = []
        with p.open("r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    data = json.loads(p.read_text(encoding="utf-8-sig"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return list(data.values())
    return []


def save_json(data: Any, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=p.parent,
            prefix=f".{p.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(json.dumps(data, ensure_ascii=False, indent=2))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, p)
    finally:
        if temporary_name:
            Path(temporary_name).unlink(missing_ok=True)


def normalize_question(item: Dict[str, Any]) -> Dict[str, Any]:
    q = dict(item)
    for key in ("question", "type", "domain", "split"):
        if isinstance(q.get(key), str):
            q[key] = repair_mojibake(q[key])
    if isinstance(q.get("options"), dict):
        q["options"] = {
            str(key): repair_mojibake(str(value))
            for key, value in q["options"].items()
        }
    answer_format = str(q.get("answer_format", "")).strip().lower()
    qtype = str(q.get("type", ""))
    if not answer_format:
        if "多选" in qtype:
            answer_format = "multi"
        elif "判断" in qtype:
            answer_format = "tf"
        elif "计算" in qtype:
            answer_format = "calc"
        elif "提取" in qtype or "抽取" in qtype:
            answer_format = "extract"
        else:
            answer_format = "mcq"
    if answer_format == "single":
        answer_format = "mcq"
    q["answer_format"] = answer_format
    q.setdefault("options", {})
    return q


def load_questions(root: str | Path, split: str, qids: Optional[set[str]] = None) -> List[Dict[str, Any]]:
    root_path = Path(root)
    split_lower = split.lower()
    candidates = [
        root_path / f"group_{split_lower}",
        root_path / f"question_{split_lower}",
        root_path / f"upload_{split_lower}",
    ]
    if root_path.name.lower().endswith(split_lower):
        candidates.insert(0, root_path)
    files: List[Path] = []
    seen: set[str] = set()
    for directory in candidates:
        if not directory.exists():
            continue
        for path in sorted(list(directory.glob("*.json")) + list(directory.glob("*.jsonl"))):
            if path.name.lower() == "manifest.json":
                continue
            key = str(path.resolve()).lower()
            if key not in seen:
                seen.add(key)
                files.append(path)
    questions: List[Dict[str, Any]] = []
    for path in files:
        for row in read_json_or_jsonl(path):
            q = normalize_question(row)
            if str(q.get("split", "")).upper() == split.upper() or split.upper() == "B":
                if qids and q.get("qid") not in qids:
                    continue
                questions.append(q)
    questions.sort(key=lambda x: str(x.get("qid", "")))
    return questions


def normalize_answer(answer: str, answer_format: str) -> str:
    text = str(answer or "").strip()
    if answer_format in {"calc", "extract"}:
        text = re.sub(r"^```(?:json|text)?|```$", "", text.strip(), flags=re.I)
        text = re.sub(r"^(答案|最终答案|answer|final_answer)\s*[:：]\s*", "", text, flags=re.I)
        if answer_format == "calc":
            return re.sub(r"\s+", "", text)
        return re.sub(r"\s+", " ", text).strip()
    letters = "".join(ch for ch in text.upper() if ch in "ABCD")
    if answer_format == "multi":
        return "".join(sorted(set(letters)))
    return letters[:1]


def normalize_calculation_answer(answer: str, question: str) -> str:
    """在不使用题目特定数值的前提下应用竞赛统一数值格式。"""
    text = normalize_answer(answer, "calc")
    date_match = re.fullmatch(
        r"(20\d{2})(?:[-/]|年)0?(\d{1,2})(?:[-/]|月)0?(\d{1,2})(?:日)?",
        text,
    )
    if date_match and re.search(r"何时|日期|哪一天|工作日|公示", question):
        year, month, day = date_match.groups()
        return f"{year}年{int(month)}月{int(day)}日"

    precision_match = re.search(r"保留([一二三四])位小数", question)
    precision_map = {"一": 1, "二": 2, "三": 3, "四": 4}
    precision = precision_map.get(precision_match.group(1), 2) if precision_match else 2

    if "；" in text or ";" in text:
        parts = [part.strip() for part in re.split(r"[；;]", text) if part.strip()]
        normalized_parts = []
        explicit_last_percent = bool(re.search(r"后者.{0,12}以百分(?:数|比)计", question))
        last_is_percent = bool(
            re.search(r"后者.{0,12}(?:百分|%)|第二项.{0,12}(?:百分|%)|绝对相对偏差.{0,20}(?:百分|%)", question)
        )
        forbid_last_percent = bool(re.search(r"后者.{0,12}不带\s*%", question))
        forbid_all_percent = bool(
            re.search(r"均.{0,12}不带单位", question)
            and "百分点" in question
            and not explicit_last_percent
        )
        explicit_percent_forbidden = bool(
            re.search(r"不带\s*(?:%|百分号)", question)
        )
        generic_unit_forbidden = bool(
            re.search(r"不带\s*单位|只填(?:写)?数值", question)
        )
        forbid_all_percent = (
            forbid_all_percent
            or explicit_percent_forbidden
            # 输出格式要求优先于计算中使用的语义单位。
            # 例如“以百分数计，均不带单位”表示输出百分数数值但不带 “%” 后缀。
            or generic_unit_forbidden
        )
        for index, part in enumerate(parts):
            percent_match = re.fullmatch(r"([-+]?\d[\d,]*(?:\.\d+)?)\s*%", part)
            if percent_match:
                suffix = ""
                if not forbid_all_percent and not (forbid_last_percent and index == len(parts) - 1):
                    suffix = "%"
                normalized_parts.append(_format_decimal(percent_match.group(1), precision, suffix))
            elif re.fullmatch(r"[-+]?\d[\d,]*(?:\.\d+)?", part):
                suffix = (
                    "%"
                    if last_is_percent
                    and index == len(parts) - 1
                    and not forbid_all_percent
                    and not forbid_last_percent
                    else ""
                )
                normalized_parts.append(_format_decimal(part, precision, suffix))
            else:
                match = re.fullmatch(r"(.+?)([-+]?\d[\d,]*(?:\.\d+)?)", part)
                if match and ">" in match.group(1):
                    normalized_parts.append(match.group(1) + _format_decimal(match.group(2), precision))
                else:
                    normalized_parts.append(part)
        return "；".join(normalized_parts)

    percentages = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?\s*%", text)
    if percentages:
        return "；".join(_format_decimal(value.rstrip("%").strip(), precision, "%") for value in percentages)

    percentage = re.fullmatch(r"([-+]?\d[\d,]*(?:\.\d+)?)\s*%", text)
    if percentage:
        return _format_decimal(percentage.group(1), precision, "%")
    day_count = re.fullmatch(r"([-+]?\d[\d,]*(?:\.\d+)?)日", text)
    if day_count and re.search(r"(?:多少|相隔|间隔).{0,8}日|日数", question):
        return _format_decimal(day_count.group(1), precision)
    asked_unit_value = re.fullmatch(
        r"([-+]?\d[\d,]*(?:\.\d+)?)(亿元|万元|元|万人|人|笔|倍)",
        text,
    )
    if asked_unit_value and re.search(
        rf"(?:多少|约为|为)\s*{re.escape(asked_unit_value.group(2))}",
        question,
    ):
        return _format_decimal(asked_unit_value.group(1), precision)
    if re.fullmatch(r"[-+]?\d[\d,]*(?:\.\d+)?", text):
        suffix = "%" if re.search(r"答案.{0,20}(?:百分|%)|以百分(?:数|比)", question) else ""
        return _format_decimal(text, precision, suffix)
    return text


def _format_decimal(value: str, precision: int, suffix: str = "") -> str:
    try:
        number = Decimal(value.replace(",", ""))
    except InvalidOperation:
        return value + suffix
    # 评测器需要纯数字字符串；千分位分隔符只是展示装饰，
    # 可能拆开原本精确的数值答案。
    rendered = f"{number:.{precision}f}"
    return rendered + suffix


def answer_parts(answer: str, answer_format: str) -> List[str]:
    normalized = normalize_answer(answer, answer_format)
    if answer_format in {"calc", "extract"}:
        parts = [p.strip() for p in re.split(r"[;,，；]", normalized) if p.strip()]
        return parts or [normalized]
    return [normalized]


def timestamped_submission_path(output_dir: str | Path, split: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(output_dir) / f"submission_{split.lower()}_{ts}.csv"


def timestamped_run_dir(output_dir: str | Path) -> Path:
    base = Path(output_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return base / stamp


def write_submission(results: Dict[str, Dict[str, Any]], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    total_prompt = sum(int(r.get("prompt_tokens", 0)) for r in results.values())
    total_completion = sum(int(r.get("completion_tokens", 0)) for r in results.values())
    total = sum(int(r.get("total_tokens", 0)) for r in results.values())
    temporary_name = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            dir=p.parent,
            prefix=f".{p.name}.",
            suffix=".tmp",
            delete=False,
        ) as f:
            temporary_name = f.name
            writer = csv.writer(f)
            writer.writerow([
                "qid",
                "answer_1",
                "answer_2",
                "answer_3",
                "answer_4",
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "reasoning",
            ])
            writer.writerow(["summary", "", "", "", "", total_prompt, total_completion, total, ""])
            for qid in sorted(results):
                row = results[qid]
                parts = (list(row.get("answer_parts") or []) + ["", "", "", ""])[:4]
                reasoning = re.sub(r"\s+", " ", str(row.get("reasoning", ""))).strip()
                writer.writerow([
                    qid,
                    parts[0],
                    parts[1],
                    parts[2],
                    parts[3],
                    row.get("prompt_tokens", 0),
                    row.get("completion_tokens", 0),
                    row.get("total_tokens", 0),
                    reasoning,
                ])
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary_name, p)
    finally:
        if temporary_name:
            Path(temporary_name).unlink(missing_ok=True)
