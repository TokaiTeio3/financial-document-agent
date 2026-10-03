from __future__ import annotations

import datetime
import math
import multiprocessing as mp
import re
import traceback
from dataclasses import dataclass
from typing import Dict


def format_date_zh(value: datetime.date | datetime.datetime) -> str:
    return f"{value.year:04d}年{value.month:02d}月{value.day:02d}日"


def format_date_iso(value: datetime.date | datetime.datetime) -> str:
    return f"{value.year:04d}-{value.month:02d}-{value.day:02d}"


@dataclass
class ExecutionResult:
    ok: bool
    answer: str = ""
    error: str = ""
    code: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {"ok": self.ok, "answer": self.answer, "error": self.error, "code": self.code}


class RestrictedPythonExecutor:
    SAFE_BUILTINS = {
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "dict": dict,
        "enumerate": enumerate,
        "float": float,
        "format": format,
        "int": int,
        "len": len,
        "list": list,
        "max": max,
        "min": min,
        "pow": pow,
        "range": range,
        "round": round,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "zip": zip,
    }

    SAFE_GLOBALS = {
        "__builtins__": SAFE_BUILTINS,
        "math": math,
        "datetime": datetime.datetime,
        "date": datetime.date,
        "timedelta": datetime.timedelta,
        "format_date_zh": format_date_zh,
        "format_date_iso": format_date_iso,
    }

    def __init__(self, timeout_seconds: float = 2.0) -> None:
        self.timeout_seconds = timeout_seconds

    def execute(self, code: str) -> ExecutionResult:
        text = self._strip(code)
        if any(blocked in text for blocked in ["open(", "exec(", "eval(", "compile(", "__import__", "input("]):
            return ExecutionResult(False, error="blocked unsafe code", code=text)
        queue: mp.Queue = mp.Queue()
        process = mp.Process(target=self._worker, args=(text, queue))
        process.start()
        process.join(self.timeout_seconds)
        if process.is_alive():
            process.terminate()
            process.join(0.2)
            return ExecutionResult(False, error="timeout", code=text)
        try:
            payload = queue.get_nowait()
        except Exception:
            return ExecutionResult(False, error="no result", code=text)
        return ExecutionResult(code=text, **payload)

    @staticmethod
    def _strip(code: str) -> str:
        text = str(code or "").strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines)
        text = text.replace("\\n", "\n").replace('\\"', '"')
        # 沙箱已经注入这些安全符号。模型偶尔仍会生成对应 import，
        # 这里只移除这类白名单 import 语法，而不是开放 __import__。
        text = re.sub(
            r"(?m)^\s*(?:from\s+datetime\s+import\s+[\w,\s]+|"
            r"import\s+(?:datetime|math)(?:\s+as\s+\w+)?)\s*$",
            "",
            text,
        )
        text = re.sub(
            r"(\b\w+)\.strftime\(\s*['\"]%Y年%m月%d日['\"]\s*\)",
            r"format_date_zh(\1)",
            text,
        )
        text = re.sub(
            r"(\b\w+)\.strftime\(\s*['\"]%Y-%m-%d['\"]\s*\)",
            r"format_date_iso(\1)",
            text,
        )
        return text.strip()

    @classmethod
    def _worker(cls, code: str, queue: mp.Queue) -> None:
        namespace: Dict[str, object] = {}
        try:
            exec(compile(code, "<qwen_calc>", "exec"), dict(cls.SAFE_GLOBALS), namespace)
            answer = namespace.get("answer", namespace.get("result", ""))
            queue.put({"ok": True, "answer": str(answer), "error": ""})
        except Exception:
            queue.put({"ok": False, "answer": "", "error": traceback.format_exc(limit=2)})
