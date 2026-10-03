from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

from app.tools.safe_math import safe_calculate


@tool
def calculate_finance(expression: str, variables: dict[str, float] | None = None, unit: str = "") -> dict[str, Any]:
    """执行金融问答所需的安全算术。支持四则运算、乘方、百分比换算及变量；禁止执行代码。"""
    value = safe_calculate(expression, variables)
    return {"expression": expression, "variables": variables or {}, "value": value, "unit": unit}
