from __future__ import annotations

import ast
import math
import operator


_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
}


def safe_calculate(expression: str, variables: dict[str, float] | None = None) -> float:
    variables = variables or {}
    if len(expression) > 300:
        raise ValueError("表达式过长")
    tree = ast.parse(expression, mode="eval")

    def evaluate(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name) and node.id in variables:
            return float(variables[node.id])
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            return float(_BINARY[type(node.op)](evaluate(node.left), evaluate(node.right)))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return float(_UNARY[type(node.op)](evaluate(node.operand)))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCTIONS:
            if node.keywords:
                raise ValueError("不允许关键字参数")
            return float(_FUNCTIONS[node.func.id](*[evaluate(argument) for argument in node.args]))
        raise ValueError(f"不支持的表达式节点: {type(node).__name__}")

    result = evaluate(tree)
    if not math.isfinite(result):
        raise ValueError("计算结果不是有限数")
    return result
