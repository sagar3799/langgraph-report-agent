"""Arithmetic calculator tool.

Deliberately does NOT use eval() — expressions come from LLM output, which is
untrusted input, so arbitrary code execution is not an acceptable risk. Instead
we walk a restricted AST that only permits numeric literals and basic operators.
"""

import ast
import operator

_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


class CalculatorError(ValueError):
    pass


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_eval_node(node.operand))
    raise CalculatorError(f"Unsupported expression element: {ast.dump(node)}")


def calculate(expression: str) -> float:
    """Safely evaluate a basic arithmetic expression, e.g. '(12 + 3) * 2 / 5'."""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise CalculatorError(f"Invalid expression syntax: {expression!r}") from exc
    try:
        return _eval_node(tree.body)
    except ZeroDivisionError as exc:
        raise CalculatorError("Division by zero") from exc
