import pytest

from agent.tools.calculator import CalculatorError, calculate


def test_basic_arithmetic():
    assert calculate("2 + 2") == 4


def test_operator_precedence_and_parens():
    assert calculate("(12 + 3) * 2 / 5") == 6.0


def test_division_by_zero_raises():
    with pytest.raises(CalculatorError):
        calculate("1 / 0")


def test_invalid_syntax_raises():
    with pytest.raises(CalculatorError):
        calculate("2 + ")


def test_rejects_non_arithmetic_code():
    """Guards against the exact class of bug eval() would introduce."""
    with pytest.raises(CalculatorError):
        calculate("__import__('os').system('echo pwned')")
