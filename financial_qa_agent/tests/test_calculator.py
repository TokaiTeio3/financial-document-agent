import pytest

from app.tools.safe_math import safe_calculate


def test_safe_calculator_supports_financial_formula():
    assert safe_calculate("(current / previous - 1) * 100", {"current": 120, "previous": 100}) == pytest.approx(20)


def test_safe_calculator_rejects_code_execution():
    with pytest.raises(ValueError):
        safe_calculate("__import__('os').system('whoami')")
