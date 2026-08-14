"""Unit tests for tick-label number parsing."""
import pytest

from mci.utils import parse_number_text


@pytest.mark.parametrize(
    "text,expected",
    [
        ("1.5", 1.5),
        ("0", 0.0),
        ("-3", -3.0),
        ("1500", 1500.0),
        ("1,500", 1500.0),
        ("0.01", 0.01),
        (".5", 0.5),
        ("1e-3", 0.001),
        ("1E3", 1000.0),
        ("2.5e+2", 250.0),
        ("1.5x10^3", 1500.0),
        ("1.5×10^3", 1500.0),
        ("1.5x10-3", 0.0015),
        ("1×10⁻³", 0.001),
        ("10^3", 1000.0),
        ("10-3", 0.001),
        ("10⁻³", 0.001),
        ("10⁰", 1.0),
        ("10^1", 10.0),
        ("−1", -1.0),
        ("–5", -5.0),
        ("1 000", 1000.0),
        ("45%", 45.0),
        ("10", 10.0),
        ("1000", 1000.0),
        ("100", 100.0),
        ("103", 103.0),
        ("10^3", 1000.0),
        ("10^00", 1.0),
    ],
)
def test_parse_ok(text, expected):
    assert parse_number_text(text) == pytest.approx(expected)


@pytest.mark.parametrize("text", ["", "abc", "1.5.2", "Creep curve", "--", "1e", "10x"])
def test_parse_fail(text):
    assert parse_number_text(text) is None


def test_parse_none():
    assert parse_number_text(None) is None
