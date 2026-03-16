"""Tests for language layer (triple extraction)."""
import pytest
from src.language_layer import extract_triples, _filter_harmful


def test_extract_triples_gravity():
    """Gravity bends spacetime -> (Gravity, bends, spacetime)."""
    triples = extract_triples("Gravity bends spacetime", use_llm=False)
    assert len(triples) >= 1
    found = any("Gravity" in str(t[0]) and "spacetime" in str(t[2]) for t in triples)
    assert found, f"Expected (Gravity, bends, spacetime) in {triples}"


def test_extract_triples_multiple():
    """Multiple patterns in one text."""
    text = "Water is a liquid. Water has hydrogen. Gravity affects planets."
    triples = extract_triples(text, use_llm=False)
    assert len(triples) >= 2


def test_filter_harmful():
    """Harmful patterns are filtered."""
    triples = [("A", "causes", "harm"), ("B", "is", "safe")]
    filtered = _filter_harmful(triples)
    assert ("A", "causes", "harm") not in filtered
    assert ("B", "is", "safe") in filtered


def test_empty_input():
    """Empty input returns empty list."""
    assert extract_triples("", use_llm=False) == []
    assert extract_triples("   ", use_llm=False) == []
