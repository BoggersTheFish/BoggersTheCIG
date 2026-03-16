"""Tests for obsidian_ollama_bridge module."""
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))


def test_parse_markdown_to_text():
    """Parse markdown to plain text."""
    from src.obsidian_ollama_bridge import _parse_markdown_to_text

    md = PROJECT_ROOT / "obsidian" / "TS-Knowledge-Vault" / "Concepts" / "Gravity.md"
    if md.exists():
        text = _parse_markdown_to_text(md)
        assert "Gravity" in text or len(text) > 0
    else:
        text = _parse_markdown_to_text(Path(__file__))
        assert isinstance(text, str)


def test_chunk_text():
    """Chunk text into overlapping segments."""
    from src.obsidian_ollama_bridge import _chunk_text

    text = "a" * 2000
    chunks = _chunk_text(text, chunk_size=500, overlap=50)
    assert len(chunks) >= 2
    assert sum(len(c) for c in chunks) >= len(text) - 200


def test_analyze_vault_empty():
    """Analyze vault returns stats dict (vault may be empty)."""
    from src.obsidian_ollama_bridge import analyze_vault

    result = analyze_vault(use_ollama=False)
    assert "files_read" in result
    assert "triples_ingested" in result
    assert isinstance(result["triples_ingested"], int)


def test_extract_triples_from_vault_format():
    """Extract triples from Obsidian neighbor format."""
    from src.obsidian_ollama_bridge import _extract_triples_from_vault_format

    raw = "# Gravity\n\n## Neighbors\n\n- [[spacetime]] (bends, weight=1)"
    triples = _extract_triples_from_vault_format(Path("Gravity.md"), raw)
    assert len(triples) >= 1
    assert any("Gravity" in str(t[0]) and "spacetime" in str(t[2]) for t in triples)


def test_analyze_vault_no_ollama():
    """Analyze vault with use_ollama=False (rule-based only)."""
    from src.obsidian_ollama_bridge import analyze_vault

    result = analyze_vault(use_ollama=False)
    assert "files_read" in result
    assert "triples_ingested" in result
    assert "contradictions" in result
