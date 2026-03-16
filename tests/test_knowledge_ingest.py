"""Tests for knowledge_ingest module."""
import json
import os
from pathlib import Path

import pytest

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))


def test_load_queries():
    """Load queries from data/queries.json."""
    from src.knowledge_ingest import _load_queries

    queries = _load_queries()
    assert isinstance(queries, list)
    assert len(queries) >= 1


def test_ingest_external_disabled():
    """When ENABLE_EXTERNAL_INGEST=false (default), returns skipped."""
    from src.knowledge_ingest import ingest_external_knowledge
    from src.config import ENABLE_EXTERNAL_INGEST

    result = ingest_external_knowledge()
    if not ENABLE_EXTERNAL_INGEST:
        assert "skipped" in result or result.get("queries_run", 0) == 0
    else:
        assert isinstance(result, dict)
        assert "queries_run" in result or "triples_ingested" in result


def test_ingest_external_force():
    """With force=True, runs even when ENABLE_EXTERNAL_INGEST=false."""
    from src.knowledge_ingest import ingest_external_knowledge

    result = ingest_external_knowledge(force=True, queries=[], max_per_query=1)
    assert isinstance(result, dict)
    assert "queries_run" in result
    assert "triples_ingested" in result
    assert "skipped" not in result


def test_ensure_queries_file():
    """ensure_queries_file creates data/queries.json if missing."""
    from src.knowledge_ingest import ensure_queries_file, _load_queries
    from src.config import QUERIES_PATH

    ensure_queries_file()
    assert QUERIES_PATH.exists()
    queries = _load_queries()
    assert isinstance(queries, list)
    assert len(queries) >= 1


def test_filter_harmful():
    """Harmful triples are filtered."""
    from src.knowledge_ingest import _filter_harmful

    triples = [("A", "causes", "harm"), ("B", "is", "safe")]
    filtered = _filter_harmful(triples)
    assert ("A", "causes", "harm") not in filtered
    assert ("B", "is", "safe") in filtered


def test_generate_queries_from_graph_no_ollama(tmp_path, monkeypatch):
    """generate_queries_from_graph with use_ollama=False uses manual fallback."""
    from src.knowledge_ingest import generate_queries_from_graph, _load_queries_structure, _save_queries
    from src.config import QUERIES_PATH

    monkeypatch.setattr("src.knowledge_ingest.QUERIES_PATH", tmp_path / "queries.json")
    monkeypatch.setattr("src.config.QUERIES_PATH", tmp_path / "queries.json")
    _save_queries(manual=["manual1", "manual2"], generated=[])

    result = generate_queries_from_graph(use_ollama=False, reason="test")
    assert "queries" in result
    assert "saved" in result
    assert result["reason"] == "test"
    assert len(result["queries"]) >= 1
    assert result["queries"][0] == "manual1"

    data = json.loads((tmp_path / "queries.json").read_text())
    assert "manual" in data
    assert "generated" in data
    assert "last_generated" in data
    assert data["last_generated"]["reason"] == "test"


def test_generate_queries_from_graph_saves_structure(tmp_path, monkeypatch):
    """Generated queries are saved with manual preserved."""
    from src.knowledge_ingest import generate_queries_from_graph, _save_queries
    from src.config import QUERIES_PATH

    monkeypatch.setattr("src.knowledge_ingest.QUERIES_PATH", tmp_path / "queries.json")
    monkeypatch.setattr("src.config.QUERIES_PATH", tmp_path / "queries.json")
    _save_queries(manual=["keep1", "keep2"], generated=["old_gen"])

    generate_queries_from_graph(use_ollama=False, reason="test")

    data = json.loads((tmp_path / "queries.json").read_text())
    assert data["manual"] == ["keep1", "keep2"]
    assert data["generated"] == ["keep1", "keep2"][:5]
    assert data["last_generated"]["ts"]
    assert data["last_generated"]["reason"] == "test"


def test_load_queries_merged(tmp_path, monkeypatch):
    """_load_queries returns manual + generated."""
    from src.knowledge_ingest import _load_queries, _save_queries

    monkeypatch.setattr("src.knowledge_ingest.QUERIES_PATH", tmp_path / "queries.json")
    monkeypatch.setattr("src.config.QUERIES_PATH", tmp_path / "queries.json")
    _save_queries(manual=["a", "b"], generated=["c", "d"])
    q = _load_queries()
    assert "a" in q and "b" in q and "c" in q and "d" in q
