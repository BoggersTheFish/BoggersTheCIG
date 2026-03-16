"""Tests for viz module (export, plot, graph snapshots)."""
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))


def test_auto_snapshot_graph_empty_returns_none(monkeypatch):
    """Empty graph skips snapshot and returns None."""
    import networkx as nx

    class EmptyGraph:
        _use_neo4j = False
        _nx_graph = nx.DiGraph()

    monkeypatch.setattr("src.viz.ConceptGraph", lambda: EmptyGraph())

    from src.viz import auto_snapshot_graph
    from src.config import PROJECT_ROOT

    tmp = PROJECT_ROOT / "tests" / "_snap_tmp"
    tmp.mkdir(exist_ok=True)
    try:
        out = auto_snapshot_graph(vault_path=tmp, reason="test-empty")
        assert out is None
    finally:
        if (tmp / "snapshots").exists():
            for f in (tmp / "snapshots").iterdir():
                f.unlink()
            (tmp / "snapshots").rmdir()


def test_auto_snapshot_graph_with_data(monkeypatch, tmp_path):
    """Snapshot creates PNG and updates index.md when graph has nodes."""
    import json
    import networkx as nx

    G = nx.DiGraph()
    G.add_edge("A", "B")
    G.add_edge("B", "C")
    graph_data = nx.node_link_data(G)
    (tmp_path / "networkx_fallback.json").write_text(json.dumps(graph_data))

    monkeypatch.setattr("src.config.GRAPHS_DIR", tmp_path)
    monkeypatch.setattr("src.concept_graph.GRAPHS_DIR", tmp_path)

    from src.viz import auto_snapshot_graph

    snap_dir = tmp_path / "snapshots"
    out = auto_snapshot_graph(vault_path=tmp_path, reason="test")
    assert out is not None
    p = Path(out)
    assert p.exists()
    assert p.suffix == ".png"
    idx = snap_dir / "index.md"
    assert idx.exists()
    content = idx.read_text()
    assert "TS Graph Snapshots" in content
    assert "test" in content


def test_export_to_obsidian():
    """Export creates Concepts dir and markdown files."""
    from src.viz import export_to_obsidian
    from src.concept_graph import ConceptGraph
    from src.config import PROJECT_ROOT

    graph = ConceptGraph()
    graph.ingest_triples([("Foo", "relates", "Bar")], source="test")

    tmp = PROJECT_ROOT / "tests" / "_viz_tmp"
    tmp.mkdir(exist_ok=True)
    try:
        export_to_obsidian(graph, target_dir=tmp / "Concepts")
        concepts = tmp / "Concepts"
        assert concepts.exists()
        files = list(concepts.glob("*.md"))
        assert len(files) >= 1
    finally:
        if (tmp / "Concepts").exists():
            for f in (tmp / "Concepts").iterdir():
                f.unlink()
            (tmp / "Concepts").rmdir()
