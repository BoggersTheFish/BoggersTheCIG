"""Tests for concept graph (ingest, query)."""
import pytest
from src.concept_graph import ConceptGraph


@pytest.fixture
def graph():
    """Fresh graph - uses NetworkX when Neo4j unavailable."""
    g = ConceptGraph()
    if g._use_neo4j:
        pytest.skip("Neo4j connected - use NetworkX for unit tests")
    # Clear for isolation
    import networkx as nx
    g._nx_graph = nx.DiGraph()
    return g


def test_ingest_triples(graph):
    """Ingest triples creates nodes and edges."""
    triples = [("Gravity", "bends", "spacetime"), ("Earth", "has", "gravity")]
    count = graph.ingest_triples(triples, source="test")
    assert count >= 2
    assert graph.node_count() >= 3
    assert graph.edge_count() >= 2


def test_get_neighbors(graph):
    """Neighbors are returned correctly."""
    graph.ingest_triples([("A", "causes", "B"), ("A", "relates_to", "C")], source="test")
    neighbors = graph.get_neighbors("A", limit=10)
    names = [n[0] for n in neighbors]
    assert "B" in names
    assert "C" in names


def test_semantic_search(graph):
    """Semantic search returns results."""
    graph.ingest_triples([("Gravity", "bends", "spacetime")], source="test")
    results = graph.semantic_search("gravity", top_k=5)
    assert len(results) >= 1
    assert any("Gravity" in r[0] or "spacetime" in r[0] for r in results)
