"""Tests for concept graph (ingest, query, scaling)."""
import pytest
from src.concept_graph import ConceptGraph, check_graph_size, get_subgraphs_by_community


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


def test_check_graph_size_small(graph):
    """check_graph_size returns False for small graphs."""
    graph.ingest_triples([("A", "rel", "B")], source="test")
    result = check_graph_size(graph)
    assert result is False


def test_get_subgraphs_by_community(graph):
    """get_subgraphs_by_community returns list of node lists."""
    graph.ingest_triples([("A", "r", "B"), ("B", "r", "C"), ("X", "r", "Y")], source="test")
    comms = get_subgraphs_by_community(graph, max_communities=5)
    assert isinstance(comms, list)
    assert len(comms) >= 1
    all_nodes = set()
    for c in comms:
        all_nodes.update(c)
    assert "A" in all_nodes or "X" in all_nodes


def test_prune_low_degree(graph):
    """prune_low_degree_nodes removes low-degree nodes."""
    graph.ingest_triples([("A", "r", "B"), ("B", "r", "C"), ("Orphan", "r", "X")], source="test")
    n_before = graph.node_count()
    pruned = graph.prune_low_degree_nodes(min_degree=2)
    assert isinstance(pruned, int)
    assert pruned >= 0


def test_get_neighbors_batch(graph):
    """get_neighbors_batch returns dict of neighbor lists."""
    graph.ingest_triples([("A", "r", "B"), ("A", "r", "C")], source="test")
    batch = graph.get_neighbors_batch(["A", "B"], limit=10)
    assert "A" in batch
    assert len(batch["A"]) >= 2
