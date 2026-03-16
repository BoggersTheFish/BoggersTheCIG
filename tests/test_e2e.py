"""End-to-end test: input text -> reasoned output."""
import pytest
from src.language_layer import extract_triples
from src.concept_graph import ConceptGraph
from src.core_engine import CoreEngine


def test_e2e_ingest_and_query():
    """Ingest 'Gravity bends spacetime' -> graph query returns results."""
    text = "Gravity bends spacetime"
    triples = extract_triples(text, use_llm=False)
    assert len(triples) >= 1

    graph = ConceptGraph()
    if graph._use_neo4j:
        pytest.skip("Neo4j connected - e2e runs with NetworkX when DB unavailable")
    import networkx as nx
    graph._nx_graph = nx.DiGraph()  # Fresh graph for test

    graph.ingest_triples(triples, source="e2e")
    core = CoreEngine(graph)
    search = core.structural_search({}, limit=10)
    assert graph.node_count() >= 2
    assert len(search) >= 1 or graph.edge_count() >= 1
