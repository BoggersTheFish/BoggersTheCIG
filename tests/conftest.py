"""Pytest configuration - force NetworkX fallback for tests (no Docker/Neo4j)."""
import os

# Run before any tests
os.environ.setdefault("GRAPH_URI", "bolt://localhost:7687")
# Use a non-connectable URI to force NetworkX when Neo4j isn't running
# Tests that need Neo4j can override this
