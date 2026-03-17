"""
Subsystem 10: Control System - FastAPI
Endpoints: /expand/{concept}, /analyse/{node}, /map/{domain}, /ingest
"""
import logging
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
import secrets

from src.concept_graph import ConceptGraph
from src.core_engine import CoreEngine
from src.hypothesis_generator import HypothesisGenerator
from src.language_layer import extract_triples
from src.validation_engine import ValidationEngine
from src.viz import export_to_obsidian

logger = logging.getLogger(__name__)

app = FastAPI(title="Full-TS Cognitive Architecture API", version="0.1.0")
security = HTTPBasic(auto_error=False)  # No 401 when auth disabled

# Optional basic auth (disabled by default for local dev)
REQUIRE_AUTH = False


def verify_auth(credentials: Optional[HTTPBasicCredentials] = Depends(security)):
    if not REQUIRE_AUTH:
        return True
    if not credentials:
        raise HTTPException(status_code=401, detail="Authentication required")
    correct = secrets.compare_digest(credentials.username.encode(), b"admin")
    correct &= secrets.compare_digest(credentials.password.encode(), b"changeme")
    if not correct:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return True


class IngestRequest(BaseModel):
    text: str


@app.get("/")
def root():
    return {"service": "Full-TS", "status": "ok"}


@app.get("/expand/{concept}")
def expand_concept(concept: str, _=Depends(verify_auth)):
    """Generate hypotheses for a concept."""
    graph = ConceptGraph()
    hyp = HypothesisGenerator(graph)
    neighbors = graph.get_neighbors(concept, limit=20)
    candidates = hyp.generate_candidates(concept, neighbors)
    return {"concept": concept, "neighbors": neighbors, "hypotheses": candidates}


@app.get("/analyse/{node}")
def analyse_node(node: str, _=Depends(verify_auth)):
    """Run reasoning on a node: structural search, patterns, conflicts."""
    graph = ConceptGraph()
    core = CoreEngine(graph)
    search = core.structural_search({"relation_type": None}, limit=20)
    patterns = core.pattern_discovery(min_pattern_size=2)
    conflicts = core.constraint_resolution()
    return {"node": node, "structural_search": search, "patterns": patterns[:10], "conflicts": conflicts}


@app.get("/map/{domain}")
def map_domain(domain: str, _=Depends(verify_auth)):
    """Export subgraph for domain to Obsidian."""
    graph = ConceptGraph()
    similar = graph.semantic_search(domain, top_k=50)
    nodes = [n for n, _ in similar]
    export_to_obsidian(graph, subgraph_nodes=nodes)
    return {"domain": domain, "nodes_exported": len(nodes), "path": "viz/"}


@app.post("/ingest")
def ingest_text(req: IngestRequest, _=Depends(verify_auth)):
    """Ingest text, extract triples, add to graph."""
    triples = extract_triples(req.text, use_llm=False)
    graph = ConceptGraph()
    val = ValidationEngine(graph)
    valid = val.validate_batch(triples)
    count = graph.ingest_triples(valid, source="api")
    return {"triples_extracted": len(triples), "triples_valid": len(valid), "edges_created": count}


@app.get("/stats")
def stats(_=Depends(verify_auth)):
    """Graph statistics."""
    graph = ConceptGraph()
    return {"nodes": graph.node_count(), "edges": graph.edge_count()}
