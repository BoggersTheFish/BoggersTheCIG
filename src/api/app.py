"""
Subsystem 10: Control System - FastAPI
Endpoints: /expand/{concept}, /analyse/{node}, /map/{domain}, /ingest,
           /ask, /trace, /contradictions, /bridges, /metrics, /dashboard
"""
import logging
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.responses import HTMLResponse
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


# ── Phase 3.1: Graph-grounded Q&A ────────────────────────────────────────────

@app.get("/ask")
def ask(
    q: str = Query(..., description="Natural language question"),
    max_hops: int = Query(2, description="Neighborhood expansion depth"),
    _=Depends(verify_auth),
):
    """
    Graph-grounded Q&A. Extracts query concepts, expands 2-hop subgraph,
    serializes as grounded context, answers via Ollama citing specific nodes.
    """
    graph = ConceptGraph()
    core = CoreEngine(graph)

    # Step 1: extract concepts from query
    query_triples = extract_triples(q, use_llm=False)
    query_concepts = list({s for s, _, _ in query_triples} | {o for _, _, o in query_triples})

    # Also semantic search for query terms directly
    semantic_hits = graph.semantic_search(q, top_k=8)
    query_concepts += [n for n, _ in semantic_hits if n not in query_concepts]
    query_concepts = query_concepts[:12]

    if not query_concepts:
        return {"question": q, "answer": "No relevant concepts found in the knowledge graph.", "cited_nodes": [], "context_edges": 0}

    # Step 2: expand neighborhood
    seen_nodes = set(query_concepts)
    all_edges = []
    for concept in query_concepts:
        edges = graph.get_edges_with_meta(concept, limit=15)
        all_edges.extend(edges)
        seen_nodes.add(concept)
        if max_hops >= 2:
            for e in edges[:5]:
                nb = e["neighbor"]
                if nb not in seen_nodes:
                    seen_nodes.add(nb)
                    hop2 = graph.get_edges_with_meta(nb, limit=8)
                    all_edges.extend(hop2)

    # Step 3: serialize subgraph as grounded context
    context_lines = []
    cited_nodes = set()
    for e in all_edges[:80]:  # cap context size
        conf = e.get("confidence", 0.6)
        if conf < 0.3:
            continue
        prov = f" [src: {e['provenance'][:60]}]" if e.get("provenance") else ""
        line = f"{e.get('src', '?')} --[{e['relation']}]--> {e['neighbor']} (conf: {conf:.2f}){prov}"
        context_lines.append(line)
        cited_nodes.add(e.get("src", "?"))
        cited_nodes.add(e["neighbor"])

    # Patch: get_edges_with_meta doesn't return src — add it from query context
    # Re-build with proper src tracking
    context_lines = []
    cited_nodes = set()
    for concept in query_concepts:
        edges = graph.get_edges_with_meta(concept, limit=15)
        for e in edges:
            conf = e.get("confidence", 0.6)
            if conf < 0.3:
                continue
            prov = f" [src: {e['provenance'][:60]}]" if e.get("provenance") else ""
            line = f"{concept} --[{e['relation']}]--> {e['neighbor']} (conf: {conf:.2f}){prov}"
            context_lines.append(line)
            cited_nodes.add(concept)
            cited_nodes.add(e["neighbor"])
        if max_hops >= 2:
            for e in edges[:5]:
                nb = e["neighbor"]
                hop2 = graph.get_edges_with_meta(nb, limit=8)
                for e2 in hop2:
                    conf2 = e2.get("confidence", 0.6)
                    if conf2 < 0.3:
                        continue
                    line2 = f"{nb} --[{e2['relation']}]--> {e2['neighbor']} (conf: {conf2:.2f})"
                    context_lines.append(line2)
                    cited_nodes.add(nb)
                    cited_nodes.add(e2["neighbor"])

    context_lines = list(dict.fromkeys(context_lines))[:80]
    graph_context = "\n".join(context_lines)

    # Step 4: Ollama grounded answer
    answer = "(Ollama unavailable — graph context only)"
    try:
        from ollama_integration import check_ollama_available, _ollama_request
        if check_ollama_available():
            system = (
                "You are a knowledge graph reasoning assistant. "
                "Answer questions using ONLY the provided graph context. "
                "Cite the specific concepts you use in your answer. "
                "If the graph context does not contain enough information, say so."
            )
            prompt = f"Graph context:\n{graph_context}\n\nQuestion: {q}\n\nAnswer (cite nodes used):"
            answer = _ollama_request(prompt, system=system, timeout=60)
    except ImportError:
        pass
    except Exception as e:
        logger.warning("Ollama Q&A failed: %s", e)

    return {
        "question": q,
        "answer": answer,
        "cited_nodes": sorted(cited_nodes),
        "context_edges": len(context_lines),
        "query_concepts": query_concepts,
    }


# ── Phase 3.2: Causal chain tracer ───────────────────────────────────────────

@app.get("/trace")
def trace_causal_chain(
    source: str = Query(..., description="Source concept"),
    target: str = Query(..., description="Target concept"),
    max_depth: int = Query(5, description="Maximum path depth"),
    _=Depends(verify_auth),
):
    """
    Trace a causal chain from source to target through causal relations.
    Traverses: causes, leads_to, produces, affects, influences, results_in.
    """
    graph = ConceptGraph()
    core = CoreEngine(graph)
    chain = core.find_causal_chain(source, target, max_depth=max_depth)
    if chain is None:
        return {
            "source": source,
            "target": target,
            "found": False,
            "message": f"No causal path found from '{source}' to '{target}' within depth {max_depth}.",
        }
    return {
        "source": source,
        "target": target,
        "found": True,
        "path": chain["path"],
        "relations": chain["relations"],
        "confidence_product": chain["confidence_product"],
        "explanation": " → ".join(
            f"{chain['path'][i]} ({chain['relations'][i]})"
            for i in range(len(chain["relations"]))
        ) + f" → {chain['path'][-1]}",
    }


# ── Phase 1.2 + Phase 3.3: Contradictions + Bridge nodes ─────────────────────

@app.get("/contradictions")
def get_contradictions(
    sample_size: int = Query(300, description="Edge sample size for semantic analysis"),
    _=Depends(verify_auth),
):
    """
    Returns semantically contradictory edge pairs, sorted by severity.
    Includes both structural (bidirectional causes) and semantic (embedding opposition) contradictions.
    """
    graph = ConceptGraph()
    core = CoreEngine(graph)
    structural = core.constraint_resolution()
    semantic = core.semantic_contradiction_detection(sample_size=sample_size)

    # Deduplicate by node pair
    all_contradictions = []
    seen = set()
    for c in structural:
        key = tuple(sorted(c["nodes"]))
        if key not in seen:
            seen.add(key)
            all_contradictions.append({**c, "detection": "structural"})
    for c in semantic:
        key = tuple(sorted(c["nodes"]))
        if key not in seen:
            seen.add(key)
            all_contradictions.append({**c, "detection": "semantic"})

    return {
        "total": len(all_contradictions),
        "structural": len(structural),
        "semantic": len(semantic),
        "contradictions": all_contradictions[:50],
    }


@app.get("/bridges")
def get_bridges(
    top_n: int = Query(10, description="Number of top bridge nodes to return"),
    _=Depends(verify_auth),
):
    """
    Returns nodes that bridge otherwise disconnected semantic clusters.
    High bridge_score = high betweenness centrality + diverse neighbor embeddings.
    """
    graph = ConceptGraph()
    core = CoreEngine(graph)
    bridges = core.find_bridge_nodes(top_n=top_n)
    return {"total": len(bridges), "bridges": bridges}


# ── Phase 5.1: Metrics + Dashboard ───────────────────────────────────────────

@app.get("/metrics")
def metrics(_=Depends(verify_auth)):
    """Live metrics: graph state, coherence, confidence, hypothesis rate, bridges."""
    from src.eval import graph_coherence
    from src.hypothesis_generator import _load_success_log, _load_strategy_stats

    graph = ConceptGraph()
    core = CoreEngine(graph)
    coh = graph_coherence(graph)
    hyp_log = _load_success_log()
    strategy_stats = _load_strategy_stats()
    corroborated = sum(1 for e in hyp_log if e.get("corroborated"))
    accepted = sum(1 for e in hyp_log if e.get("accepted"))
    bridges = core.find_bridge_nodes(top_n=5)
    contradictions = core.constraint_resolution()

    return {
        "graph": {
            "nodes": coh["nodes"],
            "edges": coh["edges"],
            "density": coh["density"],
        },
        "coherence": {
            "semantic": coh.get("semantic_coherence", 0.0),
            "avg_confidence": coh.get("avg_confidence", 0.0),
            "conflicts": coh["conflicts"],
        },
        "hypotheses": {
            "total": len(hyp_log),
            "accepted": accepted,
            "corroborated": corroborated,
            "acceptance_rate": round(accepted / len(hyp_log), 3) if hyp_log else 0,
            "corroboration_rate": round(corroborated / len(hyp_log), 3) if hyp_log else 0,
        },
        "strategy_stats": strategy_stats,
        "top_bridges": bridges[:5],
        "recent_contradictions": len(contradictions),
    }


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(_=Depends(verify_auth)):
    """Minimal live metrics dashboard with Chart.js auto-refresh."""
    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>TS Knowledge Engine — Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
  body { font-family: monospace; background: #0d0d0d; color: #e0e0e0; margin: 0; padding: 20px; }
  h1 { color: #7fdbff; font-size: 1.4em; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; margin-top: 16px; }
  .card { background: #1a1a2e; border: 1px solid #333; border-radius: 8px; padding: 16px; }
  .card h3 { color: #7fdbff; margin: 0 0 8px; font-size: 0.9em; text-transform: uppercase; letter-spacing: 1px; }
  .metric { font-size: 2em; font-weight: bold; color: #00ff9d; }
  .sub { font-size: 0.8em; color: #888; margin-top: 4px; }
  .bridge { padding: 4px 0; border-bottom: 1px solid #333; font-size: 0.85em; }
  .bridge:last-child { border: none; }
  .refresh { color: #555; font-size: 0.75em; margin-top: 16px; }
  canvas { max-height: 160px; }
</style>
</head>
<body>
<h1>TS Cognitive Engine — Live Dashboard</h1>
<div class="grid" id="metrics-grid">Loading...</div>
<p class="refresh" id="refresh-ts">Refreshing every 30s...</p>
<script>
async function loadMetrics() {
  try {
    const r = await fetch('/metrics');
    const d = await r.json();
    const g = document.getElementById('metrics-grid');
    const bridges = (d.top_bridges || []).map(b =>
      `<div class="bridge">${b.node} <span style="color:#888">(score: ${b.bridge_score})</span></div>`
    ).join('') || '<div class="bridge">—</div>';
    g.innerHTML = `
      <div class="card"><h3>Graph Size</h3>
        <div class="metric">${d.graph.nodes.toLocaleString()}</div>
        <div class="sub">${d.graph.edges.toLocaleString()} edges &nbsp;|&nbsp; density ${d.graph.density}</div>
      </div>
      <div class="card"><h3>Semantic Coherence</h3>
        <div class="metric">${(d.coherence.semantic * 100).toFixed(1)}%</div>
        <div class="sub">Avg confidence ${(d.coherence.avg_confidence * 100).toFixed(1)}% &nbsp;|&nbsp; ${d.coherence.conflicts} conflicts</div>
      </div>
      <div class="card"><h3>Hypotheses</h3>
        <div class="metric">${d.hypotheses.corroborated} / ${d.hypotheses.total}</div>
        <div class="sub">Corroboration rate ${(d.hypotheses.corroboration_rate * 100).toFixed(1)}%</div>
      </div>
      <div class="card"><h3>Top Bridge Nodes</h3>${bridges}</div>
      <div class="card"><h3>Contradictions</h3>
        <div class="metric">${d.recent_contradictions}</div>
        <div class="sub">Structural conflicts detected</div>
      </div>
    `;
    document.getElementById('refresh-ts').textContent =
      'Last updated: ' + new Date().toLocaleTimeString();
  } catch(e) { document.getElementById('metrics-grid').textContent = 'Error loading metrics: ' + e; }
}
loadMetrics();
setInterval(loadMetrics, 30000);
</script>
</body>
</html>"""
    return HTMLResponse(content=html)
