"""
Subsystem 13: Evaluation and Iteration
Metrics: graph coherence, hypothesis quality, bias scores.
"""
import json
import logging
from pathlib import Path
from typing import Dict, List

from src.config import EVAL_DIR
from src.concept_graph import ConceptGraph
from src.core_engine import CoreEngine

logger = logging.getLogger(__name__)


def semantic_coherence(graph: ConceptGraph, sample_size: int = 300) -> float:
    """
    Compute average cosine similarity between connected nodes' embeddings.
    Range [0, 1]. Higher = semantically coherent connections.
    Only edges with confidence >= 0.4 are considered.
    Returns 0.0 if embeddings unavailable.
    """
    try:
        import numpy as np
        if graph._use_neo4j:
            recs = graph.cypher(
                """
                MATCH (a:Concept)-[e:RELATES]->(b:Concept)
                WHERE e.confidence >= 0.4
                RETURN a.embedding as ea, b.embedding as eb
                LIMIT $lim
                """,
                {"lim": sample_size},
            )
            pairs = [(r["ea"], r["eb"]) for r in recs if r.get("ea") and r.get("eb")]
        else:
            G = graph._nx_graph
            pairs = []
            for u, v, d in list(G.edges(data=True))[:sample_size]:
                if d.get("confidence", 0.6) < 0.4:
                    continue
                eu = G.nodes[u].get("embedding")
                ev = G.nodes[v].get("embedding")
                if eu and ev:
                    pairs.append((eu, ev))
        if not pairs:
            return 0.0
        sims = []
        for ea, eb in pairs:
            a, b = np.array(ea, dtype=np.float32), np.array(eb, dtype=np.float32)
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na > 0 and nb > 0:
                sims.append(float(np.dot(a, b) / (na * nb)))
        return round(sum(sims) / len(sims), 4) if sims else 0.0
    except Exception as e:
        logger.debug("Semantic coherence failed: %s", e)
        return 0.0


def graph_coherence(graph: ConceptGraph) -> Dict:
    """Compute density, conflict count, semantic coherence, and avg confidence."""
    core = CoreEngine(graph)
    conflicts = core.constraint_resolution()
    n = graph.node_count()
    e = graph.edge_count()
    density = (2 * e / (n * (n - 1))) if n > 1 else 0
    sem_coh = semantic_coherence(graph)
    avg_conf = graph.avg_confidence(min_confidence=0.0)
    return {
        "nodes": n,
        "edges": e,
        "density": round(density, 4),
        "conflicts": len(conflicts),
        "semantic_coherence": sem_coh,
        "avg_confidence": round(avg_conf, 4),
    }


def hypothesis_quality(log_path: Path = None) -> Dict:
    """Compute acceptance rate from hypothesis log."""
    path = log_path or (EVAL_DIR / "hypothesis_success.json")
    if not path.exists():
        return {"acceptance_rate": 0, "total": 0}
    try:
        data = json.loads(path.read_text())
        accepted = sum(1 for e in data if e.get("accepted"))
        return {"acceptance_rate": accepted / len(data) if data else 0, "total": len(data)}
    except Exception:
        return {"acceptance_rate": 0, "total": 0}


def run_eval() -> Dict:
    """Full evaluation run."""
    graph = ConceptGraph()
    coherence = graph_coherence(graph)
    hyp = hypothesis_quality()
    result = {"coherence": coherence, "hypothesis": hyp}
    out = EVAL_DIR / "eval_result.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    logger.info("Eval: %s", result)
    return result
