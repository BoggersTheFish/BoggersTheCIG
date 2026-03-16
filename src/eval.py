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


def graph_coherence(graph: ConceptGraph) -> Dict:
    """Compute density, conflict count."""
    core = CoreEngine(graph)
    conflicts = core.constraint_resolution()
    n = graph.node_count()
    e = graph.edge_count()
    density = (2 * e / (n * (n - 1))) if n > 1 else 0
    return {
        "nodes": n,
        "edges": e,
        "density": round(density, 4),
        "conflicts": len(conflicts),
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
