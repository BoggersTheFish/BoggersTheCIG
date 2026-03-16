"""
Subsystem 6: Continuous Thinking Loop
Celery tasks + main loop: select node -> reason -> hypothesize -> validate -> update.
"""
import json
import logging
import time
from typing import Optional

from src.config import EVAL_DIR, SELF_IMPROVE_INTERVAL
from src.concept_graph import ConceptGraph
from src.core_engine import CoreEngine
from src.hypothesis_generator import HypothesisGenerator
from src.validation_engine import ValidationEngine
from src.self_improve import SelfImprover
from src.viz import export_to_obsidian

logger = logging.getLogger(__name__)

_loop_count = 0


def run_loop_iteration(graph: Optional[ConceptGraph] = None) -> dict:
    """
    Single iteration: select node -> core reasoning -> generate hypothesis -> validate -> ingest.
    Returns stats dict.
    """
    global _loop_count
    _loop_count += 1
    graph = graph or ConceptGraph()
    core = CoreEngine(graph)
    hyp = HypothesisGenerator(graph)
    val = ValidationEngine(graph)

    stats = {"iter": _loop_count, "hypotheses_generated": 0, "hypotheses_accepted": 0, "pruned": 0}

    # 1. Structural search (warm-up)
    core.structural_search({"relation_type": None}, limit=5)

    # 2. Constraint resolution
    conflicts = core.constraint_resolution()
    if conflicts:
        stats["pruned"] = val.prune_invalid()

    # 3. Generate hypotheses
    candidates = hyp.run(strategy="least_accessed")
    stats["hypotheses_generated"] = len(candidates)

    # 4. Validate and ingest
    valid = val.validate_batch(candidates)
    stats["hypotheses_accepted"] = len(valid)
    for t in valid:
        graph.ingest_triples([t], source="hypothesis")
        hyp.record_outcome(t, True)
    for t in candidates:
        if t not in valid:
            hyp.record_outcome(t, False)

    # 5. Export graph to Obsidian vault (visual memory layer)
    export_to_obsidian(graph)

    # 6. Self-improve every N iterations (meta-rules)
    if _loop_count % SELF_IMPROVE_INTERVAL == 0:
        si = SelfImprover()
        si.run()

    return stats


def run_continuous(interval_seconds: float = 60.0, max_iters: Optional[int] = None, evolve_every: Optional[int] = None):
    """
    Main continuous loop. Runs until interrupted or max_iters.
    If evolve_every is set, runs full self_improver cycle every N iterations.
    """
    log_path = EVAL_DIR / "continuous_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    graph = ConceptGraph()
    iters = 0
    logger.info("Starting continuous thinking loop (interval=%.1fs, evolve_every=%s)", interval_seconds, evolve_every)
    try:
        while max_iters is None or iters < max_iters:
            try:
                stats = run_loop_iteration(graph)
                iters += 1
                with open(log_path, "a") as f:
                    f.write(json.dumps(stats) + "\n")
                logger.info("Loop %d: generated=%d accepted=%d pruned=%d",
                            stats["iter"], stats["hypotheses_generated"],
                            stats["hypotheses_accepted"], stats["pruned"])
                if evolve_every and iters % evolve_every == 0:
                    from src.self_improver import run_one_cycle
                    run_one_cycle(skip_ollama=True, skip_git_push=True)
            except Exception as e:
                logger.exception("Loop iteration failed: %s", e)
            time.sleep(interval_seconds)
    except KeyboardInterrupt:
        logger.info("Continuous loop stopped by user")
