"""
Full-TS Cognitive Architecture - Main Entry Point
Usage:
  python src/main.py --input "Explain quantum physics"
  python src/main.py --input "Gravity bends spacetime" --ingest-only
  python src/main.py --mode=continuous
"""
import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on path when run as "python src/main.py"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROJECT_ROOT

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Full-TS Cognitive Architecture")
    parser.add_argument("--input", "-i", type=str, help="Input text to process")
    parser.add_argument("--ingest-only", action="store_true", help="Only ingest, no reasoning")
    parser.add_argument("--mode", type=str, default="single", choices=["single", "continuous"],
                        help="Single run or continuous loop")
    parser.add_argument("--interval", "--sleep", type=float, default=60.0, dest="interval",
                        help="Loop interval in seconds for continuous mode (default: 60)")
    parser.add_argument("--evolve-every", type=int, default=None,
                        help="Run full self-improver cycle every N iterations (continuous mode)")
    parser.add_argument("--self-improve", action="store_true",
                        help="Run one self-improvement cycle and exit")
    parser.add_argument("--skip-tests", action="store_true",
                        help="Skip pytest in self-improve cycle (faster)")
    parser.add_argument("--ingest-external", action="store_true",
                        help="Run external knowledge ingest in self-improve (requires ENABLE_EXTERNAL_INGEST=true)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Simulate self-improve cycle without committing")
    parser.add_argument("--notify-on-change", action="store_true",
                        help="Notify (webhook) only if meaningful changes occurred")
    parser.add_argument("--safe-evolve", action="store_true",
                        help="Backup before commit, revert if coherence drops >10% or tests fail")
    parser.add_argument("--generate-queries", action="store_true",
                        help="Generate search queries from graph gaps via Ollama, save to data/queries.json")
    parser.add_argument("--analyze-vault", action="store_true",
                        help="Analyze Obsidian vault with Ollama, extract concepts, ingest into graph")
    parser.add_argument("--no-ollama", action="store_true",
                        help="Skip Ollama in analyze-vault (use rule-based extraction only)")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM, use rule-based extraction only")
    args = parser.parse_args()

    if args.generate_queries and not args.self_improve:
        from src.knowledge_ingest import generate_queries_from_graph
        result = generate_queries_from_graph(use_ollama=not args.no_ollama)
        print("\n--- Generated Queries ---")
        for k, v in result.items():
            print(f"  {k}: {v}")
        return

    if args.analyze_vault:
        from src.obsidian_ollama_bridge import analyze_vault
        stats = analyze_vault(use_ollama=not args.no_ollama)
        print("\n--- Vault Analysis ---")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        return

    if args.self_improve:
        from src.self_improver import run_one_cycle
        stats = run_one_cycle(
            skip_ollama=True,
            skip_git_push=True,
            skip_tests=args.skip_tests,
            ingest_external=args.ingest_external,
            generate_queries=getattr(args, "generate_queries", False),
            dry_run=getattr(args, "dry_run", False),
            notify_on_change=getattr(args, "notify_on_change", False),
            safe_evolve=getattr(args, "safe_evolve", False),
        )
        print("\n--- Self-Improvement Cycle ---")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        return

    if args.mode == "continuous":
        from src.continuous_thinker import run_continuous
        run_continuous(interval_seconds=args.interval, evolve_every=args.evolve_every)
        return

    # Single mode
    if not args.input:
        parser.print_help()
        print("\nExample: python src/main.py --input 'Explain quantum physics'")
        return

    from src.language_layer import extract_triples
    from src.concept_graph import ConceptGraph
    from src.core_engine import CoreEngine
    from src.viz import export_to_obsidian, auto_snapshot_graph

    text = args.input
    logger.info("Input: %s", text[:100] + "..." if len(text) > 100 else text)

    # Extract triples
    triples = extract_triples(text, use_llm=not args.no_llm)
    logger.info("Extracted %d triples: %s", len(triples), triples)

    if not triples:
        logger.warning("No triples extracted. Try different phrasing or --no-llm for rule-based fallback.")

    # Ingest
    graph = ConceptGraph()
    count = graph.ingest_triples(triples, source="main")
    logger.info("Ingested %d edges", count)

    if args.ingest_only:
        logger.info("Ingest-only mode. Done.")
        return

    # Reasoning
    core = CoreEngine(graph)
    search = core.structural_search({"relation_type": None}, limit=10)
    conflicts = core.constraint_resolution()
    logger.info("Structural search: %d results", len(search))
    logger.info("Conflicts: %s", conflicts)

    # Semantic search
    similar = graph.semantic_search(text[:50], top_k=5)
    logger.info("Semantic search: %s", similar)

    # Export to Obsidian vault (visual memory layer)
    export_to_obsidian(graph)
    logger.info("Exported to Obsidian vault")
    try:
        from src.viz import export_coherence_metrics
        export_coherence_metrics()
    except Exception as e:
        logger.debug("Coherence metrics failed: %s", e)
    try:
        auto_snapshot_graph(reason="after-export", change_desc="single-run export")
    except Exception as e:
        logger.debug("Snapshot failed: %s", e)

    print("\n--- Result ---")
    print(f"Triples: {triples}")
    print(f"Graph: {graph.node_count()} nodes, {graph.edge_count()} edges")
    print(f"Similar concepts: {[s[0] for s in similar]}")


if __name__ == "__main__":
    main()
