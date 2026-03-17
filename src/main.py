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

# Hardware detection + Ollama model selection (before config import)
if "--help" not in sys.argv and "-h" not in sys.argv:
    _early_parser = argparse.ArgumentParser()
    _early_parser.add_argument("--force-model", type=str, default=None, help="Override auto-detection (e.g. llama3.1:70b)")
    _early_parser.add_argument("--parallel-threads", type=int, default=None, help="Parallel Ollama instances (default: CPU cores // 2)")
    _early_args, _ = _early_parser.parse_known_args()
    if _early_args.parallel_threads is not None:
        import os
        os.environ["PARALLEL_THREADS"] = str(_early_args.parallel_threads)
    from src.hardware_adapt import detect_and_set_model
    detect_and_set_model(force_model=_early_args.force_model)

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
                        help="Backup before commit, revert if coherence drops >10%% or tests fail")
    parser.add_argument("--skip-git-push", action="store_true", dest="skip_git_push",
                        help="Disable commit & push in self-improve (default: push)")
    parser.add_argument("--ollama", action="store_true",
                        help="Enable Ollama analysis in self-improve (default: skip for speed)")
    parser.add_argument("--generate-queries", action="store_true",
                        help="Generate search queries from graph gaps via Ollama, save to data/queries.json")
    parser.add_argument("--analyze-vault", action="store_true",
                        help="Analyze Obsidian vault with Ollama, extract concepts, ingest into graph")
    parser.add_argument("--organize-vault", action="store_true",
                        help="Auto-organize vault: cluster Concepts by embeddings+Ollama, create folders, move files")
    parser.add_argument("--extract-subideas", action="store_true",
                        help="Extract sub-ideas into hierarchical files, merge duplicates into Shared-Sub-ideas")
    parser.add_argument("--no-ollama", action="store_true",
                        help="Skip Ollama in analyze-vault (use rule-based extraction only)")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM, use rule-based extraction only")
    parser.add_argument("--no-auto-commit", action="store_true",
                        help="Disable auto-commit/push after analyze-vault, organize-vault, generate-queries")
    parser.add_argument("--force-model", type=str, default=None,
                        help="Override auto-detection (e.g. llama3.1:70b, qwen2.5-coder:7b)")
    parser.add_argument("--parallel-threads", type=int, default=None,
                        help="Parallel Ollama threads (default: CPU cores // 2)")
    args = parser.parse_args()

    auto_commit = not getattr(args, "no_auto_commit", False)

    if args.generate_queries and not args.self_improve:
        from src.knowledge_ingest import generate_queries_from_graph
        result = generate_queries_from_graph(use_ollama=not args.no_ollama)
        print("\n--- Generated Queries ---")
        for k, v in result.items():
            print(f"  {k}: {v}")
        if auto_commit:
            from src.self_improver import auto_commit_if_changes
            ac = auto_commit_if_changes("generate-queries", require_coherence=False)
            if ac.get("commit_ok"):
                print(f"  Auto-committed: {ac.get('commit_hash', '?')}")
        return

    if args.analyze_vault:
        from src.obsidian_ollama_bridge import analyze_vault
        stats = analyze_vault(use_ollama=not args.no_ollama)
        print("\n--- Vault Analysis ---")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        if auto_commit:
            from src.self_improver import auto_commit_if_changes
            ac = auto_commit_if_changes("analyze-vault")
            if ac.get("commit_ok"):
                print(f"  Auto-committed: {ac.get('commit_hash', '?')}")
        return

    if args.organize_vault:
        from src.obsidian_filesystem_manager import auto_organize_vault
        result = auto_organize_vault(force=True)
        print("\n--- Vault Organization ---")
        for k, v in result.items():
            print(f"  {k}: {v}")
        if auto_commit:
            from src.self_improver import auto_commit_if_changes
            ac = auto_commit_if_changes("organize-vault")
            if ac.get("commit_ok"):
                print(f"  Auto-committed: {ac.get('commit_hash', '?')}")
        return

    if args.extract_subideas:
        from src.obsidian_filesystem_manager import auto_extract_and_merge_subideas
        result = auto_extract_and_merge_subideas(use_ollama=not args.no_ollama)
        print("\n--- Extract & Merge Sub-ideas ---")
        for k, v in result.items():
            print(f"  {k}: {v}")
        if auto_commit:
            from src.self_improver import auto_commit_if_changes
            ac = auto_commit_if_changes("extract-subideas")
            if ac.get("commit_ok"):
                print(f"  Auto-committed: {ac.get('commit_hash', '?')}")
        return

    if args.self_improve:
        from src.self_improver import run_one_cycle
        stats = run_one_cycle(
            skip_ollama=not getattr(args, "ollama", False),
            skip_git_push=getattr(args, "skip_git_push", False),
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
    from src.validation_engine import ValidationEngine
    from src.viz import export_to_obsidian, auto_snapshot_graph

    text = args.input
    logger.info("Input: %s", text[:100] + "..." if len(text) > 100 else text)

    # Extract triples
    triples = extract_triples(text, use_llm=not args.no_llm)
    logger.info("Extracted %d triples: %s", len(triples), triples)

    if not triples:
        logger.warning("No triples extracted. Try different phrasing or --no-llm for rule-based fallback.")

    # Ingest (validate before mutation per graph-stability)
    graph = ConceptGraph()
    val = ValidationEngine(graph)
    valid = val.validate_batch(triples)
    count = graph.ingest_triples(valid, source="main")
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
        from src.obsidian_filesystem_manager import auto_extract_and_merge_subideas
        auto_extract_and_merge_subideas(use_ollama=not args.no_llm)
    except Exception as e:
        logger.debug("Sub-idea extract/merge failed: %s", e)
    try:
        auto_snapshot_graph(reason="after-export", change_desc="single-run export")
    except Exception as e:
        logger.debug("Snapshot failed: %s", e)

    print("\n--- Result ---")
    print(f"Triples: {valid}")
    print(f"Graph: {graph.node_count()} nodes, {graph.edge_count()} edges")
    print(f"Similar concepts: {[s[0] for s in similar]}")


if __name__ == "__main__":
    main()
