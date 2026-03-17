"""
External knowledge ingestion from web search.
Uses DuckDuckGo (free), extracts triples via Ollama, ingests into concept graph.
Rate-limited and ethical. Respects robots.txt via conservative delays.
Self-generates search queries from graph gaps via Ollama (local, $0).
"""
import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

from src.config import (
    ENABLE_EXTERNAL_INGEST,
    INGEST_RATE_LIMIT_SEC,
    INGEST_MAX_RESULTS_PER_QUERY,
    QUERIES_PATH,
)
from src.concept_graph import ConceptGraph
from src.language_layer import _filter_harmful
from src.validation_engine import ValidationEngine

logger = logging.getLogger(__name__)

_INGEST_LOG = Path(__file__).resolve().parent.parent / "eval" / "knowledge_ingest.jsonl"


def _load_queries() -> List[str]:
    """Load search queries from config path. Returns manual + generated combined."""
    if not QUERIES_PATH.exists():
        return ["cognitive architecture", "knowledge graph"]
    try:
        data = json.loads(QUERIES_PATH.read_text())
        if isinstance(data, list):
            return data
        manual = data.get("manual", [])
        generated = data.get("generated", [])
        if manual or generated:
            return list(manual) + list(generated)
        return data.get("queries", ["cognitive architecture"])
    except Exception as e:
        logger.warning("Could not load queries: %s", e)
        return ["cognitive architecture"]


def _load_queries_structure() -> dict:
    """Load full queries.json structure for read/write."""
    if not QUERIES_PATH.exists():
        return {"manual": ["cognitive architecture", "knowledge graph"], "generated": [], "last_generated": None}
    try:
        data = json.loads(QUERIES_PATH.read_text())
        if isinstance(data, list):
            return {"manual": list(data), "generated": [], "last_generated": None}
        return {
            "manual": data.get("manual", []),
            "generated": data.get("generated", []),
            "last_generated": data.get("last_generated"),
        }
    except Exception as e:
        logger.warning("Could not load queries structure: %s", e)
        return {"manual": ["cognitive architecture"], "generated": [], "last_generated": None}


def _save_queries(manual: List[str], generated: List[str], last_generated: dict = None):
    """Save queries.json with manual + generated + metadata."""
    QUERIES_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {"manual": manual, "generated": generated}
    if last_generated:
        payload["last_generated"] = last_generated
    QUERIES_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _search_duckduckgo(query: str, max_results: int = 5) -> List[dict]:
    """Search via DuckDuckGo (free, no API key). Returns list of {title, body, href}."""
    try:
        from duckduckgo_search import DDGS
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title": r.get("title", ""),
                    "body": r.get("body", ""),
                    "href": r.get("href", ""),
                })
        return results
    except ImportError:
        logger.warning("duckduckgo_search not installed. pip install duckduckgo-search")
        return []
    except Exception as e:
        logger.warning("DuckDuckGo search failed: %s", e)
        return []


def ensure_queries_file() -> None:
    """Create data/queries.json with defaults if missing."""
    if not QUERIES_PATH.exists():
        _save_queries(
            manual=["cognitive architecture", "knowledge graph reasoning", "symbolic AI"],
            generated=[],
        )


def ingest_external_knowledge(
    queries: List[str] = None,
    max_per_query: int = None,
    use_ollama: bool = True,
    force: bool = False,
) -> dict:
    """
    Search external sources, extract triples via Ollama, validate, ingest.
    Returns stats: {queries_run, results_fetched, triples_extracted, triples_ingested, sources}.
    When force=True, runs even if ENABLE_EXTERNAL_INGEST is false (e.g. --ingest-external flag).
    """
    if not ENABLE_EXTERNAL_INGEST and not force:
        logger.info("External ingest disabled (ENABLE_EXTERNAL_INGEST=false). Use force=True or --ingest-external.")
        return {"queries_run": 0, "triples_ingested": 0, "skipped": "disabled"}
    ensure_queries_file()

    queries = queries or _load_queries()
    max_per_query = max_per_query or INGEST_MAX_RESULTS_PER_QUERY
    graph = ConceptGraph()
    val = ValidationEngine(graph)

    stats = {"queries_run": 0, "results_fetched": 0, "triples_extracted": 0, "triples_ingested": 0, "sources": []}

    try:
        from tqdm import tqdm
        query_iter = tqdm(queries[:10], desc="external ingest", unit="query")
    except ImportError:
        query_iter = queries[:10]

    for q in query_iter:
        time.sleep(INGEST_RATE_LIMIT_SEC)
        results = _search_duckduckgo(q, max_results=max_per_query)
        stats["queries_run"] += 1
        stats["results_fetched"] += len(results)

        items = [(r, f"{r.get('title', '')} {r.get('body', '')}"[:2000]) for r in results]
        items = [(r, t) for r, t in items if t.strip()]
        if not items:
            continue

        if use_ollama:
            try:
                from ollama_integration import extract_triples_from_text_batch, check_ollama_available
                if check_ollama_available():
                    texts = [t for _, t in items]
                    per_text = extract_triples_from_text_batch(texts, flatten=False)
                    for (r, _), triples in zip(items, per_text):
                        triples = _filter_harmful(triples)
                        stats["triples_extracted"] += len(triples)
                        valid = val.validate_batch(triples)
                        for t in valid:
                            graph.ingest_triples([t], source="external")
                            stats["triples_ingested"] += 1
                            stats["sources"].append({"query": q, "href": r.get("href", ""), "triple": list(t)})
                        _log_ingest(q, r.get("href", ""), triples, valid)
                    continue
            except ImportError:
                pass

        from src.language_layer import extract_triples
        for r, text in items:
            triples = extract_triples(text, use_llm=False)
            triples = _filter_harmful(triples)
            stats["triples_extracted"] += len(triples)
            valid = val.validate_batch(triples)
            for t in valid:
                graph.ingest_triples([t], source="external")
                stats["triples_ingested"] += 1
                stats["sources"].append({"query": q, "href": r.get("href", ""), "triple": list(t)})
            _log_ingest(q, r.get("href", ""), triples, valid)

    _INGEST_LOG.parent.mkdir(parents=True, exist_ok=True)
    if stats.get("triples_ingested", 0) > 0:
        try:
            from src.viz import export_coherence_metrics
            export_coherence_metrics()
        except Exception as e:
            logger.debug("Coherence metrics export failed: %s", e)
        try:
            from src.viz import auto_snapshot_graph
            auto_snapshot_graph(
                reason="after-ingest",
                change_desc=f"external ingest (+{stats['triples_ingested']} triples)",
            )
        except Exception as e:
            logger.debug("Snapshot after ingest failed: %s", e)
    return stats


def _log_ingest(query: str, href: str, triples: List, valid: List):
    """Append to ingest log with timestamp."""
    entry = {
        "ts": time.time(),
        "query": query,
        "href": href,
        "triples_extracted": len(triples),
        "triples_valid": len(valid),
    }
    with open(_INGEST_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")


def _get_graph_gaps_summary() -> str:
    """Build summary of low-degree nodes and contradictions for Ollama."""
    from src.core_engine import CoreEngine
    graph = ConceptGraph()
    core = CoreEngine(graph)

    low_degree = []
    if graph._use_neo4j:
        recs = graph.cypher(
            "MATCH (n:Concept) OPTIONAL MATCH (n)-[r:RELATES]-() WITH n, count(r) as deg WHERE deg <= 1 RETURN n.name as name LIMIT 30",
            {},
        )
        low_degree = [r["name"] for r in recs]
    else:
        import networkx as nx
        G = graph._nx_graph
        for n in list(G.nodes())[:100]:
            if G.degree(n) <= 1:
                low_degree.append(n)
            if len(low_degree) >= 30:
                break

    conflicts = core.constraint_resolution()
    if graph._use_neo4j:
        recs = graph.cypher("MATCH (n:Concept) RETURN n.name as name LIMIT 20", {})
        sample_nodes = [r["name"] for r in recs]
    else:
        sample_nodes = list(graph._nx_graph.nodes())[:20]

    lines = [
        f"Nodes: {graph.node_count()}, Edges: {graph.edge_count()}",
        f"Low-degree nodes (degree<=1): {low_degree[:15]}",
        f"Contradictions: {[c.get('type') for c in conflicts[:5]]}",
        f"Sample concepts: {sample_nodes[:15]}",
    ]
    return "\n".join(lines)


def generate_queries_from_graph(use_ollama: bool = True, reason: str = "graph-gaps") -> dict:
    """
    Use Ollama to suggest 5-10 high-value search queries from graph gaps.
    Saves to data/queries.json (appends/overwrites generated, keeps manual).
    Returns {queries: [...], saved: bool, reason: str}.
    """
    structure = _load_queries_structure()
    manual = structure["manual"]
    gaps = _get_graph_gaps_summary()

    if use_ollama:
        try:
            from ollama_integration import check_ollama_available, _ollama_request
            if not check_ollama_available():
                logger.info("Ollama not available, using fallback queries")
                return {"queries": manual[:5], "saved": False, "reason": "ollama-unavailable"}
            system = """You are a TS-aligned knowledge curator. Analyze current graph gaps (low-degree nodes, contradictions).
Suggest 5-10 high-value DuckDuckGo/X search queries to fill them and increase coherence.
Output ONLY one query per line, no numbering, no bullets. Short, searchable phrases (2-6 words each).
No harmful, violent, or discriminatory content."""
            prompt = f"""Graph gaps:\n{gaps}\n\nSuggested search queries (one per line):"""
            resp = _ollama_request(prompt, system=system, timeout=60)
            raw = resp.strip()
            queries = []
            for line in raw.split("\n"):
                line = re.sub(r"^[\d\.\-\*]\s*", "", line.strip())
                if line and len(line) > 3 and len(line) < 120:
                    if not any(p in line.lower() for p in HARMFUL_PATTERNS):
                        queries.append(line)
            queries = list(dict.fromkeys(queries))[:10]
            if not queries:
                queries = manual[:5]
        except ImportError:
            queries = manual[:5]
            logger.info("ollama_integration not available, using manual fallback")
        except Exception as e:
            logger.warning("Ollama query generation failed: %s", e)
            queries = manual[:5]
    else:
        queries = manual[:5]

    structure["generated"] = queries
    ts = datetime.now(timezone.utc).isoformat()
    structure["last_generated"] = {"ts": ts, "reason": reason, "queries": queries}
    _save_queries(manual=manual, generated=queries, last_generated=structure["last_generated"])
    logger.info("Generated %d queries from graph gaps, saved to %s", len(queries), QUERIES_PATH)
    return {"queries": queries, "saved": True, "reason": reason}
