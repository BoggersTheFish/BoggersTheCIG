"""
Subsystem 4: Hypothesis Generator
Selects nodes, explores neighbors, detects gaps, generates candidate links via LLM.
Tracks success rate for self-improvement.
"""
import json
import logging
import random
from pathlib import Path
from typing import List, Optional, Tuple

from src.config import EVAL_DIR
from src.concept_graph import ConceptGraph
from src.language_layer import extract_triples, _get_llm, _filter_harmful, CONFIDENCE_HYPOTHESIS

logger = logging.getLogger(__name__)

# Success rate tracking
_hypothesis_log: List[dict] = []
_SUCCESS_LOG = EVAL_DIR / "hypothesis_success.json"
_STRATEGY_LOG = EVAL_DIR / "strategy_stats.json"

# Adaptive strategy: auto-selects best based on corroboration rate
STRATEGIES = ["least_accessed", "high_degree", "random"]
_STRATEGY_ADAPT_AFTER = 50  # re-evaluate after this many recorded outcomes


def _load_success_log() -> List[dict]:
    if _SUCCESS_LOG.exists():
        try:
            return json.loads(_SUCCESS_LOG.read_text())
        except Exception:
            pass
    return []


def _save_success_log(entries: List[dict]):
    _SUCCESS_LOG.parent.mkdir(parents=True, exist_ok=True)
    _SUCCESS_LOG.write_text(json.dumps(entries[-1000:], indent=2))  # Keep last 1000


def _load_strategy_stats() -> dict:
    if _STRATEGY_LOG.exists():
        try:
            return json.loads(_STRATEGY_LOG.read_text())
        except Exception:
            pass
    return {s: {"attempts": 0, "corroborated": 0} for s in STRATEGIES}


def _save_strategy_stats(stats: dict):
    _STRATEGY_LOG.parent.mkdir(parents=True, exist_ok=True)
    _STRATEGY_LOG.write_text(json.dumps(stats, indent=2))


def get_best_strategy() -> str:
    """
    Return the strategy with the highest corroboration rate.
    Falls back to 'least_accessed' if insufficient data (<50 attempts per strategy).
    Re-evaluates every _STRATEGY_ADAPT_AFTER total outcomes.
    """
    stats = _load_strategy_stats()
    best = "least_accessed"
    best_rate = -1.0
    for strat, data in stats.items():
        attempts = data.get("attempts", 0)
        if attempts < 10:
            continue  # Not enough data for this strategy yet
        rate = data.get("corroborated", 0) / attempts
        if rate > best_rate:
            best_rate = rate
            best = strat
    return best


class HypothesisGenerator:
    """
    Generate hypotheses: detect gaps, prompt LLM for candidate links.
    """

    def __init__(self, graph: ConceptGraph = None):
        self.graph = graph or ConceptGraph()

    def select_node(self, strategy: str = "least_accessed") -> Optional[str]:
        """Select a node for hypothesis generation."""
        if self.graph._use_neo4j:
            if strategy == "least_accessed":
                q = """
                MATCH (n:Concept)
                RETURN n.name as name
                ORDER BY n.last_access ASC
                LIMIT 1
                """
            else:  # random or high_degree
                q = """
                MATCH (n:Concept)
                WITH n ORDER BY n.degree DESC
                RETURN n.name as name
                LIMIT 100
                """
            recs = self.graph.cypher(q)
            names = [r["name"] for r in recs]
        else:
            if strategy == "least_accessed":
                nodes = sorted(
                    self.graph._nx_graph.nodes(data=True),
                    key=lambda x: x[1].get("last_access", 0),
                )
                names = [n[0] for n in nodes[:10]]
            else:
                nodes = list(self.graph._nx_graph.nodes())
                names = random.sample(nodes, min(50, len(nodes))) if nodes else []
        return random.choice(names) if names else None

    def detect_gaps(self, node: str, neighbors: List[Tuple[str, str, float]]) -> List[Tuple[str, str]]:
        """
        Detect missing edges: concepts that appear in patterns but lack direct links.
        Returns list of (node, potential_link) pairs.
        """
        gaps = []
        neighbor_names = {n[0] for n in neighbors}
        # Semantic search for related concepts not yet connected
        similar = self.graph.semantic_search(node, top_k=20)
        for name, score in similar:
            if name != node and name not in neighbor_names and score > 0.3:
                gaps.append((node, name))
        return gaps[:5]  # Limit

    def generate_candidates(self, node: str, neighbors: List[Tuple[str, str, float]]) -> List[Tuple[str, str, str]]:
        """
        Use LLM to hypothesize new links for node based on neighbors.
        Returns list of (Subject, Relation, Object) as unverified.
        """
        neighbor_str = ", ".join([f"{n[0]} ({n[1]})" for n in neighbors[:10]])
        prompt_text = f"Concept: {node}. Neighbors: {neighbor_str}. Hypothesize 2-3 new related concepts and relations as triples (Subject, Relation, Object)."
        triples = extract_triples(prompt_text, use_llm=False)  # Rule-based is cheap
        llm = _get_llm()
        if llm:
            try:
                import torch
                tokenizer, model = llm
                full_prompt = f"""Based on concept "{node}" and its neighbors: {neighbor_str}
Suggest 2-3 new concept triples that could connect to this concept. Format each as (Subject, Relation, Object).
Output only the triples, one per line:"""
                inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=256)
                if hasattr(model, "device"):
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=150, do_sample=True, temperature=0.7)
                response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                import re
                for line in response.strip().split("\n"):
                    m = re.search(r"\(([^,]+),\s*([^,]+),\s*([^)]+)\)", line)
                    if m:
                        triples.append((m.group(1).strip(), m.group(2).strip(), m.group(3).strip()))
            except Exception as e:
                logger.debug("LLM hypothesis failed: %s", e)
        # Ensure node is in each triple
        result = []
        for s, r, o in triples:
            if node.lower() in s.lower() or node.lower() in o.lower():
                result.append((s, r, o))
            else:
                result.append((node, r, o))  # Attach to selected node
        return _filter_harmful(result)[:5]

    def run(
        self,
        strategy: str = "least_accessed",
        validate_evidence: bool = False,
    ) -> List[Tuple]:
        """
        Full pipeline: select node -> get neighbors -> detect gaps -> generate candidates.
        When validate_evidence=True, each candidate is tested against web search.
        Returns triples as (s, r, o, confidence) 4-tuples.
        """
        node = self.select_node(strategy)
        if not node:
            return []
        neighbors = self.graph.get_neighbors(node, limit=20)
        candidates = self.generate_candidates(node, neighbors)
        gaps = self.detect_gaps(node, neighbors)
        for _, g in gaps[:2]:
            candidates.append((node, "possibly_related_to", g))

        result = []
        for triple in candidates:
            s, r, o = triple[0], triple[1], triple[2]
            if validate_evidence:
                corroborated, conf, source = self.validate_with_evidence((s, r, o))
                self.record_outcome(
                    (s, r, o), accepted=corroborated,
                    corroborated=corroborated, evidence_source=source, strategy=strategy,
                )
                if corroborated:
                    result.append((s, r, o, conf))
                else:
                    result.append((s, r, o, CONFIDENCE_HYPOTHESIS))
            else:
                result.append((s, r, o, CONFIDENCE_HYPOTHESIS))
        return result

    def validate_with_evidence(
        self,
        triple: Tuple[str, str, str],
        max_searches: int = 3,
    ) -> Tuple[bool, float, str]:
        """
        Search for external evidence corroborating a hypothesis triple.
        Returns (corroborated: bool, confidence: float, evidence_source: str).

        Pipeline:
          1. Generate DuckDuckGo query from triple
          2. Extract triples from top results
          3. Accept if any result triple has semantic similarity > 0.55 with hypothesis
        """
        s, r, o = triple
        query = f'"{s}" {r.replace("_", " ")} "{o}"'
        try:
            from src.knowledge_ingest import _search_duckduckgo
            results = _search_duckduckgo(query, max_results=max_searches)
        except Exception as e:
            logger.debug("Evidence search failed: %s", e)
            return False, CONFIDENCE_HYPOTHESIS, ""

        if not results:
            return False, CONFIDENCE_HYPOTHESIS, ""

        # Extract triples from result snippets
        from src.language_layer import extract_triples_with_confidence
        evidence_triples = []
        for res in results:
            text = f"{res.get('title', '')} {res.get('body', '')}".strip()[:1500]
            if text:
                evidence_triples.extend(extract_triples_with_confidence(text, use_llm=False))

        if not evidence_triples:
            return False, CONFIDENCE_HYPOTHESIS, ""

        # Semantic similarity between hypothesis and evidence
        try:
            from src.concept_graph import _embed
            import numpy as np
            h_text = f"{s} {r} {o}"
            h_emb = np.array(_embed(h_text), dtype=np.float32)
            h_norm = np.linalg.norm(h_emb)

            best_sim = 0.0
            best_source = ""
            for es, er, eo, _ in evidence_triples:
                e_text = f"{es} {er} {eo}"
                e_emb = np.array(_embed(e_text), dtype=np.float32)
                e_norm = np.linalg.norm(e_emb)
                if h_norm > 0 and e_norm > 0:
                    sim = float(np.dot(h_emb, e_emb) / (h_norm * e_norm))
                    if sim > best_sim:
                        best_sim = sim
                        best_source = results[0].get("href", "")

            CORROBORATION_THRESHOLD = 0.55
            if best_sim >= CORROBORATION_THRESHOLD:
                # Confidence scales with evidence strength: 0.55 sim → 0.65 conf; 0.9 sim → 0.85 conf
                evidence_conf = round(min(0.85, 0.5 + best_sim * 0.4), 3)
                logger.info("Hypothesis corroborated (sim=%.3f): %s", best_sim, triple)
                return True, evidence_conf, best_source
        except Exception as e:
            logger.debug("Hypothesis embedding comparison failed: %s", e)

        return False, CONFIDENCE_HYPOTHESIS, ""

    def record_outcome(self, triple: Tuple[str, str, str], accepted: bool,
                       corroborated: bool = False, evidence_source: str = "",
                       strategy: str = "unknown"):
        """Record hypothesis outcome including evidence corroboration. Updates strategy stats."""
        entries = _load_success_log()
        entries.append({
            "triple": list(triple),
            "accepted": accepted,
            "corroborated": corroborated,
            "evidence_source": evidence_source,
            "strategy": strategy,
        })
        _save_success_log(entries)
        # Update per-strategy corroboration tracking
        if strategy in STRATEGIES:
            stats = _load_strategy_stats()
            stats[strategy]["attempts"] = stats[strategy].get("attempts", 0) + 1
            if corroborated:
                stats[strategy]["corroborated"] = stats[strategy].get("corroborated", 0) + 1
            _save_strategy_stats(stats)
