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
from src.language_layer import extract_triples, _get_llm, _filter_harmful

logger = logging.getLogger(__name__)

# Success rate tracking
_hypothesis_log: List[dict] = []
_SUCCESS_LOG = EVAL_DIR / "hypothesis_success.json"


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

    def run(self, strategy: str = "least_accessed") -> List[Tuple[str, str, str]]:
        """
        Full pipeline: select node -> get neighbors -> detect gaps -> generate candidates.
        Returns new triples to add (unverified).
        """
        node = self.select_node(strategy)
        if not node:
            return []
        neighbors = self.graph.get_neighbors(node, limit=20)
        candidates = self.generate_candidates(node, neighbors)
        gaps = self.detect_gaps(node, neighbors)
        for _, g in gaps[:2]:
            candidates.append((node, "possibly_related_to", g))
        return candidates

    def record_outcome(self, triple: Tuple[str, str, str], accepted: bool):
        """Record hypothesis outcome for self-improvement."""
        entries = _load_success_log()
        entries.append({"triple": list(triple), "accepted": accepted})
        _save_success_log(entries)
