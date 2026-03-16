"""
Subsystem 3: TS Core Reasoning Engine
Structural search, pattern discovery, compression, constraint resolution.
"""
import logging
from typing import Any, Dict, List, Optional, Tuple

from src.config import SIMILARITY_THRESHOLD
from src.concept_graph import ConceptGraph, _embed

logger = logging.getLogger(__name__)


class CoreEngine:
    """
    Core reasoning: structural search, pattern discovery,
    structural compression, constraint resolution.
    """

    def __init__(self, graph: Optional[ConceptGraph] = None):
        self.graph = graph or ConceptGraph()

    def structural_search(self, constraints: Dict[str, Any], limit: int = 20) -> List[Dict]:
        """
        Cypher-based structural search for patterns matching constraints.
        E.g. MATCH (a)-[r]->(b) WHERE r.type = 'causes'
        """
        if not self.graph._use_neo4j:
            return self._structural_search_nx(constraints, limit)
        rel_type = constraints.get("relation_type")
        if rel_type:
            q = """
            MATCH (a:Concept)-[r:RELATES]->(b:Concept)
            WHERE r.type = $rel
            RETURN a.name as subject, b.name as object, r.type as rel, r.weight as weight
            ORDER BY r.weight DESC LIMIT $limit
            """
            return self.graph.cypher(q, {"rel": rel_type, "limit": limit})
        q = """
        MATCH (a:Concept)-[r:RELATES]->(b:Concept)
        RETURN a.name as subject, b.name as object, r.type as rel, r.weight as weight
        ORDER BY r.weight DESC LIMIT $limit
        """
        return self.graph.cypher(q, {"limit": limit})

    def _structural_search_nx(self, constraints: Dict[str, Any], limit: int) -> List[Dict]:
        """NetworkX fallback for structural search."""
        results = []
        rel_type = constraints.get("relation_type")
        for u, v, data in self.graph._nx_graph.edges(data=True):
            rtype = data.get("type", "relates")
            if rel_type and rtype != rel_type:
                continue
            results.append({
                "subject": u, "object": v, "rel": rtype,
                "weight": data.get("weight", 1),
            })
        return sorted(results, key=lambda x: -x["weight"])[:limit]

    def pattern_discovery(self, min_pattern_size: int = 3) -> List[List[str]]:
        """
        Find motifs / recurring patterns (e.g. triangles, chains).
        Uses subgraph analysis.
        """
        if self.graph._use_neo4j:
            # Find paths of length 2 (A->B->C)
            q = """
            MATCH path = (a:Concept)-[:RELATES*2]->(c:Concept)
            WHERE a <> c
            RETURN [node in nodes(path) | node.name] as path
            LIMIT 100
            """
            recs = self.graph.cypher(q)
            return [r["path"] for r in recs if len(r["path"]) >= min_pattern_size]
        import networkx as nx
        G = self.graph._nx_graph
        patterns = []
        for node in list(G.nodes())[:500]:  # Limit for performance
            try:
                for path in nx.all_simple_paths(G, node, list(G.successors(node))[0] if G.successors(node) else node, cutoff=2):
                    if len(path) >= min_pattern_size:
                        patterns.append(path)
            except (nx.NetworkXNoPath, StopIteration):
                pass
        return patterns[:100]

    def structural_compression(self, similarity_threshold: float = None) -> int:
        """
        Merge redundant nodes (embedding similarity > threshold).
        Returns count of merges.
        """
        thresh = similarity_threshold or SIMILARITY_THRESHOLD
        if self.graph._use_neo4j:
            return self._compress_neo4j(thresh)
        return self._compress_nx(thresh)

    def _compress_neo4j(self, thresh: float) -> int:
        """Merge similar nodes in Neo4j."""
        nodes = self.graph.cypher("MATCH (n:Concept) RETURN n.name as name, n.embedding as emb")
        merges = 0
        seen = set()
        for i, n1 in enumerate(nodes):
            if n1["name"] in seen:
                continue
            for n2 in nodes[i + 1:]:
                if n2["name"] in seen:
                    continue
                emb1, emb2 = n1.get("emb"), n2.get("emb")
                if not emb1 or not emb2:
                    continue
                sim = _cosine_sim(emb1, emb2)
                if sim >= thresh:
                    # Merge n2 into n1
                    self.graph.cypher(
                        """
                        MATCH (a:Concept {name: $keep}), (b:Concept {name: $remove})
                        MATCH (b)-[r:RELATES]->(x) WHERE x <> a
                        MERGE (a)-[e:RELATES {type: r.type}]->(x)
                        ON CREATE SET e.weight = r.weight ON MATCH SET e.weight = e.weight + r.weight
                        WITH b
                        MATCH (y)-[r2:RELATES]->(b) WHERE y <> a
                        MERGE (y)-[e2:RELATES {type: r2.type}]->(a)
                        ON CREATE SET e2.weight = r2.weight ON MATCH SET e2.weight = e2.weight + r2.weight
                        WITH b
                        DETACH DELETE b
                        """,
                        {"keep": n1["name"], "remove": n2["name"]},
                    )
                    seen.add(n2["name"])
                    merges += 1
                    break
        return merges

    def _compress_nx(self, thresh: float) -> int:
        """Merge similar nodes in NetworkX."""
        import networkx as nx
        G = self.graph._nx_graph
        nodes = list(G.nodes())
        merges = 0
        to_remove = set()
        for i, n1 in enumerate(nodes):
            if n1 in to_remove:
                continue
            emb1 = G.nodes[n1].get("embedding", [0.0] * 384)
            for n2 in nodes[i + 1:]:
                if n2 in to_remove:
                    continue
                emb2 = G.nodes[n2].get("embedding", [0.0] * 384)
                if _cosine_sim(emb1, emb2) >= thresh:
                    # Merge n2 into n1: redirect edges
                    for pred in list(G.predecessors(n2)):
                        if pred != n1:
                            w = G[pred][n2].get("weight", 1)
                            if G.has_edge(pred, n1):
                                G[pred][n1]["weight"] = G[pred][n1].get("weight", 0) + w
                            else:
                                G.add_edge(pred, n1, weight=w, type=G[pred][n2].get("type", "relates"))
                    for succ in list(G.successors(n2)):
                        if succ != n1:
                            w = G[n2][succ].get("weight", 1)
                            if G.has_edge(n1, succ):
                                G[n1][succ]["weight"] = G[n1][succ].get("weight", 0) + w
                            else:
                                G.add_edge(n1, succ, weight=w, type=G[n2][succ].get("type", "relates"))
                    G.remove_node(n2)
                    to_remove.add(n2)
                    merges += 1
                    break
        return merges

    def constraint_resolution(self) -> List[Dict]:
        """
        Detect contradictions (e.g. A causes B and B causes A with conflicting semantics).
        Returns list of detected conflicts.
        """
        conflicts = []
        if self.graph._use_neo4j:
            # Find bidirectional edges that might conflict (e.g. "causes" both ways)
            q = """
            MATCH (a:Concept)-[r1:RELATES]->(b:Concept), (b)-[r2:RELATES]->(a)
            WHERE r1.type = 'causes' AND r2.type = 'causes'
            RETURN a.name as a, b.name as b
            """
            for rec in self.graph.cypher(q):
                conflicts.append({"type": "bidirectional_causes", "nodes": [rec["a"], rec["b"]]})
        else:
            G = self.graph._nx_graph
            for u, v in G.edges():
                if G.has_edge(v, u):
                    d1, d2 = G[u][v], G[v][u]
                    if d1.get("type") == "causes" and d2.get("type") == "causes":
                        conflicts.append({"type": "bidirectional_causes", "nodes": [u, v]})
        return conflicts


def _cosine_sim(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two vectors."""
    try:
        import numpy as np
        x, y = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
        return float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-9))
    except Exception:
        return 0.0
