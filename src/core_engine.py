"""
Subsystem 3: TS Core Reasoning Engine
Structural search, pattern discovery, compression, constraint resolution.
"""
import logging
from collections import deque
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
        Detect structural contradictions (bidirectional causal cycles).
        Returns list of detected conflicts.
        """
        conflicts = []
        if self.graph._use_neo4j:
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

    def semantic_contradiction_detection(
        self, sample_size: int = 500, similarity_threshold: float = -0.15
    ) -> List[Dict]:
        """
        Detect semantically contradictory edges using embeddings.
        Two connected nodes are contradictory if:
          - Their embeddings have cosine similarity below similarity_threshold (semantically opposed)
          - OR the relation is an exclusive type and both directions exist

        Returns list of {type, nodes, relation, severity, confidence_a, confidence_b}.
        Severity = 1 - cosine_similarity (higher = more contradictory).
        """
        contradictions = []
        try:
            import numpy as np

            # Negation patterns that signal contradiction
            negation_relations = {
                "is_not", "not_a", "contradicts", "opposes", "prevents",
                "inhibits", "negates", "refutes", "disproves",
            }

            if self.graph._use_neo4j:
                recs = self.graph.cypher(
                    """
                    MATCH (a:Concept)-[e:RELATES]->(b:Concept)
                    RETURN a.name as sn, b.name as on, e.type as rel,
                           a.embedding as ea, b.embedding as eb,
                           e.confidence as conf
                    LIMIT $lim
                    """,
                    {"lim": sample_size},
                )
                edges = [
                    (r["sn"], r["on"], r["rel"], r.get("ea"), r.get("eb"), r.get("conf", 0.6))
                    for r in recs
                ]
            else:
                G = self.graph._nx_graph
                edges = []
                for u, v, d in list(G.edges(data=True))[:sample_size]:
                    eu = G.nodes[u].get("embedding")
                    ev = G.nodes[v].get("embedding")
                    edges.append((u, v, d.get("type", "relates"), eu, ev, d.get("confidence", 0.6)))

            for sn, on, rel, ea, eb, conf in edges:
                # Negation relation → always a contradiction signal
                if rel in negation_relations:
                    contradictions.append({
                        "type": "negation_relation",
                        "nodes": [sn, on],
                        "relation": rel,
                        "severity": 1.0,
                        "confidence_a": conf,
                        "confidence_b": conf,
                    })
                    continue

                # Embedding-based semantic opposition
                if ea and eb and len(ea) == len(eb):
                    a = np.array(ea, dtype=np.float32)
                    b = np.array(eb, dtype=np.float32)
                    na, nb = np.linalg.norm(a), np.linalg.norm(b)
                    if na > 0 and nb > 0:
                        sim = float(np.dot(a, b) / (na * nb))
                        if sim < similarity_threshold:
                            severity = round(1.0 - sim, 3)
                            contradictions.append({
                                "type": "semantic_opposition",
                                "nodes": [sn, on],
                                "relation": rel,
                                "similarity": round(sim, 3),
                                "severity": severity,
                                "confidence_a": conf,
                                "confidence_b": conf,
                            })

        except Exception as e:
            logger.warning("Semantic contradiction detection failed: %s", e)

        # Sort by severity descending
        return sorted(contradictions, key=lambda x: -x.get("severity", 0))

    def find_causal_chain(
        self,
        source: str,
        target: str,
        max_depth: int = 5,
    ) -> Optional[Dict]:
        """
        Find a directed causal path from source to target through causal relations.
        Traverses: causes, leads_to, produces, affects, influences, results_in.
        Returns {path: [nodes], relations: [rels], confidence_product: float} or None.
        """
        causal_rels = {"causes", "leads_to", "produces", "affects", "influences", "results_in"}

        if self.graph._use_neo4j:
            q = """
            MATCH path = (a:Concept {name: $src})-[:RELATES*1..5]->(b:Concept {name: $dst})
            WHERE ALL(r IN relationships(path) WHERE r.type IN $rels)
            RETURN [n IN nodes(path) | n.name] as nodes,
                   [r IN relationships(path) | r.type] as rels,
                   [r IN relationships(path) | coalesce(r.confidence, 0.6)] as confs
            ORDER BY length(path) ASC LIMIT 1
            """
            recs = self.graph.cypher(q, {"src": source, "dst": target, "rels": list(causal_rels)})
            if recs:
                r = recs[0]
                conf_product = 1.0
                for c in r["confs"]:
                    conf_product *= float(c)
                return {"path": r["nodes"], "relations": r["rels"], "confidence_product": round(conf_product, 4)}
            return None

        # NetworkX BFS
        import networkx as nx
        G = self.graph._nx_graph

        def causal_neighbors(node):
            for succ in G.successors(node):
                d = G[node][succ]
                if d.get("type", "relates") in causal_rels:
                    yield succ, d.get("type", "causes"), d.get("confidence", 0.6)

        # BFS with path tracking
        queue = deque([(source, [source], [], 1.0)])
        visited = {source}
        while queue:
            node, path, rels, conf_prod = queue.popleft()
            if len(path) > max_depth + 1:
                continue
            for neighbor, rel, conf in causal_neighbors(node):
                new_conf = conf_prod * conf
                new_path = path + [neighbor]
                new_rels = rels + [rel]
                if neighbor == target:
                    return {
                        "path": new_path,
                        "relations": new_rels,
                        "confidence_product": round(new_conf, 4),
                    }
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, new_path, new_rels, new_conf))
        return None

    def find_bridge_nodes(self, top_n: int = 10) -> List[Dict]:
        """
        Find nodes that bridge otherwise disconnected semantic clusters.
        A bridge node has high betweenness centrality AND neighbors spanning
        multiple semantic clusters (inter-cluster embedding distance > 0.5).
        Returns [{node, betweenness, cluster_span, description}] sorted by bridge score.
        """
        try:
            import networkx as nx
            import numpy as np

            if self.graph._use_neo4j:
                recs = self.graph.cypher(
                    "MATCH (a:Concept)-[:RELATES]->(b:Concept) RETURN a.name as a, b.name as b LIMIT 5000"
                )
                G = nx.DiGraph()
                for r in recs:
                    G.add_edge(r["a"], r["b"])
            else:
                G = self.graph._nx_graph

            if G.number_of_nodes() < 5:
                return []

            undirected = G.to_undirected()
            betweenness = nx.betweenness_centrality(undirected, k=min(100, G.number_of_nodes()))

            # For each high-betweenness node, measure neighbor embedding spread
            top_candidates = sorted(betweenness.items(), key=lambda x: -x[1])[:top_n * 3]
            bridges = []

            for node, bw in top_candidates:
                neighbors = list(undirected.neighbors(node))
                if len(neighbors) < 2:
                    continue
                # Get embeddings of neighbors
                embs = []
                for nb in neighbors[:20]:
                    if self.graph._use_neo4j:
                        recs = self.graph.cypher(
                            "MATCH (n:Concept {name: $n}) RETURN n.embedding as emb", {"n": nb}
                        )
                        if recs and recs[0].get("emb"):
                            embs.append(np.array(recs[0]["emb"], dtype=np.float32))
                    else:
                        emb = G.nodes[nb].get("embedding")
                        if emb:
                            embs.append(np.array(emb, dtype=np.float32))

                if len(embs) < 2:
                    bridges.append({"node": node, "betweenness": round(bw, 4), "cluster_span": 0.0, "bridge_score": round(bw, 4)})
                    continue

                # Compute pairwise distances to measure cluster spread
                sims = []
                for i in range(len(embs)):
                    for j in range(i + 1, len(embs)):
                        na, nb_ = np.linalg.norm(embs[i]), np.linalg.norm(embs[j])
                        if na > 0 and nb_ > 0:
                            sim = float(np.dot(embs[i], embs[j]) / (na * nb_))
                            sims.append(sim)

                # High spread (low avg similarity) = nodes span diverse clusters
                avg_sim = sum(sims) / len(sims) if sims else 1.0
                cluster_span = round(1.0 - avg_sim, 3)  # 0 = homogeneous, 1 = maximally diverse
                bridge_score = round(bw * (1 + cluster_span), 4)

                bridges.append({
                    "node": node,
                    "betweenness": round(bw, 4),
                    "cluster_span": cluster_span,
                    "bridge_score": bridge_score,
                })

            return sorted(bridges, key=lambda x: -x["bridge_score"])[:top_n]

        except Exception as e:
            logger.warning("Bridge node detection failed: %s", e)
            return []


def _cosine_sim(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two vectors."""
    try:
        import numpy as np
        x, y = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
        return float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-9))
    except Exception:
        return 0.0
