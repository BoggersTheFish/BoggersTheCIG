"""
Subsystem 2: Concept Graph
Stores concepts as nodes and relations as edges.
Auto-switch: NetworkX for <50k nodes, Memgraph above. Sharding via community detection.
Simulated: 8GB laptop → NetworkX; RTX 4090 + 60k nodes → Memgraph.
"""
import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.config import (
    GRAPH_URI,
    GRAPH_PASSWORD,
    GRAPH_USER,
    USE_NETWORKX_FALLBACK,
    NETWORKX_MAX_NODES,
    GRAPHS_DIR,
    ENABLE_LARGE_GRAPH,
    MIN_NODES_FOR_MEMGRAPH,
    PROJECT_ROOT,
)

logger = logging.getLogger(__name__)

# Embedding model cache
_embedding_model = None


def check_graph_size(graph: "ConceptGraph") -> bool:
    """
    If nodes > MIN_NODES_FOR_MEMGRAPH and Memgraph not running, attempt docker-compose up.
    Returns True if switched to Memgraph.
    """
    if not ENABLE_LARGE_GRAPH:
        return False
    n = graph.node_count()
    if n < MIN_NODES_FOR_MEMGRAPH:
        return False
    if graph._use_neo4j:
        return True
    logger.warning("Graph has %d nodes (>= %d); Memgraph not running. Attempting docker-compose up.", n, MIN_NODES_FOR_MEMGRAPH)
    try:
        dc = PROJECT_ROOT / "docker-compose.yaml"
        if dc.exists():
            subprocess.run(
                ["docker-compose", "up", "-d", "memgraph"],
                cwd=PROJECT_ROOT,
                capture_output=True,
                timeout=60,
            )
            time.sleep(5)
            from neo4j import GraphDatabase
            auth = (GRAPH_USER, GRAPH_PASSWORD) if (GRAPH_USER and GRAPH_PASSWORD) else ("neo4j", "password")
            driver = GraphDatabase.driver(GRAPH_URI, auth=auth)
            driver.verify_connectivity()
            driver.close()
            logger.info("Memgraph started. Reconnect by creating new ConceptGraph after migration.")
        else:
            logger.warning("docker-compose.yaml not found; cannot auto-start Memgraph")
    except Exception as e:
        logger.warning("Could not start Memgraph: %s", e)
    return False


def get_subgraphs_by_community(graph: "ConceptGraph", max_communities: int = 20) -> List[List[str]]:
    """Split graph into subgraphs by community detection. Returns list of node lists per community."""
    import networkx as nx
    if graph._use_neo4j:
        recs = graph.cypher("MATCH (a:Concept)-[r:RELATES]->(b:Concept) RETURN a.name as a, b.name as b LIMIT 100000")
        G = nx.Graph()
        for r in recs:
            G.add_edge(r["a"], r["b"])
    else:
        G = graph._nx_graph.to_undirected()
    if G.number_of_nodes() == 0:
        return []
    try:
        from networkx.algorithms import community
        if hasattr(community, "louvain_communities"):
            comms = list(community.louvain_communities(G))[:max_communities]
        elif hasattr(community, "greedy_modularity_communities"):
            comms = list(community.greedy_modularity_communities(G))[:max_communities]
        else:
            comms = [set(G.nodes())]
        return [list(c) for c in comms if c]
    except Exception as e:
        logger.debug("Community detection failed: %s", e)
        return [list(graph._nx_graph.nodes())] if not graph._use_neo4j else []


def _get_embedding_model():
    """Lazy-load sentence-transformers model."""
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model
    try:
        from sentence_transformers import SentenceTransformer
        from src.config import EMBEDDING_MODEL
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        return _embedding_model
    except Exception as e:
        logger.warning("Embedding model load failed: %s", e)
        return None


def _embed(text: str) -> List[float]:
    """Get embedding vector for text. Returns zeros if model unavailable."""
    model = _get_embedding_model()
    if model is None:
        return [0.0] * 384  # all-MiniLM-L6-v2 dimension
    try:
        return model.encode(text, convert_to_numpy=True).tolist()
    except Exception:
        return [0.0] * 384


class ConceptGraph:
    """
    Concept graph backend: Memgraph/Neo4j or NetworkX.
    Nodes: concepts with embedding, last_access, degree.
    Edges: relations with weights.
    """

    def __init__(self):
        self._driver = None
        self._nx_graph = None
        self._use_neo4j = False
        self._init_backend()

    def _init_backend(self):
        """Connect to graph DB or init NetworkX fallback. Prefer Memgraph when large graph enabled."""
        try:
            from neo4j import GraphDatabase
            auth = (GRAPH_USER, GRAPH_PASSWORD) if (GRAPH_USER and GRAPH_PASSWORD) else ("neo4j", "password")
            self._driver = GraphDatabase.driver(GRAPH_URI, auth=auth)
            self._driver.verify_connectivity()
            self._use_neo4j = True
            logger.info("Connected to graph DB at %s", GRAPH_URI)
        except Exception as e:
            logger.warning("Graph DB unavailable (%s), using NetworkX fallback", e)
            self._use_neo4j = False
            import networkx as nx
            self._nx_graph = nx.DiGraph()
            self._load_nx_from_disk()
            if ENABLE_LARGE_GRAPH and self._nx_graph.number_of_nodes() >= MIN_NODES_FOR_MEMGRAPH:
                check_graph_size(self)

    def _load_nx_from_disk(self):
        """Load NetworkX graph from SQLite (preferred) or JSON fallback."""
        # Try SQLite first (richer metadata)
        try:
            from src.sqlite_store import get_store
            store = get_store()
            if store.node_count() > 0:
                self._nx_graph = store.export_to_networkx()
                logger.info(
                    "Loaded graph from SQLite: %d nodes, %d edges",
                    self._nx_graph.number_of_nodes(),
                    self._nx_graph.number_of_edges(),
                )
                return
        except Exception as e:
            logger.debug("SQLite load failed, trying JSON fallback: %s", e)

        # JSON fallback
        path = GRAPHS_DIR / "networkx_fallback.json"
        if path.exists():
            try:
                data = json.loads(path.read_text())
                import networkx as nx
                self._nx_graph = nx.node_link_graph(data)
                logger.info("Loaded NetworkX graph from JSON: %d nodes", self._nx_graph.number_of_nodes())
                # Migrate to SQLite
                self._save_nx_to_disk()
            except Exception as e:
                logger.warning("Could not load NetworkX backup: %s", e)

    def _save_nx_to_disk(self):
        """Persist NetworkX graph to SQLite (primary) and JSON (backup)."""
        if self._nx_graph is None:
            return
        # SQLite primary store
        try:
            from src.sqlite_store import get_store
            store = get_store()
            store.import_from_networkx(self._nx_graph)
        except Exception as e:
            logger.debug("SQLite save failed: %s", e)
        # JSON backup (kept for compatibility)
        path = GRAPHS_DIR / "networkx_fallback.json"
        try:
            import networkx as nx
            data = nx.node_link_data(self._nx_graph)
            path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.warning("Could not save NetworkX backup: %s", e)

    def init_schema(self):
        """Create indexes/constraints in Neo4j/Memgraph."""
        if not self._use_neo4j:
            return
        with self._driver.session() as session:
            try:
                session.run("CREATE INDEX concept_name IF NOT EXISTS FOR (n:Concept) ON (n.name)")
                session.run("CREATE CONSTRAINT concept_unique IF NOT EXISTS FOR (n:Concept) REQUIRE n.name IS UNIQUE")
            except Exception as e:
                logger.debug("Schema init (may already exist): %s", e)

    def ingest_triples(
        self,
        triples: List[Tuple],
        source: str = "ingest",
        default_confidence: float = 0.6,
        source_type: str = None,
        provenance: str = "",
        use_provenance_store: bool = True,
    ) -> int:
        """
        Create or merge nodes and edges from triples.
        Each triple may be (s, r, o), (s, r, o, confidence), or (s, r, o, confidence, source_type).
        Returns count of edges created/updated.
        """
        if not triples:
            return 0
        _source_type = source_type or source
        # Lazy-load provenance store
        pstore = None
        if use_provenance_store:
            try:
                from src.provenance_store import get_store
                pstore = get_store()
            except Exception:
                pass
        count = 0
        for triple in triples:
            s, r, o = triple[0], triple[1], triple[2]
            confidence = float(triple[3]) if len(triple) > 3 else default_confidence
            st = triple[4] if len(triple) > 4 else _source_type
            # Update provenance store and get (possibly boosted) confidence
            if pstore is not None:
                try:
                    confidence = pstore.add(
                        (s, r, o), source=source, confidence=confidence,
                        source_type=st, provenance=provenance,
                    )
                except Exception as e:
                    logger.debug("Provenance store add failed: %s", e)
            try:
                if self._use_neo4j:
                    count += self._ingest_neo4j(s, r, o, source, confidence, st, provenance)
                else:
                    count += self._ingest_nx(s, r, o, source, confidence, st, provenance)
            except Exception as e:
                logger.warning("Ingest failed for (%s, %s, %s): %s", s, r, o, e)
        if not self._use_neo4j:
            self._save_nx_to_disk()
        return count

    def _ingest_neo4j(
        self, s: str, r: str, o: str, source: str,
        confidence: float = 0.6, source_type: str = "ingest", provenance: str = "",
    ) -> int:
        """Ingest one triple into Neo4j/Memgraph."""
        now = int(time.time())
        emb_s = _embed(s)
        emb_o = _embed(o)
        rel = r.replace(" ", "_").replace("-", "_")[:64]
        with self._driver.session() as session:
            session.run(
                """
                MERGE (a:Concept {name: $s})
                ON CREATE SET a.embedding = $emb_s, a.last_access = $now, a.degree = 1
                ON MATCH SET a.last_access = $now, a.degree = a.degree + 1
                MERGE (b:Concept {name: $o})
                ON CREATE SET b.embedding = $emb_o, b.last_access = $now, b.degree = 1
                ON MATCH SET b.last_access = $now, b.degree = b.degree + 1
                MERGE (a)-[e:RELATES {type: $rel}]->(b)
                ON CREATE SET e.weight = 1, e.source = $source,
                              e.confidence = $confidence, e.source_type = $source_type,
                              e.provenance = $provenance, e.created_at = $now,
                              e.last_reinforced = $now
                ON MATCH SET e.weight = e.weight + 1,
                             e.confidence = CASE WHEN $confidence > e.confidence
                                            THEN CASE WHEN e.confidence + 0.1 > 1.0 THEN 1.0
                                                 ELSE e.confidence + 0.1 END
                                            ELSE e.confidence END,
                             e.last_reinforced = $now
                """,
                s=s, o=o, rel=rel, emb_s=emb_s, emb_o=emb_o, now=now, source=source,
                confidence=confidence, source_type=source_type, provenance=provenance,
            )
        return 1

    def _ingest_nx(
        self, s: str, r: str, o: str, source: str,
        confidence: float = 0.6, source_type: str = "ingest", provenance: str = "",
    ) -> int:
        """Ingest one triple into NetworkX."""
        now = int(time.time())
        if self._nx_graph.number_of_nodes() >= NETWORKX_MAX_NODES:
            logger.warning("NetworkX at max nodes (%d), skipping ingest", NETWORKX_MAX_NODES)
            return 0
        if not self._nx_graph.has_node(s):
            self._nx_graph.add_node(s, embedding=_embed(s), last_access=now, degree=0)
        if not self._nx_graph.has_node(o):
            self._nx_graph.add_node(o, embedding=_embed(o), last_access=now, degree=0)
        if self._nx_graph.has_edge(s, o):
            data = self._nx_graph[s][o]
            data["weight"] = data.get("weight", 1) + 1
            data["relations"] = data.get("relations", [r])
            if r not in data["relations"]:
                data["relations"].append(r)
            # Boost confidence on corroboration (multi-source agreement)
            old_conf = data.get("confidence", 0.6)
            data["confidence"] = min(1.0, old_conf + 0.1)
            data["last_reinforced"] = now
        else:
            self._nx_graph.add_edge(
                s, o,
                type=r, weight=1, source=source,
                confidence=confidence, source_type=source_type,
                provenance=provenance, created_at=now, last_reinforced=now,
            )
        self._nx_graph.nodes[s]["last_access"] = now
        self._nx_graph.nodes[o]["last_access"] = now
        return 1

    def get_node(self, name: str) -> Optional[Dict[str, Any]]:
        """Get node by name."""
        if self._use_neo4j:
            with self._driver.session() as session:
                r = session.run(
                    "MATCH (n:Concept {name: $name}) RETURN n.name as name, n.last_access as last_access, n.degree as degree",
                    name=name,
                )
                rec = r.single()
                return dict(rec) if rec else None
        if self._nx_graph.has_node(name):
            data = self._nx_graph.nodes[name]
            return {"name": name, "last_access": data.get("last_access"), "degree": self._nx_graph.degree(name)}
        return None

    def get_neighbors(self, name: str, limit: int = 50, max_depth: int = 1) -> List[Tuple[str, str, float]]:
        """Get (neighbor, relation, weight) for a node. max_depth=1 for memory efficiency at scale."""
        if self._use_neo4j:
            with self._driver.session() as session:
                r = session.run(
                    """
                    MATCH (a:Concept {name: $name})-[e:RELATES]->(b:Concept)
                    RETURN b.name as neighbor, e.type as rel, e.weight as weight
                    ORDER BY e.weight DESC LIMIT $limit
                    """,
                    name=name, limit=limit,
                )
                return [(rec["neighbor"], rec["rel"], rec["weight"]) for rec in r]
        if not self._nx_graph.has_node(name):
            return []
        out = []
        for t in self._nx_graph.successors(name):
            data = self._nx_graph[name][t]
            out.append((t, data.get("type", "relates"), data.get("weight", 1)))
        return sorted(out, key=lambda x: -x[2])[:limit]

    def get_edges_with_meta(self, name: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Return full edge metadata for a node's outgoing edges.
        Each dict: {neighbor, relation, weight, confidence, source_type, provenance, created_at, last_reinforced}
        """
        if self._use_neo4j:
            with self._driver.session() as session:
                r = session.run(
                    """
                    MATCH (a:Concept {name: $name})-[e:RELATES]->(b:Concept)
                    RETURN b.name as neighbor, e.type as relation, e.weight as weight,
                           e.confidence as confidence, e.source_type as source_type,
                           e.provenance as provenance, e.created_at as created_at,
                           e.last_reinforced as last_reinforced
                    ORDER BY e.weight DESC LIMIT $limit
                    """,
                    name=name, limit=limit,
                )
                return [dict(rec) for rec in r]
        if not self._nx_graph.has_node(name):
            return []
        result = []
        for t in self._nx_graph.successors(name):
            d = self._nx_graph[name][t]
            result.append({
                "neighbor": t,
                "relation": d.get("type", "relates"),
                "weight": d.get("weight", 1),
                "confidence": d.get("confidence", 0.6),
                "source_type": d.get("source_type", "unknown"),
                "provenance": d.get("provenance", ""),
                "created_at": d.get("created_at", 0),
                "last_reinforced": d.get("last_reinforced", 0),
            })
        return sorted(result, key=lambda x: -x["weight"])[:limit]

    def avg_confidence(self, min_confidence: float = 0.0) -> float:
        """Average edge confidence across the graph, optionally filtered."""
        if self._use_neo4j:
            r = self.cypher(
                "MATCH ()-[e:RELATES]->() WHERE e.confidence >= $c RETURN avg(e.confidence) as avg",
                {"c": min_confidence},
            )
            return float(r[0]["avg"] or 0.0) if r else 0.0
        if self._nx_graph is None or self._nx_graph.number_of_edges() == 0:
            return 0.0
        confs = [
            d.get("confidence", 0.6)
            for _, _, d in self._nx_graph.edges(data=True)
            if d.get("confidence", 0.6) >= min_confidence
        ]
        return sum(confs) / len(confs) if confs else 0.0

    def get_neighbors_batch(self, names: List[str], limit: int = 20) -> Dict[str, List[Tuple[str, str, float]]]:
        """Batch retrieval for memory efficiency. Returns {node: [(neighbor, rel, weight), ...]}."""
        return {n: self.get_neighbors(n, limit=limit) for n in names}

    def apply_decay(
        self,
        decay_days: float = 30.0,
        archive_threshold: float = 0.1,
        protect_source_types: Tuple = ("human",),
    ) -> dict:
        """
        Apply Ebbinghaus-style exponential confidence decay to all edges.
        c(t) = c0 * exp(-Δt / stability)
        where stability = decay_days * 86400 seconds.
        Edges below archive_threshold and not in protect_source_types are removed.
        Returns {decayed: N, archived: N}.
        """
        import math
        now = int(time.time())
        stability = decay_days * 86400  # convert days to seconds
        decay_lambda = 1.0 / stability
        decayed = 0
        archived = 0

        # SQLite path: delegate entirely
        try:
            from src.sqlite_store import get_store
            store = get_store()
            archived = store.apply_decay(decay_lambda=decay_lambda, archive_threshold=archive_threshold)
            # Reload NetworkX from SQLite after decay
            if not self._use_neo4j:
                self._nx_graph = store.export_to_networkx()
            decayed = self._nx_graph.number_of_edges() if not self._use_neo4j else 0
            logger.info("Decay applied via SQLite: %d archived", archived)
            return {"decayed": decayed, "archived": archived}
        except Exception as e:
            logger.debug("SQLite decay failed, falling back to in-memory: %s", e)

        # In-memory NetworkX fallback
        if self._use_neo4j:
            recs = self.cypher(
                "MATCH ()-[e:RELATES]->() RETURN id(e) as eid, e.confidence as conf, "
                "e.last_reinforced as lr, e.source_type as st"
            )
            for r in recs:
                lr = r.get("lr") or now
                st = r.get("st") or ""
                if st in protect_source_types:
                    continue
                delta_t = now - lr
                new_conf = (r.get("conf") or 0.6) * math.exp(-decay_lambda * delta_t)
                if new_conf < archive_threshold:
                    self.cypher("MATCH ()-[e:RELATES]->() WHERE id(e)=$eid DELETE e", {"eid": r["eid"]})
                    archived += 1
                else:
                    self.cypher(
                        "MATCH ()-[e:RELATES]->() WHERE id(e)=$eid SET e.confidence=$c",
                        {"eid": r["eid"], "c": round(new_conf, 4)},
                    )
                    decayed += 1
        else:
            G = self._nx_graph
            edges_to_remove = []
            for u, v, d in G.edges(data=True):
                if d.get("source_type") in protect_source_types:
                    continue
                lr = d.get("last_reinforced") or now
                delta_t = now - lr
                new_conf = d.get("confidence", 0.6) * math.exp(-decay_lambda * delta_t)
                if new_conf < archive_threshold:
                    edges_to_remove.append((u, v))
                    archived += 1
                else:
                    d["confidence"] = round(new_conf, 4)
                    decayed += 1
            for u, v in edges_to_remove:
                G.remove_edge(u, v)
            self._save_nx_to_disk()

        logger.info("Decay applied: %d decayed, %d archived", decayed, archived)
        return {"decayed": decayed, "archived": archived}

    def prune_low_degree_nodes(self, min_degree: int = None) -> int:
        """Remove nodes with degree < min_degree. Returns count pruned. For coherence at scale."""
        from src.config import PRUNE_DEGREE_THRESHOLD
        thresh = max(min_degree or 0, PRUNE_DEGREE_THRESHOLD)
        if self._use_neo4j:
            r = self.cypher(
                "MATCH (n:Concept) WHERE size((n)--()) < $thresh RETURN n.name as name",
                {"thresh": thresh},
            )
            with self._driver.session() as session:
                for rec in r:
                    session.run("MATCH (n:Concept {name: $name}) DETACH DELETE n", name=rec["name"])
            return len(r)
        to_remove = [n for n in self._nx_graph.nodes() if self._nx_graph.degree(n) < thresh]
        for n in to_remove:
            self._nx_graph.remove_node(n)
        self._save_nx_to_disk()
        return len(to_remove)

    def cypher(self, query: str, params: Optional[Dict] = None) -> List[Dict]:
        """Run Cypher query (Neo4j only). Returns list of records as dicts."""
        if not self._use_neo4j:
            logger.warning("Cypher only supported with Neo4j/Memgraph")
            return []
        with self._driver.session() as session:
            r = session.run(query, params or {})
            return [dict(rec) for rec in r]

    def semantic_search(self, query: str, top_k: int = 10, batch_limit: int = 5000) -> List[Tuple[str, float]]:
        """Find concepts similar to query via embeddings. batch_limit for memory efficiency at scale."""
        q_emb = _embed(query)
        if self._use_neo4j:
            r = self.cypher("MATCH (n:Concept) RETURN n.name as name, n.embedding as emb", {})
            r = r[:batch_limit]
            candidates = [(rec["name"], rec["emb"]) for rec in r if rec.get("emb")]
        else:
            nodes = list(self._nx_graph.nodes(data=True))[:batch_limit]
            candidates = [
                (n, data.get("embedding", [0.0] * 384))
                for n, data in nodes
                if data.get("embedding")
            ]
        if not candidates:
            return []
        try:
            import numpy as np
            q = np.array(q_emb, dtype=np.float32)
            scores = []
            for name, emb in candidates:
                if emb and len(emb) == len(q_emb):
                    e = np.array(emb, dtype=np.float32)
                    sim = np.dot(q, e) / (np.linalg.norm(q) * np.linalg.norm(e) + 1e-9)
                    scores.append((name, float(sim)))
            return sorted(scores, key=lambda x: -x[1])[:top_k]
        except Exception as e:
            logger.debug("Vector semantic search unavailable, using lexical fallback: %s", e)
            q_text = query.lower()
            scores = []
            for name, _ in candidates:
                name_text = str(name).lower()
                score = 1.0 if q_text in name_text else 0.0
                scores.append((name, score))
            return sorted(scores, key=lambda x: (-x[1], x[0]))[:top_k]

    def node_count(self) -> int:
        """Total node count."""
        if self._use_neo4j:
            r = self.cypher("MATCH (n:Concept) RETURN count(n) as c")
            return r[0]["c"] if r else 0
        return self._nx_graph.number_of_nodes()

    def edge_count(self) -> int:
        """Total edge count."""
        if self._use_neo4j:
            r = self.cypher("MATCH ()-[r:RELATES]->() RETURN count(r) as c")
            return r[0]["c"] if r else 0
        return self._nx_graph.number_of_edges()

    def close(self):
        """Close connections."""
        if self._driver:
            self._driver.close()
