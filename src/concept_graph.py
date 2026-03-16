"""
Subsystem 2: Concept Graph
Stores concepts as nodes and relations as edges.
Uses Memgraph/Neo4j (Cypher) when available; NetworkX fallback for <10k nodes.
"""
import json
import logging
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
)

logger = logging.getLogger(__name__)

# Embedding model cache
_embedding_model = None


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
        """Connect to graph DB or init NetworkX fallback."""
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

    def _load_nx_from_disk(self):
        """Load NetworkX graph from disk if exists."""
        path = GRAPHS_DIR / "networkx_fallback.json"
        if path.exists():
            try:
                data = json.loads(path.read_text())
                import networkx as nx
                self._nx_graph = nx.node_link_graph(data)
                logger.info("Loaded NetworkX graph with %d nodes", self._nx_graph.number_of_nodes())
            except Exception as e:
                logger.warning("Could not load NetworkX backup: %s", e)

    def _save_nx_to_disk(self):
        """Persist NetworkX graph to disk."""
        if self._nx_graph is None:
            return
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

    def ingest_triples(self, triples: List[Tuple[str, str, str]], source: str = "ingest") -> int:
        """
        Create or merge nodes and edges from triples.
        Returns count of edges created/updated.
        """
        if not triples:
            return 0
        count = 0
        for s, r, o in triples:
            try:
                if self._use_neo4j:
                    count += self._ingest_neo4j(s, r, o, source)
                else:
                    count += self._ingest_nx(s, r, o, source)
            except Exception as e:
                logger.warning("Ingest failed for (%s, %s, %s): %s", s, r, o, e)
        if not self._use_neo4j:
            self._save_nx_to_disk()
        return count

    def _ingest_neo4j(self, s: str, r: str, o: str, source: str) -> int:
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
                ON CREATE SET e.weight = 1, e.source = $source
                ON MATCH SET e.weight = e.weight + 1
                """,
                s=s, o=o, rel=rel, emb_s=emb_s, emb_o=emb_o, now=now, source=source,
            )
        return 1

    def _ingest_nx(self, s: str, r: str, o: str, source: str) -> int:
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
        else:
            self._nx_graph.add_edge(s, o, type=r, weight=1, source=source)
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

    def get_neighbors(self, name: str, limit: int = 50) -> List[Tuple[str, str, float]]:
        """Get (neighbor, relation, weight) for a node."""
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

    def cypher(self, query: str, params: Optional[Dict] = None) -> List[Dict]:
        """Run Cypher query (Neo4j only). Returns list of records as dicts."""
        if not self._use_neo4j:
            logger.warning("Cypher only supported with Neo4j/Memgraph")
            return []
        with self._driver.session() as session:
            r = session.run(query, params or {})
            return [dict(rec) for rec in r]

    def semantic_search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Find concepts similar to query via embeddings."""
        q_emb = _embed(query)
        if self._use_neo4j:
            # Neo4j doesn't have built-in vector search; use FAISS or simple cosine in Python
            r = self.cypher("MATCH (n:Concept) RETURN n.name as name, n.embedding as emb")
            candidates = [(rec["name"], rec["emb"]) for rec in r if rec.get("emb")]
        else:
            candidates = [
                (n, data.get("embedding", [0.0] * 384))
                for n, data in self._nx_graph.nodes(data=True)
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
            logger.warning("Semantic search failed: %s", e)
            return []

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
