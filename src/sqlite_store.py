"""
SQLite Knowledge Store — persistent backing for the concept graph.

Replaces networkx_fallback.json with a proper relational store that preserves:
  - Full edge metadata (confidence, source_type, provenance, created_at, last_reinforced)
  - Node embeddings as binary blobs
  - Contradiction index
  - Hypothesis log

ConceptGraph uses this transparently: on startup, loads from SQLite into NetworkX;
on ingest, writes to SQLite. This survives restarts and accumulates history.
"""
import json
import logging
import sqlite3
import struct
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_DB_PATH = Path(__file__).resolve().parent.parent / "memory" / "knowledge.db"


def _pack_embedding(emb: List[float]) -> bytes:
    return struct.pack(f"{len(emb)}f", *emb)


def _unpack_embedding(blob: bytes) -> List[float]:
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


class SQLiteStore:
    """
    SQLite-backed persistent store for the concept graph.
    Thread-safe for single-process use (same-thread connections).
    """

    def __init__(self, db_path: Path = None):
        self._path = db_path or _DB_PATH
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        c = self._conn
        c.executescript("""
            CREATE TABLE IF NOT EXISTS nodes (
                name        TEXT PRIMARY KEY,
                embedding   BLOB,
                created_at  INTEGER DEFAULT 0,
                last_seen   INTEGER DEFAULT 0,
                access_count INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS edges (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                src             TEXT NOT NULL,
                dst             TEXT NOT NULL,
                relation        TEXT NOT NULL DEFAULT 'relates',
                weight          INTEGER DEFAULT 1,
                confidence      REAL DEFAULT 0.6,
                source_type     TEXT DEFAULT 'unknown',
                provenance      TEXT DEFAULT '',
                created_at      INTEGER DEFAULT 0,
                last_reinforced INTEGER DEFAULT 0,
                UNIQUE(src, dst, relation)
            );

            CREATE TABLE IF NOT EXISTS contradictions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                edge_a_src  TEXT, edge_a_rel TEXT, edge_a_dst TEXT,
                edge_b_src  TEXT, edge_b_rel TEXT, edge_b_dst TEXT,
                type        TEXT,
                severity    REAL DEFAULT 0.0,
                resolved    INTEGER DEFAULT 0,
                resolution  TEXT DEFAULT '',
                detected_at INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS hypothesis_log (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                src TEXT, rel TEXT, dst TEXT,
                strategy        TEXT,
                corroborated    INTEGER DEFAULT 0,
                confidence      REAL DEFAULT 0.25,
                evidence_source TEXT DEFAULT '',
                created_at      INTEGER DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src);
            CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst);
            CREATE INDEX IF NOT EXISTS idx_edges_confidence ON edges(confidence);
        """)
        c.commit()

    # ── Node operations ──────────────────────────────────────────────────────

    def upsert_node(self, name: str, embedding: List[float] = None):
        now = int(time.time())
        emb_blob = _pack_embedding(embedding) if embedding else None
        self._conn.execute(
            """
            INSERT INTO nodes (name, embedding, created_at, last_seen, access_count)
            VALUES (?, ?, ?, ?, 1)
            ON CONFLICT(name) DO UPDATE SET
                last_seen = excluded.last_seen,
                access_count = access_count + 1,
                embedding = COALESCE(excluded.embedding, embedding)
            """,
            (name, emb_blob, now, now),
        )
        self._conn.commit()

    def get_node(self, name: str) -> Optional[Dict]:
        row = self._conn.execute(
            "SELECT name, created_at, last_seen, access_count FROM nodes WHERE name=?", (name,)
        ).fetchone()
        return dict(row) if row else None

    def get_node_embedding(self, name: str) -> Optional[List[float]]:
        row = self._conn.execute(
            "SELECT embedding FROM nodes WHERE name=?", (name,)
        ).fetchone()
        if row and row["embedding"]:
            return _unpack_embedding(row["embedding"])
        return None

    def node_count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]

    def all_nodes(self) -> List[str]:
        rows = self._conn.execute("SELECT name FROM nodes").fetchall()
        return [r["name"] for r in rows]

    # ── Edge operations ───────────────────────────────────────────────────────

    def upsert_edge(
        self,
        src: str, dst: str, relation: str = "relates",
        confidence: float = 0.6, source_type: str = "unknown",
        provenance: str = "", weight: int = 1,
    ):
        now = int(time.time())
        self._conn.execute(
            """
            INSERT INTO edges (src, dst, relation, weight, confidence, source_type,
                               provenance, created_at, last_reinforced)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(src, dst, relation) DO UPDATE SET
                weight = weight + 1,
                confidence = MIN(1.0, CASE
                    WHEN excluded.confidence > confidence THEN confidence + 0.1
                    ELSE confidence END),
                last_reinforced = excluded.last_reinforced
            """,
            (src, dst, relation, weight, confidence, source_type, provenance, now, now),
        )
        self._conn.commit()

    def get_edges(self, src: str, limit: int = 50) -> List[Dict]:
        rows = self._conn.execute(
            """
            SELECT dst, relation, weight, confidence, source_type, provenance,
                   created_at, last_reinforced
            FROM edges WHERE src=?
            ORDER BY weight DESC LIMIT ?
            """,
            (src, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def edge_count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]

    def avg_confidence(self, min_confidence: float = 0.0) -> float:
        row = self._conn.execute(
            "SELECT AVG(confidence) FROM edges WHERE confidence >= ?", (min_confidence,)
        ).fetchone()
        return float(row[0] or 0.0)

    def get_low_confidence_edges(self, threshold: float = 0.15) -> List[Dict]:
        rows = self._conn.execute(
            "SELECT src, dst, relation, confidence, last_reinforced FROM edges WHERE confidence < ?",
            (threshold,),
        ).fetchall()
        return [dict(r) for r in rows]

    def delete_edge(self, src: str, dst: str, relation: str):
        self._conn.execute(
            "DELETE FROM edges WHERE src=? AND dst=? AND relation=?", (src, dst, relation)
        )
        self._conn.commit()

    def apply_decay(self, decay_lambda: float = 1 / (30 * 86400), archive_threshold: float = 0.1) -> int:
        """
        Apply Ebbinghaus-style exponential decay to all edge confidences.
        c(t) = c0 * e^(-Δt * decay_lambda)
        Edges below archive_threshold are deleted (archived).
        Returns count of archived edges.
        """
        now = int(time.time())
        import math
        rows = self._conn.execute(
            "SELECT rowid, confidence, last_reinforced FROM edges"
        ).fetchall()
        archived = 0
        for row in rows:
            rowid, conf, last_reinforced = row[0], row[1], row[2]
            if not last_reinforced:
                continue
            delta_t = now - last_reinforced
            new_conf = conf * math.exp(-decay_lambda * delta_t)
            if new_conf < archive_threshold:
                self._conn.execute("DELETE FROM edges WHERE rowid=?", (rowid,))
                archived += 1
            else:
                self._conn.execute(
                    "UPDATE edges SET confidence=? WHERE rowid=?", (round(new_conf, 4), rowid)
                )
        self._conn.commit()
        return archived

    # ── Contradiction log ─────────────────────────────────────────────────────

    def log_contradiction(
        self, edge_a: Tuple, edge_b: Tuple, ctype: str, severity: float
    ):
        now = int(time.time())
        self._conn.execute(
            """
            INSERT OR IGNORE INTO contradictions
            (edge_a_src, edge_a_rel, edge_a_dst, edge_b_src, edge_b_rel, edge_b_dst,
             type, severity, detected_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (*edge_a, *edge_b, ctype, severity, now),
        )
        self._conn.commit()

    def get_contradictions(self, resolved: bool = False, limit: int = 50) -> List[Dict]:
        rows = self._conn.execute(
            "SELECT * FROM contradictions WHERE resolved=? ORDER BY severity DESC LIMIT ?",
            (1 if resolved else 0, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Hypothesis log ────────────────────────────────────────────────────────

    def log_hypothesis(
        self, triple: Tuple[str, str, str], strategy: str,
        corroborated: bool, confidence: float, evidence_source: str = "",
    ):
        now = int(time.time())
        self._conn.execute(
            """
            INSERT INTO hypothesis_log (src, rel, dst, strategy, corroborated,
                                        confidence, evidence_source, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (*triple, strategy, 1 if corroborated else 0, confidence, evidence_source, now),
        )
        self._conn.commit()

    # ── Export / import ───────────────────────────────────────────────────────

    def export_to_networkx(self):
        """Build a NetworkX DiGraph from SQLite data (used on startup)."""
        import networkx as nx
        G = nx.DiGraph()
        for name in self.all_nodes():
            emb = self.get_node_embedding(name)
            node_data = self.get_node(name) or {}
            G.add_node(
                name,
                embedding=emb or [0.0] * 384,
                last_access=node_data.get("last_seen", 0),
                degree=0,
            )
        rows = self._conn.execute(
            "SELECT src, dst, relation, weight, confidence, source_type, provenance, created_at, last_reinforced FROM edges"
        ).fetchall()
        for row in rows:
            G.add_edge(
                row["src"], row["dst"],
                type=row["relation"],
                weight=row["weight"],
                confidence=row["confidence"],
                source_type=row["source_type"],
                provenance=row["provenance"],
                created_at=row["created_at"],
                last_reinforced=row["last_reinforced"],
                source=row["source_type"],
            )
        return G

    def import_from_networkx(self, G):
        """Bulk-import a NetworkX graph into SQLite (migration from JSON fallback)."""
        for name, data in G.nodes(data=True):
            self.upsert_node(name, data.get("embedding"))
        for u, v, data in G.edges(data=True):
            self.upsert_edge(
                src=u, dst=v,
                relation=data.get("type", "relates"),
                confidence=data.get("confidence", 0.6),
                source_type=data.get("source_type", data.get("source", "legacy")),
                provenance=data.get("provenance", ""),
                weight=data.get("weight", 1),
            )

    def close(self):
        self._conn.close()


# Module-level singleton
_default_store: Optional[SQLiteStore] = None


def get_store(db_path: Path = None) -> SQLiteStore:
    global _default_store
    if _default_store is None:
        _default_store = SQLiteStore(db_path)
    return _default_store
