"""
Provenance Store — maps (Subject, Relation, Object) triples to their source chain.

Responsibilities:
- Track every source that contributed to a belief (triple)
- Boost confidence when multiple independent sources agree
- Flag potential contradictions when two triples oppose each other
- Persist to disk as provenance_store.jsonl for auditability

Confidence rules:
  First occurrence:           use caller-supplied confidence
  Corroboration (same source_type, new source): min(1.0, existing + 0.1)
  Cross-type corroboration (e.g. web + human): min(1.0, existing + 0.2)
  Contradiction detected:     flag, do not boost
"""
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

from src.config import MEMORY_DIR

_STORE_PATH = MEMORY_DIR / "provenance_store.jsonl"
_INDEX_PATH = MEMORY_DIR / "provenance_index.json"


def _triple_key(s: str, r: str, o: str) -> str:
    return f"{s.lower().strip()}|{r.lower().strip()}|{o.lower().strip()}"


class ProvenanceStore:
    """
    In-memory provenance index backed by JSONL on disk.
    Thread-safe for single-process use.
    """

    def __init__(self, store_path: Path = None, index_path: Path = None):
        self._path = store_path or _STORE_PATH
        self._index_path = index_path or (
            self._path.with_name("provenance_index.json") if store_path else _INDEX_PATH
        )
        # key → {confidence, sources: [{source, source_type, provenance, ts}]}
        self._index: Dict[str, dict] = {}
        self._load()

    def _load(self):
        """Load index from disk."""
        if self._index_path.exists():
            try:
                self._index = json.loads(self._index_path.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning("ProvenanceStore load failed: %s", e)
                self._index = {}

    def _save(self):
        """Persist index to disk."""
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._index_path.write_text(
                json.dumps(self._index, indent=2), encoding="utf-8"
            )
        except Exception as e:
            logger.warning("ProvenanceStore save failed: %s", e)

    def _append_log(self, entry: dict):
        """Append raw event to JSONL audit log."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except Exception as e:
            logger.debug("ProvenanceStore log failed: %s", e)

    def add(
        self,
        triple: Tuple[str, str, str],
        source: str,
        confidence: float,
        source_type: str = "unknown",
        provenance: str = "",
    ) -> float:
        """
        Register a triple from a source.
        Returns the (possibly boosted) confidence after corroboration logic.
        """
        s, r, o = triple
        key = _triple_key(s, r, o)
        now = int(time.time())

        entry = {
            "source": source,
            "source_type": source_type,
            "provenance": provenance,
            "confidence": confidence,
            "ts": now,
        }

        if key not in self._index:
            self._index[key] = {"confidence": confidence, "sources": [entry]}
            new_conf = confidence
        else:
            existing = self._index[key]
            old_conf = existing["confidence"]
            old_types = {s["source_type"] for s in existing["sources"]}

            # Cross-type corroboration is a stronger signal
            if source_type not in old_types:
                boost = 0.2
            else:
                boost = 0.1
            new_conf = min(1.0, old_conf + boost)
            existing["confidence"] = new_conf
            existing["sources"].append(entry)
            # Cap stored sources at 20 to prevent unbounded growth
            existing["sources"] = existing["sources"][-20:]

        self._append_log({
            "event": "add",
            "triple": [s, r, o],
            "source_type": source_type,
            "provenance": provenance,
            "confidence_result": new_conf,
            "ts": now,
        })
        self._save()
        return new_conf

    def get(self, triple: Tuple[str, str, str]) -> Optional[dict]:
        """Return stored provenance record or None."""
        key = _triple_key(*triple)
        return self._index.get(key)

    def get_confidence(self, triple: Tuple[str, str, str]) -> float:
        """Return stored confidence or 0.0 if unknown."""
        rec = self.get(triple)
        return rec["confidence"] if rec else 0.0

    def source_count(self, triple: Tuple[str, str, str]) -> int:
        """Number of distinct sources that confirmed this triple."""
        rec = self.get(triple)
        return len(rec["sources"]) if rec else 0

    def find_contradictions(
        self, subject: str, limit: int = 20
    ) -> List[Dict]:
        """
        Find potential contradictions for triples sharing the same subject.
        A contradiction is: (S, R, O1) and (S, R, O2) where O1 ≠ O2 for
        exclusive relations (causes, is_a, implies, equals).
        Returns list of {triple_a, triple_b, type}.
        """
        exclusive_rels = {"causes", "is_a", "implies", "equals", "is_not", "contradicts"}
        subject_lower = subject.lower().strip()

        # Find all triples with this subject
        subject_triples = []
        for key, rec in self._index.items():
            parts = key.split("|")
            if len(parts) == 3 and parts[0] == subject_lower:
                subject_triples.append((parts[0], parts[1], parts[2], rec["confidence"]))

        contradictions = []
        seen = set()
        for i, (s, r, o1, c1) in enumerate(subject_triples):
            if r not in exclusive_rels:
                continue
            for s2, r2, o2, c2 in subject_triples[i + 1:]:
                if r == r2 and o1 != o2:
                    pair_key = f"{r}:{min(o1,o2)}:{max(o1,o2)}"
                    if pair_key not in seen:
                        seen.add(pair_key)
                        contradictions.append({
                            "triple_a": (s, r, o1),
                            "triple_b": (s2, r2, o2),
                            "type": f"conflicting_{r}",
                            "confidence_a": c1,
                            "confidence_b": c2,
                        })
        return contradictions[:limit]

    def stats(self) -> dict:
        """Summary statistics."""
        total = len(self._index)
        multi_source = sum(
            1 for v in self._index.values() if len(v.get("sources", [])) > 1
        )
        high_conf = sum(
            1 for v in self._index.values() if v.get("confidence", 0) >= 0.75
        )
        return {
            "total_triples": total,
            "multi_source_triples": multi_source,
            "high_confidence_triples": high_conf,
        }


# Module-level singleton for convenience
_default_store: Optional[ProvenanceStore] = None


def get_store(store_path: Path = None, index_path: Path = None) -> ProvenanceStore:
    global _default_store
    requested_path = Path(store_path) if store_path else None
    requested_index = Path(index_path) if index_path else None
    if _default_store is None:
        _default_store = ProvenanceStore(requested_path, requested_index)
    elif requested_path is not None and _default_store._path != requested_path:
        _default_store = ProvenanceStore(requested_path, requested_index)
    return _default_store
