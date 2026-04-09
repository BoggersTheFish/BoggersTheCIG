"""
Bidirectional Obsidian Sync — propagates human vault edits back into the concept graph.

Direction: Obsidian .md files → ConceptGraph edges

When a user edits a concept note (adds/removes wikilinks, changes relation labels),
this module detects the diff and applies it to the graph:
  - New wikilinks   → add edge with confidence=CONFIDENCE_HUMAN (0.9), source_type='human'
  - Removed links   → reduce edge confidence by 0.3 (or remove if below 0.15)
  - Human-added edges are protected from Ebbinghaus decay

Usage:
  sync = ObsidianSync(vault_path)
  result = sync.sync_vault_changes()  # one-shot scan of modified files

The sync uses a state file (obsidian_sync_state.json) to track the last-seen
wikilink structure of each note. On each call, it diffs current vs. stored state.
"""
import json
import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

_WIKILINK_RE = re.compile(r"\[\[([^\]|#]+?)(?:\|[^\]]+)?\]\]")
_RELATION_RE = re.compile(
    r"-\s+\[\[([^\]]+)\]\]\s+\(([^,)]+?)(?:,\s*weight=[\d.]+)?\)"
)


def _extract_wikilinks(content: str) -> Set[str]:
    """Extract all wikilink targets from Markdown content."""
    return {m.group(1).strip() for m in _WIKILINK_RE.finditer(content)}


def _extract_neighbor_relations(content: str) -> Dict[str, str]:
    """
    Extract neighbor relations from the standard concept note format:
      - [[NeighborName]] (relation_type, weight=N)
    Returns {neighbor_name: relation_type}.
    """
    relations = {}
    for m in _RELATION_RE.finditer(content):
        neighbor = m.group(1).strip()
        relation = m.group(2).strip().replace(" ", "_")
        relations[neighbor] = relation
    return relations


class ObsidianSync:
    """
    Detects user edits to Obsidian vault concept notes and syncs them to the graph.
    """

    def __init__(self, vault_path: Path = None):
        from src.config import OBSIDIAN_VAULT
        self._vault = vault_path or (OBSIDIAN_VAULT / "Concepts")
        self._state_path = (vault_path or OBSIDIAN_VAULT).parent / "obsidian_sync_state.json" \
            if vault_path else OBSIDIAN_VAULT / "obsidian_sync_state.json"
        self._state: Dict[str, dict] = self._load_state()

    def _load_state(self) -> Dict[str, dict]:
        if self._state_path.exists():
            try:
                return json.loads(self._state_path.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning("Sync state load failed: %s", e)
        return {}

    def _save_state(self):
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            self._state_path.write_text(json.dumps(self._state, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning("Sync state save failed: %s", e)

    def _read_note(self, path: Path) -> Optional[str]:
        try:
            return path.read_text(encoding="utf-8")
        except Exception:
            return None

    def _get_note_state(self, path: Path, content: str) -> dict:
        """Build state snapshot for a note."""
        return {
            "mtime": path.stat().st_mtime if path.exists() else 0,
            "relations": _extract_neighbor_relations(content),
            "wikilinks": list(_extract_wikilinks(content)),
        }

    def _node_name_from_path(self, path: Path) -> str:
        """Derive concept node name from filename."""
        return path.stem.replace("_", " ")

    def sync_vault_changes(self, graph=None) -> dict:
        """
        Scan vault for modified notes. Diff wikilinks/relations against stored state.
        Apply changes to the graph as high-confidence human edits.

        Returns {files_scanned, edges_added, edges_removed, files_updated}.
        """
        if not self._vault.exists():
            return {"files_scanned": 0, "edges_added": 0, "edges_removed": 0, "files_updated": 0}

        if graph is None:
            from src.concept_graph import ConceptGraph
            graph = ConceptGraph()

        from src.language_layer import CONFIDENCE_HUMAN
        stats = {"files_scanned": 0, "edges_added": 0, "edges_removed": 0, "files_updated": 0}

        md_files = list(self._vault.glob("*.md"))
        stats["files_scanned"] = len(md_files)

        for md_path in md_files:
            note_key = md_path.name
            try:
                current_mtime = md_path.stat().st_mtime
            except Exception:
                continue

            prev_state = self._state.get(note_key, {})
            prev_mtime = prev_state.get("mtime", 0)

            # Skip if not modified since last sync
            if current_mtime <= prev_mtime:
                continue

            content = self._read_note(md_path)
            if content is None:
                continue

            node_name = self._node_name_from_path(md_path)
            curr_relations = _extract_neighbor_relations(content)
            prev_relations = prev_state.get("relations", {})

            # Detect added edges
            for neighbor, relation in curr_relations.items():
                if neighbor not in prev_relations:
                    graph.ingest_triples(
                        [(node_name, relation, neighbor)],
                        source="obsidian_human",
                        default_confidence=CONFIDENCE_HUMAN,
                        source_type="human",
                        provenance=str(md_path),
                    )
                    stats["edges_added"] += 1
                    logger.info("Human edit: +(%s, %s, %s)", node_name, relation, neighbor)

            # Detect removed edges
            for neighbor, relation in prev_relations.items():
                if neighbor not in curr_relations:
                    # Reduce confidence; if very low, the decay will remove it naturally
                    self._reduce_edge_confidence(graph, node_name, relation, neighbor)
                    stats["edges_removed"] += 1
                    logger.info("Human edit: -(%s, %s, %s)", node_name, relation, neighbor)

            # Update state
            new_state = self._get_note_state(md_path, content)
            self._state[note_key] = new_state
            stats["files_updated"] += 1

        self._save_state()
        return stats

    def _reduce_edge_confidence(
        self, graph, src: str, relation: str, dst: str, penalty: float = 0.4
    ):
        """Reduce confidence on an edge removed by human. Removes if below threshold."""
        REMOVE_THRESHOLD = 0.15
        if graph._use_neo4j:
            recs = graph.cypher(
                "MATCH (a:Concept {name:$s})-[e:RELATES {type:$r}]->(b:Concept {name:$o}) "
                "RETURN e.confidence as c",
                {"s": src, "r": relation, "o": dst},
            )
            if recs:
                old_conf = recs[0].get("c", 0.6)
                new_conf = old_conf - penalty
                if new_conf < REMOVE_THRESHOLD:
                    graph.cypher(
                        "MATCH (a:Concept {name:$s})-[e:RELATES {type:$r}]->(b:Concept {name:$o}) DELETE e",
                        {"s": src, "r": relation, "o": dst},
                    )
                else:
                    graph.cypher(
                        "MATCH (a:Concept {name:$s})-[e:RELATES {type:$r}]->(b:Concept {name:$o}) SET e.confidence=$c",
                        {"s": src, "r": relation, "o": dst, "c": round(new_conf, 4)},
                    )
        else:
            G = graph._nx_graph
            if G.has_edge(src, dst):
                d = G[src][dst]
                if d.get("type") == relation:
                    old_conf = d.get("confidence", 0.6)
                    new_conf = old_conf - penalty
                    if new_conf < REMOVE_THRESHOLD:
                        G.remove_edge(src, dst)
                    else:
                        d["confidence"] = round(new_conf, 4)
            graph._save_nx_to_disk()

    def initialize_state(self) -> int:
        """
        Build initial state snapshot from current vault contents.
        Run once before first sync to establish baseline.
        Returns number of files indexed.
        """
        if not self._vault.exists():
            return 0
        count = 0
        for md_path in self._vault.glob("*.md"):
            content = self._read_note(md_path)
            if content is None:
                continue
            self._state[md_path.name] = self._get_note_state(md_path, content)
            count += 1
        self._save_state()
        logger.info("Obsidian sync state initialized: %d files", count)
        return count


def sync_obsidian_to_graph(vault_path: Path = None, graph=None) -> dict:
    """Convenience function: run one sync pass."""
    syncer = ObsidianSync(vault_path)
    return syncer.sync_vault_changes(graph)
