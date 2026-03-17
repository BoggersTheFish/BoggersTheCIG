"""
Subsystem 5: Validation Engine
Logical consistency, graph stability, empirical compatibility, bias checks.
"""
import logging
from typing import List, Tuple

from src.config import HARMFUL_PATTERNS
from src.concept_graph import ConceptGraph
from src.core_engine import CoreEngine

logger = logging.getLogger(__name__)


class ValidationEngine:
    """
    Validate hypotheses: logical consistency, stability, empirical check, bias.
    """

    def __init__(self, graph: ConceptGraph = None):
        self.graph = graph or ConceptGraph()
        self.core = CoreEngine(self.graph)

    def check_logical_consistency(self, triple: Tuple[str, str, str]) -> bool:
        """
        No cycles/conflicts. E.g. (A, causes, B) and (B, causes, A) is inconsistent.
        """
        s, r, o = triple
        if s == o and r in ("causes", "implies"):
            return False
        conflicts = self.core.constraint_resolution()
        for c in conflicts:
            if s in c.get("nodes", []) and o in c.get("nodes", []):
                return False
        return True

    def check_graph_stability(self, triple: Tuple[str, str, str]) -> bool:
        """Degree thresholds: avoid connecting to isolated nodes that could be noise."""
        # Allow for now; can add degree thresholds later
        return True

    def check_empirical(self, triple: Tuple[str, str, str]) -> bool:
        """
        Cross-check with external APIs (e.g. Wikipedia) if available.
        Pass-through: no external API wired; override to add empirical validation.
        """
        return True

    def check_bias(self, triple: Tuple[str, str, str]) -> bool:
        """Reject triples containing harmful/discriminatory patterns."""
        text = f"{triple[0]} {triple[1]} {triple[2]}".lower()
        if any(p in text for p in HARMFUL_PATTERNS):
            return False
        return True

    def validate(self, triple: Tuple[str, str, str]) -> bool:
        """Run all checks. Returns True if valid."""
        if not self.check_bias(triple):
            logger.debug("Rejected (bias): %s", triple)
            return False
        if not self.check_logical_consistency(triple):
            logger.debug("Rejected (logic): %s", triple)
            return False
        if not self.check_graph_stability(triple):
            logger.debug("Rejected (stability): %s", triple)
            return False
        if not self.check_empirical(triple):
            logger.debug("Rejected (empirical): %s", triple)
            return False
        return True

    def validate_batch(self, triples: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
        """Validate list of triples, return only valid ones."""
        return [t for t in triples if self.validate(t)]

    def prune_invalid(self) -> int:
        """
        Remove invalid nodes/edges from graph.
        Returns count of pruned items.
        """
        # For now, only prune contradictory edges
        conflicts = self.core.constraint_resolution()
        pruned = 0
        for c in conflicts:
            nodes = c.get("nodes", [])
            if len(nodes) == 2:
                a, b = nodes
                if self.graph._use_neo4j:
                    self.graph.cypher(
                        "MATCH (a:Concept {name: $a})-[r:RELATES]-(b:Concept {name: $b}) WHERE r.type = 'causes' DELETE r",
                        {"a": a, "b": b},
                    )
                    pruned += 1
                else:
                    if self.graph._nx_graph.has_edge(a, b):
                        self.graph._nx_graph.remove_edge(a, b)
                        pruned += 1
                    if self.graph._nx_graph.has_edge(b, a):
                        self.graph._nx_graph.remove_edge(b, a)
                        pruned += 1
        return pruned
