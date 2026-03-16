"""
Subsystem 11: Self-Improvement Mechanism
Monitor outcomes, detect inefficient patterns, rewrite rules via LLM.
"""
import json
import logging
from pathlib import Path

from src.config import EVAL_DIR, SELF_IMPROVE_INTERVAL
from src.memory_layers import MemoryLayers

logger = logging.getLogger(__name__)

_SUCCESS_LOG = EVAL_DIR / "hypothesis_success.json"


class SelfImprover:
    """
    Analyze meta-layer, hypothesis logs; generate improved rules.
    """

    def __init__(self):
        self.memory = MemoryLayers()

    def analyze_logs(self) -> dict:
        """Compute success rate and patterns from hypothesis log."""
        if not _SUCCESS_LOG.exists():
            return {"acceptance_rate": 0.0, "total": 0}
        try:
            data = json.loads(_SUCCESS_LOG.read_text())
            accepted = sum(1 for e in data if e.get("accepted"))
            return {"acceptance_rate": accepted / len(data) if data else 0, "total": len(data)}
        except Exception as e:
            logger.warning("Could not analyze logs: %s", e)
            return {"acceptance_rate": 0.0, "total": 0}

    def run(self):
        """
        Detect inefficient patterns, optionally use LLM to suggest new rules.
        Store in meta-layer.
        """
        stats = self.analyze_logs()
        logger.info("Self-improve: acceptance_rate=%.2f total=%d", stats["acceptance_rate"], stats["total"])

        # Simple rule: if acceptance rate is low, add a "be more conservative" meta-rule
        if stats["total"] > 10 and stats["acceptance_rate"] < 0.3:
            rule = {
                "type": "conservative_hypothesis",
                "description": "Lower hypothesis generation rate when acceptance is low",
                "params": {"min_acceptance": 0.3},
            }
            self.memory.set_meta_rule(rule)
            logger.info("Added meta-rule: %s", rule["type"])
