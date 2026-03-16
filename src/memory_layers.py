"""
Subsystem 8: Memory Architecture
Layers: Raw (Redis), Structured (graph), Hypothesis, Meta.
"""
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.config import MEMORY_DIR, REDIS_URL

logger = logging.getLogger(__name__)

_redis_client = None


def _get_redis():
    """Lazy Redis connection."""
    global _redis_client
    if _redis_client is not None:
        return _redis_client
    try:
        import redis
        _redis_client = redis.from_url(REDIS_URL)
        _redis_client.ping()
        return _redis_client
    except Exception as e:
        logger.warning("Redis unavailable (%s), using file-based raw layer", e)
        return None


class MemoryLayers:
    """
    Raw -> Structured -> Hypothesis -> Meta.
    Raw: Redis list or file-based queue.
    Structured: main graph (handled by concept_graph).
    Hypothesis: separate namespace/labels.
    Meta: rules, reasoning strategies.
    """

    RAW_KEY = "full_ts:raw_inputs"
    META_KEY = "full_ts:meta_rules"

    def __init__(self):
        self.redis = _get_redis()
        self.raw_dir = MEMORY_DIR / "raw"
        self.hypothesis_dir = MEMORY_DIR / "hypothesis"
        self.meta_dir = MEMORY_DIR / "meta"
        for d in [self.raw_dir, self.hypothesis_dir, self.meta_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def push_raw(self, item: Union[str, Dict]) -> bool:
        """Add to raw input layer."""
        data = json.dumps(item) if isinstance(item, dict) else item
        if self.redis:
            try:
                self.redis.lpush(self.RAW_KEY, data)
                return True
            except Exception as e:
                logger.warning("Redis push failed: %s", e)
        (self.raw_dir / f"{int(time.time() * 1000)}.json").write_text(
            json.dumps({"content": data, "ts": time.time()})
        )
        return True

    def pop_raw(self) -> Optional[str]:
        """Pop from raw layer (FIFO for processing)."""
        if self.redis:
            try:
                return self.redis.rpop(self.RAW_KEY)
            except Exception:
                pass
        files = sorted(self.raw_dir.glob("*.json"))
        if not files:
            return None
        f = files[-1]
        try:
            data = json.loads(f.read_text())
            f.unlink()
            return data.get("content", "")
        except Exception:
            return None

    def raw_count(self) -> int:
        if self.redis:
            try:
                return self.redis.llen(self.RAW_KEY)
            except Exception:
                pass
        return len(list(self.raw_dir.glob("*.json")))

    def store_hypothesis(self, triple: List, accepted: bool, metadata: Dict = None):
        """Store hypothesis outcome."""
        path = self.hypothesis_dir / f"{int(time.time() * 1000)}.json"
        path.write_text(json.dumps({
            "triple": triple,
            "accepted": accepted,
            "metadata": metadata or {},
            "ts": time.time(),
        }, indent=2))

    def get_meta_rules(self) -> List[Dict]:
        """Load meta-layer rules."""
        rules = []
        for f in self.meta_dir.glob("*.json"):
            try:
                rules.append(json.loads(f.read_text()))
            except Exception:
                pass
        if self.redis:
            try:
                raw = self.redis.get(self.META_KEY)
                if raw:
                    rules.extend(json.loads(raw))
            except Exception:
                pass
        return rules

    def set_meta_rule(self, rule: Dict):
        """Add/update meta rule."""
        rules = self.get_meta_rules()
        rules.append(rule)
        path = self.meta_dir / f"rule_{int(time.time())}.json"
        path.write_text(json.dumps(rule, indent=2))
        if self.redis:
            try:
                self.redis.set(self.META_KEY, json.dumps(rules[-100:]))
            except Exception:
                pass

    def decay_prune(self, graph, max_idle_seconds: int = 86400 * 30) -> int:
        """
        Prune low-access nodes (decay).
        Returns count pruned.
        """
        # Handled by concept_graph / validation; this is a placeholder for meta-decay
        return 0
