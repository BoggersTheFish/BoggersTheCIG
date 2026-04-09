"""
Prompt Evolution Engine — automated prompt optimization for triple extraction.

Maintains a pool of candidate extraction prompts, tracks per-prompt quality
(average confidence of triples extracted using that prompt), and evolves
the pool by:
  1. Retiring the lowest-performing prompt after RETIRE_AFTER uses
  2. Generating a variation of the best prompt via Ollama
  3. Storing the winning prompts persistently

Usage:
  registry = PromptRegistry()
  prompt = registry.get_prompt()          # get best current prompt
  registry.record_result(prompt_id, confidence)  # feedback after extraction
  registry.maybe_evolve()                 # evolve pool if threshold reached

The registry is wired into language_layer.extract_triples_with_confidence().
"""
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_REGISTRY_PATH = Path(__file__).resolve().parent.parent / "memory" / "prompt_registry.json"

RETIRE_AFTER = 100      # Retire worst prompt after this many uses across all prompts
EVOLVE_EVERY = 50       # Attempt evolution every N total recordings
MIN_USES_TO_RANK = 10   # Minimum uses before a prompt can be ranked

# Seed prompts — different styles to A/B test
_SEED_PROMPTS = [
    {
        "id": "standard_v1",
        "template": (
            "Extract concept triples from the following text. "
            "Each triple is (Subject, Relation, Object). "
            "Output ONLY a list of triples, one per line, format: (Subject, Relation, Object)\n\n"
            "Text: {text}\n\nTriples:"
        ),
        "description": "Standard extraction prompt",
        "uses": 0,
        "total_confidence": 0.0,
        "created_at": 0,
    },
    {
        "id": "specific_v1",
        "template": (
            "Extract precise factual relationships from this text as (Subject, Relation, Object) triples. "
            "Use specific, concrete relations like 'causes', 'is_a', 'part_of', 'produces', 'enables'. "
            "Avoid vague relations like 'related_to'. One triple per line.\n\n"
            "Text: {text}\n\nTriples:"
        ),
        "description": "Specificity-focused prompt",
        "uses": 0,
        "total_confidence": 0.0,
        "created_at": 0,
    },
    {
        "id": "chain_of_thought_v1",
        "template": (
            "Read the following text carefully. Identify the key concepts and the relationships between them. "
            "For each relationship, output a triple as (Subject, Relation, Object). "
            "Focus on causal, hierarchical, and functional relationships. "
            "Output only the triples, one per line.\n\n"
            "Text: {text}\n\nTriples:"
        ),
        "description": "Chain-of-thought style prompt",
        "uses": 0,
        "total_confidence": 0.0,
        "created_at": 0,
    },
    {
        "id": "scientific_v1",
        "template": (
            "You are a scientific knowledge extractor. Extract structured knowledge triples from the text. "
            "Format: (Subject, Relation, Object). Use scientific terminology where appropriate. "
            "Prefer relations: causes, inhibits, activates, regulates, produces, requires, is_a, part_of. "
            "One triple per line.\n\n"
            "Text: {text}\n\nTriples:"
        ),
        "description": "Scientific domain prompt",
        "uses": 0,
        "total_confidence": 0.0,
        "created_at": 0,
    },
    {
        "id": "compressed_v1",
        "template": (
            "List knowledge triples (S, R, O) from this text. Short, precise. One per line.\n\n"
            "Text: {text}\n\nTriples:"
        ),
        "description": "Minimal compressed prompt",
        "uses": 0,
        "total_confidence": 0.0,
        "created_at": 0,
    },
]


class PromptRegistry:
    """
    Maintains and evolves a pool of triple extraction prompts.
    Tracks quality (avg confidence) per prompt and evolves the pool.
    """

    def __init__(self, registry_path: Path = None):
        self._path = registry_path or _REGISTRY_PATH
        self._prompts: Dict[str, dict] = {}
        self._total_recordings = 0
        self._load()

    def _load(self):
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text(encoding="utf-8"))
                self._prompts = {p["id"]: p for p in data.get("prompts", [])}
                self._total_recordings = data.get("total_recordings", 0)
            except Exception as e:
                logger.warning("PromptRegistry load failed: %s", e)

        # Seed missing prompts
        for seed in _SEED_PROMPTS:
            if seed["id"] not in self._prompts:
                self._prompts[seed["id"]] = dict(seed)
                self._prompts[seed["id"]]["created_at"] = int(time.time())

        self._save()

    def _save(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._path.write_text(
                json.dumps(
                    {"prompts": list(self._prompts.values()), "total_recordings": self._total_recordings},
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning("PromptRegistry save failed: %s", e)

    def get_prompt(self) -> Tuple[str, str]:
        """
        Return (prompt_id, prompt_template) for the best-performing prompt.
        Falls back to round-robin for under-tested prompts.
        """
        # Prefer prompts with fewer than MIN_USES_TO_RANK uses (exploration)
        under_tested = [p for p in self._prompts.values() if p["uses"] < MIN_USES_TO_RANK]
        if under_tested:
            # Pick least-used
            p = min(under_tested, key=lambda x: x["uses"])
            return p["id"], p["template"]

        # Pick highest avg confidence
        best = max(
            self._prompts.values(),
            key=lambda p: (p["total_confidence"] / p["uses"]) if p["uses"] > 0 else 0.0,
        )
        return best["id"], best["template"]

    def format_prompt(self, prompt_id: str, text: str) -> str:
        """Format a prompt with the given text."""
        p = self._prompts.get(prompt_id)
        if not p:
            return _SEED_PROMPTS[0]["template"].format(text=text[:500])
        return p["template"].format(text=text[:500])

    def record_result(self, prompt_id: str, confidence: float):
        """Record extraction quality for a prompt."""
        if prompt_id not in self._prompts:
            return
        p = self._prompts[prompt_id]
        p["uses"] = p.get("uses", 0) + 1
        p["total_confidence"] = p.get("total_confidence", 0.0) + confidence
        self._total_recordings += 1
        self._save()
        # Attempt evolution at threshold
        if self._total_recordings % EVOLVE_EVERY == 0:
            self.maybe_evolve()

    def avg_quality(self, prompt_id: str) -> float:
        """Average confidence for a prompt. 0.0 if unused."""
        p = self._prompts.get(prompt_id, {})
        uses = p.get("uses", 0)
        return (p.get("total_confidence", 0.0) / uses) if uses > 0 else 0.0

    def rankings(self) -> List[dict]:
        """Return prompts sorted by avg quality."""
        return sorted(
            [
                {
                    "id": p["id"],
                    "description": p.get("description", ""),
                    "uses": p["uses"],
                    "avg_quality": round(self.avg_quality(p["id"]), 4),
                }
                for p in self._prompts.values()
                if p["uses"] >= MIN_USES_TO_RANK
            ],
            key=lambda x: -x["avg_quality"],
        )

    def maybe_evolve(self):
        """
        Retire the worst-performing prompt (if pool size > 3) and generate
        a variation of the best prompt via Ollama.
        """
        ranked = self.rankings()
        if len(ranked) < 3:
            return  # Not enough data

        worst_id = ranked[-1]["id"]
        best_id = ranked[0]["id"]
        best_prompt = self._prompts[best_id]

        logger.info(
            "Prompt evolution: retiring '%s' (avg_q=%.3f), evolving from '%s' (avg_q=%.3f)",
            worst_id, ranked[-1]["avg_quality"], best_id, ranked[0]["avg_quality"],
        )

        # Retire worst if pool has > 3 prompts
        if len(self._prompts) > 3:
            del self._prompts[worst_id]

        # Generate variation of best via Ollama
        try:
            from ollama_integration import check_ollama_available, _ollama_request
            if not check_ollama_available():
                return
            evolve_prompt = (
                f"Here is a high-performing prompt for extracting knowledge triples from text:\n\n"
                f"---\n{best_prompt['template']}\n---\n\n"
                "Generate ONE improved variation of this prompt. The variation should extract "
                "more specific and accurate (Subject, Relation, Object) triples. "
                "Keep the {{text}} placeholder. Output ONLY the prompt text, nothing else."
            )
            new_template = _ollama_request(evolve_prompt, timeout=45).strip()
            if new_template and "{text}" in new_template and len(new_template) > 50:
                new_id = f"evolved_{int(time.time())}"
                self._prompts[new_id] = {
                    "id": new_id,
                    "template": new_template,
                    "description": f"Evolved from {best_id}",
                    "uses": 0,
                    "total_confidence": 0.0,
                    "created_at": int(time.time()),
                    "parent_id": best_id,
                }
                logger.info("Evolved new prompt '%s' from '%s'", new_id, best_id)
        except Exception as e:
            logger.debug("Prompt evolution via Ollama failed: %s", e)

        self._save()

    def stats(self) -> dict:
        return {
            "pool_size": len(self._prompts),
            "total_recordings": self._total_recordings,
            "rankings": self.rankings(),
        }


# Module-level singleton
_registry: Optional[PromptRegistry] = None


def get_registry() -> PromptRegistry:
    global _registry
    if _registry is None:
        _registry = PromptRegistry()
    return _registry
