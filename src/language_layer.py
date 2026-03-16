"""
Subsystem 1: Language Layer
Translates human text to concept triples (Subject, Relation, Object) via LLM.
Uses chain-of-thought for accuracy; filters biases via ethical patterns.
"""
import logging
import re
from typing import List, Tuple, Optional

from src.config import LLM_MODEL, USE_4BIT, HARMFUL_PATTERNS

logger = logging.getLogger(__name__)

# Lazy-loaded LLM components (avoid heavy load at import)
_llm_cache: dict = {}


def _get_llm():
    """Load and cache LLM. Returns (tokenizer, model) or None for fallback."""
    global _llm_cache
    if "model" in _llm_cache:
        return _llm_cache["tokenizer"], _llm_cache["model"]
    if "fallback" in _llm_cache:
        return None
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        logger.info("Loading LLM: %s (4bit=%s)", LLM_MODEL, USE_4BIT)
        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL, trust_remote_code=True)
        model_kwargs = {"trust_remote_code": True}
        if USE_4BIT and torch.cuda.is_available():
            model_kwargs["load_in_4bit"] = True
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["torch_dtype"] = torch.float32
        model = AutoModelForCausalLM.from_pretrained(LLM_MODEL, **model_kwargs)
        _llm_cache["tokenizer"] = tokenizer
        _llm_cache["model"] = model
        return tokenizer, model
    except Exception as e:
        logger.warning("LLM load failed (%s), using rule-based fallback", e)
        _llm_cache["fallback"] = True
        return None


def _filter_harmful(triples: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
    """Filter triples containing harmful patterns."""
    filtered = []
    for s, r, o in triples:
        text = f"{s} {r} {o}".lower()
        if any(p in text for p in HARMFUL_PATTERNS):
            logger.debug("Filtered harmful triple: (%s, %s, %s)", s, r, o)
            continue
        filtered.append((s, r, o))
    return filtered


def _rule_based_extract(text: str) -> List[Tuple[str, str, str]]:
    """Fallback: simple rule-based triple extraction (no LLM)."""
    triples = []
    # Split on sentences
    sentences = re.split(r"[.!?]+", text)
    for sent in sentences:
        sent = sent.strip()
        if len(sent) < 3:
            continue
        # Question patterns first (to avoid "Why is X" matching is_a)
        question_patterns = [
            (r"explain\s+(\w+(?:\s+\w+)*)", "explainable"),
            (r"why\s+(?:is|does|do)\s+(\w+(?:\s+\w+)*)\s+(\w+(?:\s+\w+)*)", "has_property"),
            (r"what\s+causes?\s+(\w+(?:\s+\w+)*)", "explainable"),
            (r"how\s+does\s+(\w+(?:\s+\w+)*)\s+work", "explainable"),
            (r"how\s+does\s+(\w+(?:\s+\w+)*)\s+(?:affect|relate)\s+(\w+(?:\s+\w+)*)", "affects"),
        ]
        # Factual patterns
        patterns = [
            (r"(\w+(?:\s+\w+)*)\s+is\s+(?:a\s+)?(\w+(?:\s+\w+)*)", "is_a"),
            (r"(\w+(?:\s+\w+)*)\s+has\s+(\w+(?:\s+\w+)*)", "has"),
            (r"(\w+(?:\s+\w+)*)\s+causes?\s+(\w+(?:\s+\w+)*)", "causes"),
            (r"(\w+(?:\s+\w+)*)\s+bends?\s+(\w+(?:\s+\w+)*)", "bends"),
            (r"(\w+(?:\s+\w+)*)\s+affects?\s+(\w+(?:\s+\w+)*)", "affects"),
            (r"(\w+(?:\s+\w+)*)\s+influences?\s+(\w+(?:\s+\w+)*)", "influences"),
            (r"(\w+(?:\s+\w+)*)\s+relates?\s+to\s+(\w+(?:\s+\w+)*)", "relates_to"),
            (r"(\w+(?:\s+\w+)*)\s+and\s+(\w+(?:\s+\w+)*)", "related_to"),
        ]
        all_patterns = question_patterns + patterns
        for pattern, rel in all_patterns:
            for m in re.finditer(pattern, sent, re.I):
                groups = [g.strip() if g else "" for g in m.groups()]
                if rel == "explainable":
                    s, o = groups[0], "topic"
                elif len(groups) >= 2 and groups[1]:
                    s, o = groups[0], groups[1]
                else:
                    continue
                if len(s) > 1 and (len(o) > 1 or rel == "explainable") and s.lower() != o.lower():
                    triples.append((s, rel, o))
    # Drop triples with question words as subject (Why, What, How, etc.)
    stop_subjects = {"why", "what", "how", "when", "where", "which", "who"}
    triples = [(s, r, o) for s, r, o in triples if s.lower().split()[0] not in stop_subjects]
    return triples


def _is_question(text: str) -> bool:
    """Detect if input is a question (explanation request, why/what/how, etc.)."""
    t = text.strip().lower()
    return (
        t.endswith("?")
        or t.startswith(("explain", "why", "what", "how", "when", "where", "which"))
        or " why " in t
        or " what " in t
        or " how " in t
    )


def _build_prompt(text: str) -> str:
    """Build LLM prompt; use question-aware template when input is a query/explanation request."""
    text_slice = text[:500]
    if _is_question(text):
        return f"""Extract factual concept triples from this query/explanation: '{text_slice}'. Focus on scientific entities and relations, not the question phrasing itself. Output as list of (Subject, Predicate, Object). One triple per line."""

    return f"""Extract concept triples from the following text. Each triple is (Subject, Relation, Object).
Output ONLY a list of triples, one per line, format: (Subject, Relation, Object)

Text: {text_slice}

Triples:"""


def extract_triples(text: str, use_llm: bool = True) -> List[Tuple[str, str, str]]:
    """
    Extract concept triples (Subject, Relation, Object) from text.
    Uses LLM when available; falls back to rule-based extraction.
    Applies ethical filtering.
    """
    if not text or not text.strip():
        return []

    triples: List[Tuple[str, str, str]] = []

    if use_llm:
        llm = _get_llm()
        if llm is not None:
            tokenizer, model = llm
            try:
                import torch

                prompt = _build_prompt(text)
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
                if hasattr(model, "device"):
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
                response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                for line in response.strip().split("\n"):
                    m = re.search(r"\(([^,]+),\s*([^,]+),\s*([^)]+)\)", line)
                    if m:
                        triples.append((m.group(1).strip(), m.group(2).strip(), m.group(3).strip()))
            except Exception as e:
                logger.warning("LLM extraction failed: %s", e)
                triples = _rule_based_extract(text)
        else:
            triples = _rule_based_extract(text)
    else:
        triples = _rule_based_extract(text)

    triples = _filter_harmful(triples)
    return triples
