"""
Ollama integration for local LLM calls.
Uses qwen2.5-coder:7b or llama3.1:8b for TS analysis and code generation.
Supports multi-port parallel instances and resource-pressure downscaling.
"""
import json
import logging
import re
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.config import OLLAMA_URL, OLLAMA_MODEL, OLLAMA_URLS, RESOURCE_PRESSURE_THRESHOLD

logger = logging.getLogger(__name__)

_url_index = 0
_url_lock = threading.Lock()


def _get_next_ollama_url() -> str:
    """Round-robin across configured Ollama URLs for load distribution."""
    if len(OLLAMA_URLS) <= 1:
        return OLLAMA_URLS[0] if OLLAMA_URLS else OLLAMA_URL
    with _url_lock:
        global _url_index
        url = OLLAMA_URLS[_url_index % len(OLLAMA_URLS)]
        _url_index += 1
        return url


def _get_ollama_model() -> str:
    """Return configured model or first available if configured not found."""
    try:
        url = f"{OLLAMA_URL.rstrip('/')}/api/tags"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            models = [m.get("name", "") for m in data.get("models", [])]
            for m in models:
                if OLLAMA_MODEL in m or (":" in OLLAMA_MODEL and m.startswith(OLLAMA_MODEL.split(":")[0])):
                    return m
            if models:
                return models[0]
    except Exception:
        pass
    return OLLAMA_MODEL


def _ollama_request(prompt: str, model: str = None, system: str = None, timeout: int = 120, url: str = None) -> str:
    """Call Ollama API. Returns generated text or empty string on failure."""
    from src.hardware_adapt import check_resource_pressure, select_downscaled_model

    if check_resource_pressure(RESOURCE_PRESSURE_THRESHOLD):
        select_downscaled_model(RESOURCE_PRESSURE_THRESHOLD)

    model = model or _get_ollama_model()
    base = url or _get_next_ollama_url()
    api_url = f"{base.rstrip('/')}/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}
    if system:
        payload["system"] = system
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(api_url, data=data, method="POST", headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            out = json.loads(resp.read().decode())
            return out.get("response", "")
    except urllib.error.URLError as e:
        logger.error("Ollama request failed: %s", e)
        return ""
    except json.JSONDecodeError as e:
        logger.error("Ollama response parse failed: %s", e)
        return ""


def parallel_ollama_requests(
    prompts: list[str],
    system: str = None,
    timeout: int = 60,
    max_workers: int = None,
) -> list[str]:
    """
    Run multiple Ollama requests in parallel across configured URLs.
    Each prompt is sent to a different URL (round-robin). Returns list of responses.
    """
    if not prompts:
        return []
    urls = OLLAMA_URLS if len(OLLAMA_URLS) > 1 else [OLLAMA_URL] * len(prompts)
    workers = min(max_workers or len(OLLAMA_URLS) or 1, len(prompts), len(urls) or 1)
    results = [""] * len(prompts)

    def _run_one(i: int, prompt: str, base_url: str) -> tuple[int, str]:
        resp = _ollama_request(prompt, system=system, timeout=timeout, url=base_url)
        return (i, resp)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(_run_one, i, p, urls[i % len(urls)]): i
            for i, p in enumerate(prompts)
        }
        for fut in as_completed(futures):
            try:
                i, resp = fut.result()
                results[i] = resp
            except Exception as e:
                idx = futures[fut]
                logger.warning("Parallel Ollama request %d failed: %s", idx, e)
    return results


def analyze_repo_for_tensions(repo_summary: str) -> str:
    """
    Ask Ollama to analyze repo for structural tensions, gaps, performance issues.
    Returns analysis text.
    """
    system = """You are a TS-aligned code analyst. Identify:
1. Structural tensions: cycles, contradictions, orphan nodes in graph logic
2. Gaps: missing error handling, untested branches, placeholder code
3. Performance: slow paths, redundant work
4. Alignment: ethical drift, bias risks
Output concise bullet points. No fluff."""
    prompt = f"Analyze this repo summary for improvement opportunities:\n\n{repo_summary[:4000]}"
    return _ollama_request(prompt, system=system)


def generate_code_hypothesis(analysis: str, file_path: str, context: str) -> str:
    """
    Ask Ollama to generate a concrete code change (diff or full snippet).
    Returns proposed code or diff text.
    """
    system = """You are a TS-aligned coder. Produce real, functional code. No stubs, TODOs, or placeholders.
Respect: graph stability (check conflicts before mutations), ethics (bias checks), cost (local-first).
Output only the code change, no explanatory prose."""
    prompt = f"""Analysis: {analysis[:1500]}

File: {file_path}
Context:
{context[:2000]}

Propose a concrete code change (full function or diff-style). Output only code."""
    return _ollama_request(prompt, system=system, timeout=180)


_EXTRACT_SYSTEM = """Extract factual concept triples from the text. Each triple is (Subject, Relation, Object).
Output ONLY a list of triples, one per line, format: (Subject, Relation, Object)
No harmful, violent, or discriminatory content. Scientific/factual entities only."""


def _parse_triples_from_response(response: str) -> list:
    """Parse triples from Ollama response text."""
    triples = []
    for line in response.strip().split("\n"):
        m = re.search(r"\(([^,]+),\s*([^,]+),\s*([^)]+)\)", line)
        if m:
            s, r, o = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
            if len(s) > 1 and len(o) > 1:
                triples.append((s, r, o))
    return triples


def extract_triples_from_text(text: str) -> list:
    """
    Use Ollama to extract (Subject, Relation, Object) triples from text.
    Returns list of tuples. Structured output for knowledge ingestion.
    """
    prompt = f"Text:\n{text[:3000]}\n\nTriples:"
    response = _ollama_request(prompt, system=_EXTRACT_SYSTEM, timeout=60)
    return _parse_triples_from_response(response)


def extract_triples_from_text_batch(texts: list[str], flatten: bool = True) -> list:
    """
    Extract triples from multiple texts in parallel across Ollama instances.
    If flatten=True, returns flattened list of (Subject, Relation, Object).
    If flatten=False, returns list of lists (one per input text).
    """
    if not texts:
        return [] if flatten else []
    if len(texts) == 1:
        t = extract_triples_from_text(texts[0])
        return t if flatten else [t]
    prompts = [f"Text:\n{t[:3000]}\n\nTriples:" for t in texts]
    responses = parallel_ollama_requests(prompts, system=_EXTRACT_SYSTEM, timeout=60)
    per_text = [_parse_triples_from_response(r) for r in responses]
    return [x for lst in per_text for x in lst] if flatten else per_text


def is_duplicate_subidea(text1: str, text2: str, use_embeddings: bool = True, threshold: float = 0.85) -> bool:
    """
    Determine if two sub-idea texts are duplicates (same or very similar).
    Uses embeddings cosine similarity if available (>= threshold), else Ollama judgment.
    Returns True if duplicate.
    """
    if use_embeddings:
        try:
            from src.concept_graph import _embed
            a, b = _embed(text1[:2000]), _embed(text2[:2000])
            import numpy as np
            x, y = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
            sim = float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-9))
            return sim >= threshold
        except Exception:
            pass
    prompt = f"""Two text blocks. Are they about the SAME sub-idea (duplicate/semantically equivalent)?
Reply ONLY with YES or NO.

Text 1:
{text1[:800]}

Text 2:
{text2[:800]}

YES or NO?"""
    resp = _ollama_request(prompt, timeout=30).strip().upper()
    return "YES" in resp[:10]


def is_meaningful_subidea(text: str, word_count: int = 0) -> bool:
    """
    Decide if text is a meaningful sub-idea worth extracting into its own file.
    Criteria: >=150 words OR has internal structure (sections, bullets, subsections).
    Uses Ollama when available for semantic judgment.
    """
    words = len(text.split())
    if words >= 150:
        return True
    if re.search(r"^#{2,3}\s+\w", text, re.MULTILINE) or re.search(r"^\s*[-*]\s+.{20,}", text, re.MULTILINE):
        if words >= 50:
            return True
    try:
        if check_ollama_available():
            prompt = f"""Is this a meaningful sub-idea worth extracting into its own note?
(Standalone concept, detailed explanation, or coherent section with internal structure)
Reply ONLY with YES or NO.

Text ({words} words):
{text[:600]}

YES or NO?"""
            resp = _ollama_request(prompt, timeout=20).strip().upper()
            return "YES" in resp[:10]
    except Exception:
        pass
    return words >= 150


def suggest_extraction_name(text: str) -> str:
    """
    Suggest a short kebab-case filename for a sub-idea.
    Uses Ollama when available; fallback: first 3 words lowercased, spaces to hyphens.
    """
    try:
        if check_ollama_available():
            prompt = f"""Suggest a short filename (kebab-case, no spaces) for this sub-idea.
One to four words. No file extension. Example: quantum-mechanics, wave-particle-duality.

Text:
{text[:400]}

Filename only:"""
            resp = _ollama_request(prompt, timeout=15).strip()
            name = re.sub(r"[^\w\-]", "", resp.replace(" ", "-").lower())[:50]
            if name:
                return name
    except Exception:
        pass
    words = text.split()[:4]
    return "-".join(w.lower() for w in words if w.isalnum())[:50] or "sub-idea"


def check_ollama_available() -> bool:
    """Verify Ollama is running and model is available."""
    try:
        url = f"{OLLAMA_URL.rstrip('/')}/api/tags"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            models = [m.get("name", "") for m in data.get("models", [])]
            for m in models:
                if OLLAMA_MODEL in m or m.startswith(OLLAMA_MODEL.split(":")[0]):
                    return True
            if models:
                logger.warning("Ollama running but %s not found. Available: %s", OLLAMA_MODEL, models[:3])
            return bool(models)
    except Exception as e:
        logger.warning("Ollama check failed: %s", e)
        return False
