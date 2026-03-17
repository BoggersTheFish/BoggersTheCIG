"""
Subsystem 12: Data Curation / Training
Seed from free datasets (Wikipedia), preprocess, triples, debias.
"""
import logging
from pathlib import Path
from typing import List, Tuple

from src.config import DATA_DIR
from src.language_layer import extract_triples
from src.concept_graph import ConceptGraph
from src.validation_engine import ValidationEngine

logger = logging.getLogger(__name__)


def load_wikipedia_seed(limit: int = 100) -> List[str]:
    """Load Wikipedia subset from Hugging Face datasets (microsoft/wiki_qa)."""
    try:
        from datasets import load_dataset
        ds = load_dataset("microsoft/wiki_qa", split="train", trust_remote_code=True)
        texts = []
        for i, row in enumerate(ds):
            if len(texts) >= limit:
                break
            q = row.get("question", "")
            a = row.get("answer", "")
            if q:
                texts.append(str(q))
            if a and str(a).strip():
                texts.append(str(a))
        return texts[:limit]
    except Exception as e:
        logger.warning("wiki_qa load failed: %s, using fallback", e)
        return _fallback_seed()


def _fallback_seed() -> List[str]:
    """Fallback seed data when HF unavailable."""
    return [
        "Gravity bends spacetime. Einstein discovered general relativity.",
        "Water is composed of hydrogen and oxygen. Water freezes at zero degrees Celsius.",
        "Quantum physics describes particles at the atomic scale. Light behaves as both wave and particle.",
        "Photosynthesis converts sunlight into energy. Plants use chlorophyll for photosynthesis.",
    ]


def run_pipeline(limit: int = 50, ingest: bool = True) -> List[Tuple[str, str, str]]:
    """
    Load seed data, extract triples, optionally ingest into graph.
    Returns all triples.
    """
    texts = load_wikipedia_seed(limit=limit)
    all_triples = []
    for text in texts:
        triples = extract_triples(text, use_llm=False)
        all_triples.extend(triples)
    if ingest and all_triples:
        graph = ConceptGraph()
        val = ValidationEngine(graph)
        valid = val.validate_batch(all_triples)
        graph.ingest_triples(valid, source="pipeline")
        logger.info("Ingested %d triples from pipeline (validated %d)", len(valid), len(all_triples))
    return all_triples
