"""
Ollama analyzes Obsidian vault directly.
Reads vault, parses markdown, chunks, sends to Ollama for concept extraction.
Ingests triples into concept graph. Detects contradictions across notes.
"""
import json
import logging
import re
from pathlib import Path
from typing import List, Tuple

from src.config import OBSIDIAN_VAULT, VAULT_MAX_FILES
from src.concept_graph import ConceptGraph
from src.validation_engine import ValidationEngine
from src.core_engine import CoreEngine
from src.language_layer import _filter_harmful

logger = logging.getLogger(__name__)

_CHUNK_SIZE = 1500
_CHUNK_OVERLAP = 200


def _parse_markdown_to_text(md_path: Path) -> str:
    """Extract plain text from markdown file. Strips wikilinks, headers, code blocks."""
    text = md_path.read_text(encoding="utf-8", errors="replace")
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"^#+\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"```[\s\S]*?```", " ", text)
    text = re.sub(r"`[^`]+`", " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"[*_~`#\[\]]", " ", text)
    return " ".join(text.split())


def _extract_triples_from_vault_format(md_path: Path, raw_text: str) -> List[Tuple[str, str, str]]:
    """Extract triples from Obsidian neighbor format: # Node\n## Neighbors\n- [[X]] (rel, weight=N)."""
    triples = []
    title_match = re.search(r"^#\s+(.+)$", raw_text, re.MULTILINE)
    subject = title_match.group(1).strip() if title_match else md_path.stem
    for m in re.finditer(r"-\s*\[\[([^\]]+)\]\]\s*\(([^,]+)", raw_text):
        obj, rel = m.group(1).strip(), m.group(2).strip()
        if subject and obj and rel:
            triples.append((subject, rel, obj))
    return triples


def _chunk_text(text: str, chunk_size: int = _CHUNK_SIZE, overlap: int = _CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start = end - overlap
    return chunks


def _extract_triples_ollama(text: str) -> List[Tuple[str, str, str]]:
    """Use Ollama to extract triples from text."""
    try:
        from ollama_integration import extract_triples_from_text, check_ollama_available
        if check_ollama_available():
            return extract_triples_from_text(text)
    except ImportError:
        pass
    from src.language_layer import extract_triples
    return extract_triples(text, use_llm=False)


def _extract_triples_ollama_batch(chunks: List[str]) -> List[Tuple[str, str, str]]:
    """Use Ollama to extract triples from multiple chunks in parallel when possible."""
    try:
        from ollama_integration import extract_triples_from_text_batch, check_ollama_available
        if check_ollama_available() and chunks:
            return extract_triples_from_text_batch(chunks)
    except ImportError:
        pass
    result = []
    for chunk in chunks:
        result.extend(_extract_triples_ollama(chunk))
    return result


def _detect_contradictions(graph: ConceptGraph) -> List[dict]:
    """Detect contradictions across graph (e.g. bidirectional causes)."""
    core = CoreEngine(graph)
    return core.constraint_resolution()


def analyze_vault(vault_path: Path = None, use_ollama: bool = True) -> dict:
    """
    Read vault, parse markdown, extract concepts via Ollama, ingest into graph.
    Returns stats: {files_read, chunks_processed, triples_extracted, triples_ingested, contradictions}.
    """
    vault = vault_path or OBSIDIAN_VAULT
    if not vault.exists():
        logger.warning("Vault not found: %s", vault)
        return {"files_read": 0, "triples_ingested": 0, "error": "vault not found"}

    graph = ConceptGraph()
    val = ValidationEngine(graph)
    stats = {"files_read": 0, "chunks_processed": 0, "triples_extracted": 0, "triples_ingested": 0, "contradictions": []}

    md_files = [f for f in vault.rglob("*.md") if ".obsidian" not in str(f) and "Templates" not in str(f)]
    md_files = md_files[:VAULT_MAX_FILES]
    total = len(md_files)
    logger.info("Analyzing vault: %d markdown files", total)

    try:
        from tqdm import tqdm
        file_iter = tqdm(md_files, desc="vault scan", unit="file")
    except ImportError:
        file_iter = md_files

    for idx, md in enumerate(file_iter):
        if (idx + 1) % 20 == 0 or idx == 0:
            logger.info("Processing file %d/%d: %s", idx + 1, total, md.name)
        raw = md.read_text(encoding="utf-8", errors="replace")
        text = _parse_markdown_to_text(md)
        triples = _extract_triples_from_vault_format(md, raw)
        if not triples and text.strip():
            chunks = _chunk_text(text)
            stats["chunks_processed"] += len(chunks)
            if use_ollama and chunks:
                t = _extract_triples_ollama_batch(chunks)
                triples.extend(t)
            if not triples:
                from src.language_layer import extract_triples
                for chunk in chunks:
                    triples.extend(extract_triples(chunk, use_llm=False))
        if triples or text.strip():
            stats["files_read"] += 1
        triples = _filter_harmful(triples)
        stats["triples_extracted"] += len(triples)
        valid = val.validate_batch(triples)
        for t in valid:
            graph.ingest_triples([t], source=f"vault:{md.name}")
            stats["triples_ingested"] += 1

    stats["contradictions"] = [{"type": c.get("type"), "nodes": c.get("nodes")} for c in _detect_contradictions(graph)]
    try:
        from src.viz import export_coherence_metrics
        export_coherence_metrics(vault_path=vault)
    except Exception as e:
        logger.debug("Coherence metrics export failed: %s", e)
    if stats.get("triples_ingested", 0) > 0:
        try:
            from src.viz import auto_snapshot_graph
            auto_snapshot_graph(
                vault_path=vault,
                reason="after-analyze-vault",
                change_desc=f"vault analysis (+{stats['triples_ingested']} triples)",
            )
        except Exception as e:
            logger.debug("Snapshot after vault analysis failed: %s", e)
        try:
            from src.obsidian_filesystem_manager import auto_extract_and_merge_subideas
            sub = auto_extract_and_merge_subideas(vault_path=vault, use_ollama=use_ollama)
            stats["subideas"] = sub
        except Exception as e:
            logger.debug("Sub-idea extract/merge failed: %s", e)
    return stats
