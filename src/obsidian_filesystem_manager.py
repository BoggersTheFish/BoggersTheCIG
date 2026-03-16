"""
Obsidian vault filesystem organization.
Clusters Concepts by embeddings + Ollama structural role, proposes folders, moves files.
Extracts sub-ideas into hierarchical structure, merges duplicates into Shared-Sub-ideas.
Logs to Evolution-Log.md. $0 cost (local sentence-transformers + Ollama).

Simulated example:
  Before: Physics.md with long quantum section, Chemistry.md with similar section
  After: Physics.md and Chemistry.md both link to [[Shared-Sub-ideas/Quantum-Mechanics]];
         Shared-Sub-ideas/Quantum-Mechanics.md has "Used in: [[Physics]], [[Chemistry]]"
"""
import json
import logging
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

from src.config import OBSIDIAN_VAULT, ORGANIZE_VAULT_ROOT_FILES_THRESHOLD, ORGANIZE_VAULT_COHERENCE_THRESHOLD

logger = logging.getLogger(__name__)

# Sub-idea extraction: min words to consider, similarity threshold for merge
SUBIDEA_MIN_WORDS = 50
SUBIDEA_MERGE_SIMILARITY = 0.85


def _count_root_files(concepts_dir: Path) -> int:
    """Count .md files directly in concepts_dir (not in subdirs)."""
    if not concepts_dir.exists():
        return 0
    return sum(1 for f in concepts_dir.iterdir() if f.suffix == ".md" and f.is_file())


def _get_coherence_for_organize() -> float:
    """Return current graph density (0-1). Used to decide if organize needed."""
    try:
        from src.eval import graph_coherence
        from src.concept_graph import ConceptGraph
        coh = graph_coherence(ConceptGraph())
        return coh.get("density", 0) or 0
    except Exception:
        return 0


def _classify_note_ollama(content: str, path: Path) -> str:
    """Use Ollama to assign folder category. Returns folder name (e.g. Physics, Hypotheses)."""
    try:
        from ollama_integration import check_ollama_available, _ollama_request
        if not check_ollama_available():
            return "Other"
        prompt = f"""Note title: {path.stem}
Content (first 500 chars): {content[:500]}

Assign ONE folder from: Physics, Hypotheses, External, Mathematics, Biology, Philosophy, Other.
Reply with only the folder name, nothing else."""
        resp = _ollama_request(prompt, timeout=15).strip()
        for cat in ["Physics", "Hypotheses", "External", "Mathematics", "Biology", "Philosophy", "Other"]:
            if cat.lower() in resp.lower():
                return cat
        return "Other"
    except Exception as e:
        logger.debug("Ollama classify failed: %s", e)
        return "Other"


def _embed_text(text: str) -> List[float]:
    """Get embedding for text. Uses concept_graph's embed or fallback."""
    try:
        from src.concept_graph import _embed
        return _embed(text)
    except Exception:
        return [0.0] * 384


def _cosine_sim(a: List[float], b: List[float]) -> float:
    """Cosine similarity."""
    try:
        import numpy as np
        x, y = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
        return float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-9))
    except Exception:
        return 0


def _fix_wikilinks_in_file(path: Path, moves: dict):
    """Update wikilinks in file if linked files were moved. Obsidian resolves by name, so optional."""
    if not path.exists() or path.suffix != ".md":
        return
    content = path.read_text(encoding="utf-8", errors="replace")
    changed = False
    for old_path, new_path in moves.items():
        stem = old_path.stem
        if f"[[{stem}]]" in content or f"[[{stem}|" in content:
            pass
        if f"]({old_path.as_posix()})" in content:
            content = content.replace(f"]({old_path.as_posix()})", f"]({new_path.as_posix()})")
            changed = True
    if changed:
        path.write_text(content, encoding="utf-8")


def _append_evolution_log(vault: Path, entries: List[str]):
    """Append organization entries to Evolution-Log.md."""
    log_path = vault / "Evolution-Log.md"
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    block = f"\n## Auto-organize {ts}\n\n" + "\n".join(entries) + "\n"
    if log_path.exists():
        content = log_path.read_text(encoding="utf-8")
        content += block
    else:
        content = "# Evolution Log\n\n" + block
    log_path.write_text(content, encoding="utf-8")


def _append_subidea_log(vault: Path, entries: List[str]):
    """Append sub-idea extraction/merge entries to Evolution-Log.md."""
    log_path = vault / "Evolution-Log.md"
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    block = f"\n## Extract & Merge Sub-ideas {ts}\n\n" + "\n".join(entries) + "\n"
    if log_path.exists():
        content = log_path.read_text(encoding="utf-8")
        content += block
    else:
        content = "# Evolution Log\n\n" + block
    log_path.write_text(content, encoding="utf-8")


def _split_note_into_blocks(content: str) -> List[Tuple[str, int, int]]:
    """
    Split note content into blocks (## sections or contiguous bullet groups).
    Returns [(block_text, start_pos, end_pos), ...].
    """
    blocks = []
    lines = content.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        if re.match(r"^#{2,3}\s+", line):
            section_lines = [line]
            start = i
            i += 1
            while i < len(lines) and not re.match(r"^#{1,3}\s+", lines[i]):
                section_lines.append(lines[i])
                i += 1
            block = "\n".join(section_lines).strip()
            if len(block.split()) >= SUBIDEA_MIN_WORDS:
                blocks.append((block, start, i))
            continue
        if re.match(r"^\s*[-*]\s+", line):
            bullet_lines = []
            start = i
            while i < len(lines):
                l = lines[i]
                if re.match(r"^\s*[-*]\s+", l) or (bullet_lines and (l.startswith("  ") or l.startswith("\t") or not l.strip())):
                    bullet_lines.append(l)
                    i += 1
                elif not l.strip():
                    bullet_lines.append(l)
                    i += 1
                else:
                    break
            block = "\n".join(bullet_lines).strip()
            if len(block.split()) >= SUBIDEA_MIN_WORDS:
                blocks.append((block, start, i))
            continue
        i += 1
    return blocks


def _replace_block_in_content(content: str, block: str, replacement: str) -> str:
    """Replace a block in content with replacement text. Handles exact match."""
    if block not in content:
        return content
    return content.replace(block, replacement, 1)


def auto_extract_and_merge_subideas(
    vault_path: Path = None,
    use_ollama: bool = True,
    force: bool = False,
) -> dict:
    """
    Extract meaningful sub-ideas from concept notes into hierarchical files.
    Merge duplicate sub-ideas across notes into Shared-Sub-ideas/.
    Add backlink section "Used in: [[A]], [[B]]" to shared files.
    Replace original content with wikilinks. Log to Evolution-Log.md.
    Returns {extractions: [...], merges: [...], skipped: reason}.
    """
    vault = vault_path or OBSIDIAN_VAULT
    concepts_dir = vault / "Concepts"
    if not concepts_dir.exists():
        return {"extractions": [], "merges": [], "skipped": "no Concepts dir"}

    md_files = [f for f in concepts_dir.rglob("*.md") if f.is_file() and ".obsidian" not in str(f)]
    if len(md_files) < 2:
        return {"extractions": [], "merges": [], "skipped": "too few files"}

    candidates: List[Tuple[Path, str, str, int, int]] = []
    for f in md_files:
        try:
            content = f.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.debug("Skip %s: %s", f, e)
            continue
        parent_name = f.stem
        blocks = _split_note_into_blocks(content)
        for block, start, end in blocks:
            if len(block.split()) < SUBIDEA_MIN_WORDS:
                continue
            if use_ollama:
                try:
                    from ollama_integration import is_meaningful_subidea
                    if not is_meaningful_subidea(block):
                        continue
                except ImportError:
                    pass
            candidates.append((f, parent_name, block, start, end))

    if not candidates:
        return {"extractions": [], "merges": [], "skipped": "no meaningful sub-ideas"}

    shared_dir = concepts_dir / "Shared-Sub-ideas"
    shared_dir.mkdir(parents=True, exist_ok=True)

    merge_groups: List[List[Tuple[Path, str, str, int, int]]] = []
    used = [False] * len(candidates)
    for i in range(len(candidates)):
        if used[i]:
            continue
        group = [candidates[i]]
        used[i] = True
        for j in range(i + 1, len(candidates)):
            if used[j]:
                continue
            dup = False
            try:
                from ollama_integration import is_duplicate_subidea
                dup = is_duplicate_subidea(
                    candidates[i][2], candidates[j][2],
                    use_embeddings=True, threshold=SUBIDEA_MERGE_SIMILARITY,
                )
            except ImportError:
                pass
            if dup:
                group.append(candidates[j])
                used[j] = True
        merge_groups.append(group)

    log_entries = []
    extractions = []
    merges = []
    content_cache: dict[Path, str] = {}

    for group in merge_groups:
        parent_path, parent_name, block, _, _ = group[0]
        if parent_path not in content_cache:
            content_cache[parent_path] = parent_path.read_text(encoding="utf-8", errors="replace")

        if len(group) > 1:
            parent_names = sorted(set(g[1] for g in group))
            try:
                from ollama_integration import suggest_extraction_name
                name = suggest_extraction_name(block)
            except ImportError:
                name = "sub-idea"
            name = re.sub(r"[^\w\-]", "", name.replace(" ", "-").lower())[:50] or "sub-idea"
            shared_file = shared_dir / f"{name}.md"
            if shared_file.exists():
                existing = shared_file.read_text(encoding="utf-8", errors="replace")
                used_in = []
                m = re.search(r"Used in:\s*(.+)", existing, re.DOTALL)
                if m:
                    used_in = re.findall(r"\[\[([^\]]+)\]\]", m.group(1))
                for pn in parent_names:
                    if pn not in used_in:
                        used_in.append(pn)
                backlink = "Used in: " + ", ".join(f"[[{u}]]" for u in sorted(used_in))
                if "Used in:" in existing:
                    existing = re.sub(r"Used in:.*", backlink, existing, count=1, flags=re.DOTALL)
                else:
                    existing = backlink + "\n\n" + existing
                shared_file.write_text(existing, encoding="utf-8")
            else:
                backlink = "Used in: " + ", ".join(f"[[{u}]]" for u in sorted(parent_names))
                body = f"# {name.replace('-', ' ').title()}\n\n{backlink}\n\n---\n\n{block}"
                shared_file.write_text(body, encoding="utf-8")

            wikilink = f"[[Shared-Sub-ideas/{name}]]"
            for g in group:
                fp, pn, blk, _, _ = g
                if fp not in content_cache:
                    content_cache[fp] = fp.read_text(encoding="utf-8", errors="replace")
                content_cache[fp] = _replace_block_in_content(content_cache[fp], blk, wikilink)
            merges.append({"shared": str(shared_file), "parents": parent_names})
            log_entries.append(f"- MERGE: {parent_names} -> Shared-Sub-ideas/{name}.md")
        else:
            parent_folder = concepts_dir / parent_name
            subideas_dir = parent_folder / "Sub-ideas"
            subideas_dir.mkdir(parents=True, exist_ok=True)
            try:
                from ollama_integration import suggest_extraction_name
                name = suggest_extraction_name(block)
            except ImportError:
                name = "sub-idea"
            name = re.sub(r"[^\w\-]", "", name.replace(" ", "-").lower())[:50] or "sub-idea"
            dest = subideas_dir / f"{name}.md"
            if dest.exists():
                dest_content = dest.read_text(encoding="utf-8", errors="replace")
                dest_content += "\n\n---\n\n" + block
                dest.write_text(dest_content, encoding="utf-8")
            else:
                backlink = f"Used in: [[{parent_name}]]"
                body = f"# {name.replace('-', ' ').title()}\n\n{backlink}\n\n---\n\n{block}"
                dest.write_text(body, encoding="utf-8")

            try:
                rel_dest = dest.relative_to(concepts_dir)
                wikilink = f"[[{rel_dest.with_suffix('').as_posix()}]]"
            except ValueError:
                wikilink = f"[[Sub-ideas/{name}]]"
            content_cache[parent_path] = _replace_block_in_content(content_cache[parent_path], block, wikilink)
            extractions.append({"from": parent_name, "to": str(dest)})
            log_entries.append(f"- EXTRACT: {parent_name} -> {dest.relative_to(concepts_dir)}")

    for path, content in content_cache.items():
        path.write_text(content, encoding="utf-8")

    if log_entries:
        _append_subidea_log(vault, log_entries)

    try:
        from src.viz import export_coherence_metrics, auto_snapshot_graph
        export_coherence_metrics(vault_path=vault)
        auto_snapshot_graph(vault_path=vault, reason="extract-subideas", change_desc=f"{len(extractions)} extracts, {len(merges)} merges")
    except Exception as e:
        logger.debug("Post-extract metrics/snapshot failed: %s", e)

    return {
        "extractions": extractions,
        "merges": merges,
        "skipped": None if (extractions or merges) else "no changes",
    }


def auto_organize_vault(
    vault_path: Path = None,
    root_files_threshold: int = None,
    coherence_threshold: float = None,
    force: bool = False,
) -> dict:
    """
    Organize vault Concepts by clustering (embeddings + Ollama category).
    Creates folders (Physics/, Hypotheses/, etc.), moves files, logs to Evolution-Log.md.
    Runs only if root_files > threshold or coherence < threshold, unless force=True.
    Returns {moves: [(from, to)], folders_created: [...], skipped: reason}.
    """
    vault = vault_path or OBSIDIAN_VAULT
    concepts_dir = vault / "Concepts"
    if not concepts_dir.exists():
        return {"moves": [], "folders_created": [], "skipped": "no Concepts dir"}

    root_count = _count_root_files(concepts_dir)
    coherence = _get_coherence_for_organize()
    thresh_files = root_files_threshold or ORGANIZE_VAULT_ROOT_FILES_THRESHOLD
    thresh_coh = coherence_threshold if coherence_threshold is not None else ORGANIZE_VAULT_COHERENCE_THRESHOLD
    if not force and root_count <= thresh_files and coherence >= thresh_coh:
        return {"moves": [], "folders_created": [], "skipped": f"root={root_count}, coh={coherence:.3f}"}

    md_files = list(concepts_dir.rglob("*.md"))
    md_files = [f for f in md_files if f.is_file() and ".obsidian" not in str(f)]
    if len(md_files) < 5:
        return {"moves": [], "folders_created": [], "skipped": "too few files"}

    category_to_files: dict[str, List[Path]] = defaultdict(list)
    for f in md_files:
        rel = f.relative_to(concepts_dir)
        if len(rel.parts) > 1:
            continue
        try:
            content = f.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.debug("Skip %s: %s", f, e)
            continue
        cat = _classify_note_ollama(content, f)
        category_to_files[cat].append(f)

    moves = []
    folders_created = []
    for cat, files in category_to_files.items():
        if cat == "Other" or len(files) < 2:
            continue
        folder = concepts_dir / cat
        folder.mkdir(parents=True, exist_ok=True)
        if folder not in folders_created:
            folders_created.append(folder)
        for f in files:
            dest = folder / f.name
            if dest == f:
                continue
            if dest.exists():
                logger.debug("Skip move %s already exists", dest)
                continue
            try:
                f.rename(dest)
                moves.append((str(f), str(dest)))
            except Exception as e:
                logger.warning("Move failed %s -> %s: %s", f, dest, e)

    if moves:
        moves_dict = {Path(a): Path(b) for a, b in moves}
        for f in concepts_dir.rglob("*.md"):
            _fix_wikilinks_in_file(f, moves_dict)
        entries = [f"- {a} -> {b}" for a, b in moves]
        _append_evolution_log(vault, entries)

    try:
        from src.viz import export_coherence_metrics, auto_snapshot_graph
        export_coherence_metrics(vault_path=vault)
        auto_snapshot_graph(vault_path=vault, reason="after-organize", change_desc=f"vault organize ({len(moves)} moves)")
    except Exception as e:
        logger.debug("Post-organize metrics/snapshot failed: %s", e)

    return {
        "moves": moves,
        "folders_created": [str(p) for p in folders_created],
        "skipped": None if moves else "no moves needed",
    }
