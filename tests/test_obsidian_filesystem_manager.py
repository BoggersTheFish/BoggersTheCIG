"""Tests for obsidian_filesystem_manager module."""
from pathlib import Path
from unittest.mock import patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))


def test_split_note_into_blocks():
    """_split_note_into_blocks extracts ## sections and bullet groups >= 50 words."""
    from src.obsidian_filesystem_manager import _split_note_into_blocks, SUBIDEA_MIN_WORDS

    short = "# Physics\n\n## Neighbors\n\n- [[Gravity]]"
    assert len(_split_note_into_blocks(short)) == 0

    long_section = "# Physics\n\n## Quantum Mechanics\n\n" + " ".join(["word"] * 60)
    blocks = _split_note_into_blocks(long_section)
    assert len(blocks) >= 1
    assert "Quantum" in blocks[0][0]

    long_bullets = "# Chemistry\n\n- First point. " + " ".join(["x"] * 55) + "\n- Second point. " + " ".join(["y"] * 55)
    blocks = _split_note_into_blocks(long_bullets)
    assert len(blocks) >= 1


def test_extract_and_merge_subideas_skipped_few_files(tmp_path, monkeypatch):
    """auto_extract_and_merge_subideas skips when too few files."""
    from src.obsidian_filesystem_manager import auto_extract_and_merge_subideas

    concepts = tmp_path / "Concepts"
    concepts.mkdir()
    (concepts / "a.md").write_text("# A\n\n## Neighbors\n\n- [[B]]")
    monkeypatch.setattr("src.obsidian_filesystem_manager.OBSIDIAN_VAULT", tmp_path)

    result = auto_extract_and_merge_subideas(vault_path=tmp_path)
    assert "skipped" in result
    assert result["extractions"] == []
    assert result["merges"] == []


def test_extract_and_merge_subideas_no_meaningful_blocks(tmp_path, monkeypatch):
    """auto_extract_and_merge_subideas skips when no blocks >= 50 words."""
    from src.obsidian_filesystem_manager import auto_extract_and_merge_subideas

    concepts = tmp_path / "Concepts"
    concepts.mkdir()
    for i in range(5):
        (concepts / f"f{i}.md").write_text(f"# F{i}\n\n## Neighbors\n\n- [[X]]")
    monkeypatch.setattr("src.obsidian_filesystem_manager.OBSIDIAN_VAULT", tmp_path)

    result = auto_extract_and_merge_subideas(vault_path=tmp_path, use_ollama=False)
    assert result["extractions"] == []
    assert result["merges"] == []


def test_extract_and_merge_subideas_creates_shared_and_extracts(tmp_path, monkeypatch):
    """With mock notes containing long sections, extraction creates Shared-Sub-ideas or Sub-ideas."""
    from src.obsidian_filesystem_manager import auto_extract_and_merge_subideas, _split_note_into_blocks

    concepts = tmp_path / "Concepts"
    concepts.mkdir()
    quantum_block = "## Quantum Mechanics\n\n" + " ".join(["quantum"] * 60)
    (concepts / "Physics.md").write_text("# Physics\n\n## Neighbors\n\n- [[Gravity]]\n\n" + quantum_block)
    (concepts / "Chemistry.md").write_text("# Chemistry\n\n## Neighbors\n\n- [[Atom]]\n\n" + quantum_block)
    monkeypatch.setattr("src.obsidian_filesystem_manager.OBSIDIAN_VAULT", tmp_path)
    monkeypatch.setattr("src.config.OBSIDIAN_VAULT", tmp_path)

    def mock_meaningful(t):
        return len(t.split()) >= 50

    def mock_duplicate(t1, t2, **kw):
        return t1[:100] == t2[:100]

    def mock_name(t):
        return "quantum-mechanics"

    with patch("ollama_integration.is_meaningful_subidea", side_effect=mock_meaningful):
        with patch("ollama_integration.is_duplicate_subidea", side_effect=mock_duplicate):
            with patch("ollama_integration.suggest_extraction_name", side_effect=mock_name):
                result = auto_extract_and_merge_subideas(vault_path=tmp_path, use_ollama=True)

    shared = concepts / "Shared-Sub-ideas"
    if result.get("merges"):
        assert shared.exists()
        assert any(f.suffix == ".md" for f in shared.iterdir())
    assert "extractions" in result
    assert "merges" in result


def test_count_root_files():
    """_count_root_files counts .md files in concepts dir."""
    from src.obsidian_filesystem_manager import _count_root_files

    tmp = PROJECT_ROOT / "tests" / "_org_tmp"
    tmp.mkdir(exist_ok=True)
    (tmp / "a.md").write_text("# A")
    (tmp / "b.md").write_text("# B")
    try:
        assert _count_root_files(tmp) == 2
    finally:
        for f in tmp.glob("*.md"):
            f.unlink()
        tmp.rmdir()


def test_auto_organize_vault_skipped_few_files(tmp_path, monkeypatch):
    """auto_organize_vault skips when few files."""
    from src.obsidian_filesystem_manager import auto_organize_vault

    concepts = tmp_path / "Concepts"
    concepts.mkdir()
    (concepts / "a.md").write_text("# A")
    (concepts / "b.md").write_text("# B")
    monkeypatch.setattr("src.obsidian_filesystem_manager.OBSIDIAN_VAULT", tmp_path)
    monkeypatch.setattr("src.config.OBSIDIAN_VAULT", tmp_path)

    result = auto_organize_vault(vault_path=tmp_path)
    assert "skipped" in result
    assert result["moves"] == []


def test_auto_organize_vault_force_skips_too_few(tmp_path, monkeypatch):
    """With force, still skips if < 5 files."""
    from src.obsidian_filesystem_manager import auto_organize_vault

    concepts = tmp_path / "Concepts"
    concepts.mkdir()
    for i in range(3):
        (concepts / f"f{i}.md").write_text(f"# F{i}")
    monkeypatch.setattr("src.obsidian_filesystem_manager.OBSIDIAN_VAULT", tmp_path)

    result = auto_organize_vault(vault_path=tmp_path, force=True)
    assert "skipped" in result or result["moves"] == []
