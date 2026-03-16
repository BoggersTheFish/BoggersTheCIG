"""Tests for hardware_adapt module."""
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_select_force_model():
    """--force-model overrides auto-detection."""
    from src.hardware_adapt import select_ollama_model

    with patch.dict(os.environ, {}, clear=False):
        result = select_ollama_model(force_model="llama3.1:70b")
        assert result == "llama3.1:70b"
        assert os.environ.get("OLLAMA_MODEL") == "llama3.1:70b"


def test_select_low_end_laptop():
    """8GB RAM, no GPU → phi3.5-mini-instruct (6.5GB free → 5.2 effective >= 5)."""
    from src.hardware_adapt import select_ollama_model

    with patch("src.hardware_adapt._get_ram_gb", return_value=(8.0, 6.5)):
        with patch("src.hardware_adapt._get_gpu_info", return_value=(False, "", 0.0)):
            with patch("src.hardware_adapt._ollama_list_models", return_value=["phi3.5-mini-instruct"]):
                with patch.dict(os.environ, {}, clear=False):
                    result = select_ollama_model()
    assert result == "phi3.5-mini-instruct"


def test_select_gaming_pc():
    """RTX 4090 24GB, 64GB RAM → qwen2.5:32b or qwen2.5-coder:32b (if pulled)."""
    from src.hardware_adapt import select_ollama_model

    with patch("src.hardware_adapt._get_ram_gb", return_value=(64.0, 50.0)):
        with patch("src.hardware_adapt._get_gpu_info", return_value=(True, "NVIDIA GeForce RTX 4090", 24.0)):
            with patch("src.hardware_adapt._ollama_list_models", return_value=["qwen2.5:32b", "qwen2.5-coder:7b"]):
                with patch.dict(os.environ, {}, clear=False):
                    result = select_ollama_model()
    assert "32b" in result


def test_select_server_a100():
    """A100 80GB → llama3.1:70b (if pulled)."""
    from src.hardware_adapt import select_ollama_model

    with patch("src.hardware_adapt._get_ram_gb", return_value=(256.0, 200.0)):
        with patch("src.hardware_adapt._get_gpu_info", return_value=(True, "NVIDIA A100-SXM4-80GB", 80.0)):
            with patch("src.hardware_adapt._ollama_list_models", return_value=["llama3.1:70b", "qwen2.5-coder:32b"]):
                with patch.dict(os.environ, {}, clear=False):
                    result = select_ollama_model()
    assert result == "llama3.1:70b"


def test_fallback_when_model_not_pulled():
    """If 32b fits but not pulled, fallback to 7b if pulled."""
    from src.hardware_adapt import select_ollama_model

    with patch("src.hardware_adapt._get_ram_gb", return_value=(64.0, 50.0)):
        with patch("src.hardware_adapt._get_gpu_info", return_value=(True, "NVIDIA RTX 4090", 24.0)):
            with patch("src.hardware_adapt._ollama_list_models", return_value=["qwen2.5-coder:7b"]):
                with patch.dict(os.environ, {}, clear=False):
                    result = select_ollama_model()
    assert result == "qwen2.5-coder:7b"


def test_detect_and_set_model():
    """detect_and_set_model returns selected model."""
    from src.hardware_adapt import detect_and_set_model

    with patch("src.hardware_adapt._get_ram_gb", return_value=(24.0, 18.0)):
        with patch("src.hardware_adapt._get_gpu_info", return_value=(False, "", 0.0)):
            with patch("src.hardware_adapt._ollama_list_models", return_value=["qwen2.5-coder:7b"]):
                with patch.dict(os.environ, {}, clear=False):
                    result = detect_and_set_model()
                    assert result == "qwen2.5-coder:7b"
                    assert os.environ.get("OLLAMA_MODEL") == "qwen2.5-coder:7b"


def test_get_ram_gb():
    """_get_ram_gb returns (total, available) via psutil."""
    from src.hardware_adapt import _get_ram_gb

    mock_vm = MagicMock()
    mock_vm.total = 16 * 1024**3
    mock_vm.available = 10 * 1024**3
    try:
        import psutil
        with patch("psutil.virtual_memory", return_value=mock_vm):
            total, avail = _get_ram_gb()
        assert total == 16.0
        assert avail == 10.0
    except ImportError:
        total, avail = _get_ram_gb()
        assert total >= 0 and avail >= 0


def test_model_matches_pulled():
    """_model_matches_pulled handles exact and variant match."""
    from src.hardware_adapt import _model_matches_pulled

    assert _model_matches_pulled("qwen2.5-coder:7b", ["qwen2.5-coder:7b"]) is True
    assert _model_matches_pulled("qwen2.5-coder:7b", ["qwen2.5-coder:7b-instruct"]) is True
    assert _model_matches_pulled("phi3.5-mini-instruct", ["phi3.5-mini-instruct"]) is True
    assert _model_matches_pulled("llama3.1:70b", ["llama3.1:8b"]) is False
    assert _model_matches_pulled("llama3.1:70b", ["llama3.1:70b-instruct"]) is True


def test_get_resource_usage():
    """get_resource_usage returns dict with ram_pct, vram_pct."""
    from src.hardware_adapt import get_resource_usage

    u = get_resource_usage()
    assert "ram_pct" in u
    assert u["ram_pct"] >= 0


def test_check_resource_pressure():
    """check_resource_pressure returns bool."""
    from src.hardware_adapt import check_resource_pressure

    with patch("src.hardware_adapt.get_resource_usage", return_value={"ram_pct": 95.0, "vram_pct": 0}):
        assert check_resource_pressure(90.0) is True
    with patch("src.hardware_adapt.get_resource_usage", return_value={"ram_pct": 50.0, "vram_pct": 50.0}):
        assert check_resource_pressure(90.0) is False


def test_select_downscaled_model():
    """select_downscaled_model downscales when under pressure."""
    from src.hardware_adapt import select_downscaled_model

    with patch("src.hardware_adapt.check_resource_pressure", return_value=True):
        with patch("src.hardware_adapt._ollama_list_models", return_value=["qwen2.5-coder:7b", "phi3.5-mini-instruct"]):
            with patch.dict(os.environ, {"OLLAMA_MODEL": "qwen2.5-coder:7b"}, clear=False):
                result = select_downscaled_model(90.0)
                assert result == "phi3.5-mini-instruct"
                assert os.environ.get("OLLAMA_MODEL") == "phi3.5-mini-instruct"

    with patch("src.hardware_adapt.check_resource_pressure", return_value=False):
        with patch.dict(os.environ, {"OLLAMA_MODEL": "qwen2.5-coder:7b"}, clear=False):
            result = select_downscaled_model(90.0)
            assert result is None

    with patch("src.hardware_adapt.check_resource_pressure", return_value=True):
        with patch.dict(os.environ, {"OLLAMA_MODEL": "tinyllama"}, clear=False):
            result = select_downscaled_model(90.0)
            assert result is None
