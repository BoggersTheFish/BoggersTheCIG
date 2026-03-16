"""
Automatic hardware detection and Ollama model selection.
Picks the largest model that fits safely (20% margin) on current CPU/GPU/RAM/VRAM.
Simulated examples:
  - Low-end laptop (8GB RAM, no GPU) → phi3.5-mini-instruct
  - Gaming PC (RTX 4090 24GB, 64GB RAM) → qwen2.5-coder:32b
  - Server (A100 80GB) → llama3.1:70b
"""
import logging
import os
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

# Ensure project root on path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Model fallback list: (model_name, min_vram_gb, min_ram_gb, description)
# effective = actual * (1 - SAFETY_MARGIN). e.g. 48GB VRAM → 38.4GB effective.
OLLAMA_MODEL_TIERS = [
    ("llama3.1:70b", 60, 64, "70B params, GPU 48GB+ VRAM"),
    ("qwen2.5-coder:32b", 19, 16, "32B params, GPU 24GB+ VRAM (~20GB needed)"),
    ("qwen2.5-coder:14b", 15, 32, "14B params, GPU 12GB+ or 32GB+ RAM"),
    ("qwen2.5-coder:7b", 0, 8, "7B params, CPU 8–16GB RAM"),
    ("phi3.5-mini-instruct", 0, 5, "3.8B params, 4GB RAM min"),
]

SAFETY_MARGIN = 0.20  # 20% headroom


def _get_ram_gb() -> tuple[float, float]:
    """Return (total_gb, available_gb). Uses psutil."""
    try:
        import psutil
        total = psutil.virtual_memory().total / (1024**3)
        avail = psutil.virtual_memory().available / (1024**3)
        return round(total, 1), round(avail, 1)
    except ImportError:
        return 0.0, 0.0
    except Exception as e:
        logger.debug("psutil RAM detection failed: %s", e)
        return 0.0, 0.0


def _get_gpu_info() -> tuple[bool, str, float]:
    """Return (has_nvidia, gpu_name, vram_gb). Uses torch.cuda."""
    try:
        import torch
        if not torch.cuda.is_available():
            return False, "", 0.0
        props = torch.cuda.get_device_properties(0)
        name = props.name if props else "Unknown"
        vram_bytes = props.total_memory if props else 0
        vram_gb = vram_bytes / (1024**3)
        return True, name, round(vram_gb, 1)
    except ImportError:
        return False, "", 0.0
    except Exception as e:
        logger.debug("torch GPU detection failed: %s", e)
        return False, "", 0.0


def _ollama_list_models() -> list[str]:
    """Return list of pulled model names via `ollama list`."""
    try:
        r = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=_PROJECT_ROOT,
        )
        if r.returncode != 0:
            return []
        models = []
        for line in r.stdout.strip().split("\n")[1:]:
            parts = line.split()
            if parts:
                models.append(parts[0])
        return models
    except Exception as e:
        logger.debug("ollama list failed: %s", e)
        return []


def _model_matches_pulled(candidate: str, pulled: list[str]) -> bool:
    """True if candidate is in pulled list (exact or variant match, e.g. 7b-instruct)."""
    for p in pulled:
        if p == candidate or p.startswith(candidate + "-") or p.startswith(candidate + ":"):
            return True
    return False


def select_ollama_model(
    force_model: str | None = None,
    safety_margin: float = SAFETY_MARGIN,
) -> str:
    """
    Select best Ollama model for current hardware.
    Applies safety margin. Falls back to next tier if selected model not pulled.
    Returns model name. Sets os.environ["OLLAMA_MODEL"].
    """
    if force_model:
        os.environ["OLLAMA_MODEL"] = force_model
        logger.info("Force model: %s (override)", force_model)
        return force_model

    ram_total, ram_avail = _get_ram_gb()
    has_gpu, gpu_name, vram_gb = _get_gpu_info()
    pulled = _ollama_list_models()

    effective_vram = vram_gb * (1 - safety_margin) if has_gpu else 0
    effective_ram = ram_avail * (1 - safety_margin) if ram_avail > 0 else ram_total * (1 - safety_margin)

    selected = None
    reason = ""

    for model_name, min_vram, min_ram, desc in OLLAMA_MODEL_TIERS:
        if min_vram > 0 and not has_gpu:
            continue
        if min_vram > 0 and effective_vram < min_vram:
            continue
        if effective_ram < min_ram:
            continue
        if _model_matches_pulled(model_name, pulled):
            selected = model_name
            if min_vram > 0:
                reason = f"VRAM {vram_gb}GB (eff {effective_vram:.0f}GB) >= {min_vram}GB"
            else:
                reason = f"RAM {ram_avail}GB (eff {effective_ram:.0f}GB) >= {min_ram}GB"
            break
        logger.warning("Model %s fits hardware but not pulled; trying next tier", model_name)

    if not selected:
        for model_name, min_vram, min_ram, _ in OLLAMA_MODEL_TIERS:
            if min_vram > 0 and not has_gpu:
                continue
            if min_vram > 0 and effective_vram < min_vram:
                continue
            if effective_ram < min_ram:
                continue
            selected = model_name
            reason = "fallback (not pulled, run: ollama pull %s)" % model_name
            logger.warning("No pulled model fits; selecting %s. Run: ollama pull %s", selected, selected)
            break

    if not selected:
        selected = OLLAMA_MODEL_TIERS[-1][0]
        reason = "minimal fallback"
        logger.warning("Hardware below minimum; using %s", selected)

    hw_desc = []
    if has_gpu:
        hw_desc.append(f"{gpu_name}, {vram_gb}GB VRAM")
    hw_desc.append(f"{ram_total}GB RAM")
    logger.info(
        "Hardware detected: %s → selecting %s (%s)",
        ", ".join(hw_desc),
        selected,
        reason,
    )
    os.environ["OLLAMA_MODEL"] = selected
    return selected


def detect_and_set_model(force_model: str | None = None) -> str:
    """
    Run hardware detection and set OLLAMA_MODEL. Call before importing config.
    Returns selected model name.
    """
    return select_ollama_model(force_model=force_model)
