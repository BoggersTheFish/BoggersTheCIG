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
# Use only 80% of detected VRAM/RAM (SAFETY_MARGIN = 0.20).
OLLAMA_MODEL_TIERS = [
    ("llama3.1:70b", 60, 64, "70B params, GPU 48GB+ VRAM"),
    ("qwen2.5:32b", 19, 16, "32B params, GPU 24GB+ VRAM"),
    ("mixtral:22b", 19, 16, "22B MoE, GPU 24GB+ VRAM"),
    ("qwen2.5-coder:14b", 15, 26, "14B params, GPU 12GB+ or CPU 32GB+ RAM"),
    ("qwen2.5-coder:7b", 0, 13, "7B params, CPU 16–32GB RAM"),
    ("phi3.5-mini-instruct", 0, 5, "3.8B params, CPU <16GB RAM"),
    ("tinyllama", 0, 3, "1.1B params, minimal RAM"),
]

SAFETY_MARGIN = 0.20  # Use 80% of available (20% headroom)


def _get_cpu_cores() -> int:
    """Return CPU core count."""
    try:
        import psutil
        return psutil.cpu_count(logical=True) or 1
    except Exception:
        return 1


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
        msg = f"Force model: {force_model} (override)"
        logger.info(msg)
        print(msg)
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

    cpu_cores = _get_cpu_cores()
    hw_desc = [f"CPU {cpu_cores} cores"]
    if has_gpu:
        hw_desc.append(f"{gpu_name}, {vram_gb}GB VRAM")
    hw_desc.append(f"{ram_total}GB RAM ({ram_avail}GB free)")
    msg = f"Hardware detected: {', '.join(hw_desc)} → selecting {selected} ({reason})"
    logger.info(msg)
    print(msg)
    os.environ["OLLAMA_MODEL"] = selected
    return selected


def detect_and_set_model(force_model: str | None = None) -> str:
    """
    Run hardware detection and set OLLAMA_MODEL. Call before importing config.
    Returns selected model name.
    """
    return select_ollama_model(force_model=force_model)


def get_resource_usage() -> dict:
    """Return current RAM and VRAM usage (percent used). For monitoring at scale."""
    out = {"ram_pct": 0.0, "vram_pct": 0.0}
    try:
        import psutil
        vm = psutil.virtual_memory()
        out["ram_pct"] = vm.percent
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            out["vram_used_gb"] = torch.cuda.memory_allocated(0) / (1024**3)
            out["vram_total_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            out["vram_pct"] = 100 * out["vram_used_gb"] / out["vram_total_gb"] if out["vram_total_gb"] else 0
    except Exception:
        pass
    return out


def check_resource_pressure(threshold: float = 90.0) -> bool:
    """True if RAM or VRAM usage > threshold (should downscale)."""
    u = get_resource_usage()
    return u.get("ram_pct", 0) > threshold or u.get("vram_pct", 0) > threshold


def select_downscaled_model(threshold: float = 90.0) -> str | None:
    """
    If resource pressure > threshold, pick next smaller model tier.
    Returns new model name or None if already at smallest or no pressure.
    Sets os.environ["OLLAMA_MODEL"] when downscaling.
    Tiers: 0=largest, -1=smallest.
    """
    if not check_resource_pressure(threshold):
        return None
    current = os.environ.get("OLLAMA_MODEL", "")
    idx = None
    for i in range(len(OLLAMA_MODEL_TIERS) - 1, -1, -1):
        name = OLLAMA_MODEL_TIERS[i][0]
        if name == current or current.startswith(name.split(":")[0] + ":") or current.startswith(name + "-"):
            idx = i
            break
    if idx is None or idx >= len(OLLAMA_MODEL_TIERS) - 1:
        return None
    pulled = _ollama_list_models()
    for j in range(idx + 1, len(OLLAMA_MODEL_TIERS)):
        candidate = OLLAMA_MODEL_TIERS[j][0]
        if _model_matches_pulled(candidate, pulled):
            os.environ["OLLAMA_MODEL"] = candidate
            logger.warning("Resource pressure > %.0f%% → downscaling to %s", threshold, candidate)
            return candidate
    return None
