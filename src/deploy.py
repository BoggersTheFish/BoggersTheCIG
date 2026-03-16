"""
Subsystem 14: Deployment and Scaling
Hardware check, local Docker, cloud notes.
"""
import argparse
import logging
import platform
import sys

logger = logging.getLogger(__name__)


def check_hardware() -> dict:
    """Detect CPU/GPU, RAM; suggest free cloud if low."""
    info = {
        "platform": platform.system(),
        "python": sys.version.split()[0],
        "cpu_cores": None,
        "ram_gb": None,
        "gpu": None,
        "suggestion": "Local OK",
    }
    try:
        import os
        info["cpu_cores"] = os.cpu_count() or 0
    except Exception:
        pass
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / (1024 ** 3), 1)
        if info["ram_gb"] and info["ram_gb"] < 8:
            info["suggestion"] = "Consider free Colab/Kaggle for heavy LLM; 8GB+ recommended"
    except ImportError:
        info["suggestion"] = "Install psutil for RAM check: pip install psutil"
    try:
        import torch
        info["gpu"] = torch.cuda.is_available()
        if not info["gpu"]:
            info["suggestion"] = "No GPU detected. Use CPU (slower) or free Colab for GPU."
    except ImportError:
        info["gpu"] = False
        info["suggestion"] = "Install torch for GPU check"
    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Hardware check")
    args = parser.parse_args()
    if args.check:
        info = check_hardware()
        for k, v in info.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
