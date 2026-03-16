#!/bin/bash
# Full-TS Self-Improvement Loop
# Run one cycle: git pull, Ollama analysis, TS coherence, export, tests
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"
python -m src.self_improver "$@"
