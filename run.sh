#!/bin/bash
# Full-TS Cognitive Architecture - Run script
export PYTHONPATH="$(cd "$(dirname "$0")" && pwd)"
python src/main.py "$@"
