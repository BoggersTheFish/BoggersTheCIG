@echo off
REM Full-TS Self-Improvement Loop
REM Run one cycle: git pull, Ollama analysis, TS coherence, export, tests
setlocal
cd /d "%~dp0\.."
set PYTHONPATH=%CD%
python -m src.self_improver %*
