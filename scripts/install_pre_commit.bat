@echo off
REM Install TS pre-commit hook
cd /d "%~dp0\.."
git config core.hooksPath .githooks
echo Pre-commit hook installed. Git will use .githooks/pre-commit
