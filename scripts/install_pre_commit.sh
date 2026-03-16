#!/bin/bash
# Install TS pre-commit hook
cd "$(dirname "$0")/.."
git config core.hooksPath .githooks
chmod +x .githooks/pre-commit
echo "Pre-commit hook installed. Git will use .githooks/pre-commit"
