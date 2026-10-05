#!/usr/bin/env bash
# ci.sh - Deterministic CI script for maurice
# Triggered by jules-mcp-server webhook endpoint

set -e

echo "==> Running Jules CI Gates"

# 1. Formatting & Linting
echo "--> Running ruff check and format..."
poetry run ruff check --fix . || ruff check --fix .
poetry run ruff format . || ruff format .

# 2. Testing
echo "--> Running pytest..."
poetry run pytest tests/ -v -m "not slow" || pytest tests/ -v -m "not slow"

# 3. Compile checks
echo "--> Running py_compile..."
poetry run python3 -m py_compile scripts/*.py || python3 -m py_compile scripts/*.py

# 4. Glibc vs Musl static validation
echo "--> Validating Containerfile glibc/musl rule..."
if [ -f "Containerfile" ]; then
    if grep -iq "alpine" Containerfile || grep -iq "musl" Containerfile; then
        echo "[ERROR] Alpine/musl detected in Containerfile. Ensure compatibility with glibc runtime."
        exit 1
    fi
fi

echo "==> All CI Gates passed successfully!"
