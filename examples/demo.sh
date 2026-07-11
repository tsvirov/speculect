#!/usr/bin/env bash
# Offline demo — runs entirely against fixture GGUF files in examples/fixtures/,
# no Ollama or llama.cpp install required.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== speculect scan --gguf-dir examples/fixtures ==="
speculect scan --gguf-dir examples/fixtures

echo
echo "=== speculect pair --target target-7b.gguf --gguf-dir examples/fixtures ==="
speculect pair --target target-7b.gguf --gguf-dir examples/fixtures || true

echo
echo "=== speculect bench --target target-7b.gguf --draft draft-500m.gguf --mock ==="
speculect bench --target target-7b.gguf --draft draft-500m.gguf --mock
