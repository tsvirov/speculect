# Contributing to speculect

Thanks for considering a contribution.

## Setup

```bash
git clone https://github.com/tsvirov/speculect.git
cd speculect
python3 -m venv .venv
.venv/bin/pip install -U pip
.venv/bin/pip install -e ".[dev]"
```

## Before opening a PR

```bash
.venv/bin/pytest -q
.venv/bin/ruff check .
```

Both must pass. CI runs the same checks on Python 3.9–3.12 — this project
supports Python 3.9, so avoid `X | Y` union syntax in type annotations and
`match` statements; use `typing.Optional`/`Union` instead.

## Guidelines

- Tests must not touch the network or require a real Ollama/llama.cpp
  install — use `httpx.MockTransport` and the GGUF fixture builder in
  `tests/conftest.py`.
- Keep the GGUF parser reading only the header (via `seek`/`read`), never
  loading a whole model file into memory.
- If you add a CLI flag, update `README.md`'s CLI usage section in the same PR.

## Reporting bugs / requesting features

Open a GitHub issue using the provided template.
