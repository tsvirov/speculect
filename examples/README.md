# Offline demo

`demo.sh` runs `speculect scan`, `pair`, and `bench --mock` against three
fixture GGUF files in `examples/fixtures/` — built by the same programmatic
byte-builder used in the test suite (`tests/conftest.py`), not downloaded.
No Ollama, no llama.cpp, no network.

```bash
speculect scan --gguf-dir examples/fixtures
```

```
speculect scan --gguf-dir examples/fixtures
NAME                           ARCHITECTURE   TOKENIZER       VOCAB           PARAMS
draft-500m.gguf                llama          llama             303      500,000,000
incompatible-arch.gguf         gemma          llama               3      300,000,000
target-7b.gguf                 llama          llama             303    7,000,000,000
```

```bash
speculect pair --target target-7b.gguf --gguf-dir examples/fixtures
```

```
target: target-7b.gguf

[COMPATIBLE] draft-500m.gguf — architecture, tokenizer, vocab size, and token hash all match
[INCOMPATIBLE] incompatible-arch.gguf — architecture mismatch: target='llama' draft='gemma'
```

```bash
speculect bench --target target-7b.gguf --draft draft-500m.gguf --mock
```

```
baseline:    18.4 tok/s
speculative: 41.2 tok/s
speedup:     2.24x
acceptance:  71%
```

Run it yourself:

```bash
git clone https://github.com/tsvirov/speculect.git
cd speculect
python3 -m venv .venv && .venv/bin/pip install -e .
.venv/bin/pip install -e . >/dev/null && PATH=".venv/bin:$PATH" ./examples/demo.sh
```

The `bench --mock` numbers are fixed constants from `MockRunner`, not
measured — they exist to exercise the CLI's output format without needing a
real llama.cpp binary. See `README.md`'s Limitations section for what a real
benchmark run requires.

## Animated version

`wow_demo.py` walks through the same scan → compat → bench sequence above,
styled with [rich](https://github.com/Textualize/rich) (panels, spinners, a
progress bar) for recording a terminal GIF. All the numbers it prints are
the real output of speculect's library functions against the same fixture
files — the animation is cosmetic, the data isn't.

```bash
pip install -e ".[demo]"
python examples/wow_demo.py               # animated, live in your terminal
python examples/wow_demo.py --svg out.svg  # static snapshot (no animation) —
                                            # this is what's embedded at the
                                            # top of README.md
```
