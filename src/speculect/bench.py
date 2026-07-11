"""Benchmark runners for a draft/target pair.

``BenchRunner`` is a small protocol so the CLI and tests can swap in a
deterministic ``MockRunner`` without needing a real llama.cpp binary.
"""
import re
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional, Protocol


class BenchRunnerError(Exception):
    """Raised when a runner cannot produce a result (missing binary, unparseable output)."""


@dataclass
class BenchResult:
    tokens_per_sec_baseline: float
    tokens_per_sec_spec: float
    acceptance_rate: float

    @property
    def speedup(self) -> float:
        if self.tokens_per_sec_baseline == 0:
            return 0.0
        return self.tokens_per_sec_spec / self.tokens_per_sec_baseline


class BenchRunner(Protocol):
    def run(self, target: str, draft: str, prompts: Sequence[str]) -> BenchResult: ...


class MockRunner:
    """Deterministic fake numbers — powers tests and the offline demo.

    Values are fixed, not random, so `examples/demo.sh` output is
    reproducible and can be pasted verbatim into examples/README.md.
    """

    def run(self, target: str, draft: str, prompts: Sequence[str]) -> BenchResult:
        if not prompts:
            raise BenchRunnerError("at least one prompt is required to run a benchmark")
        baseline = 18.4
        spec = 41.2
        acceptance = 0.71
        return BenchResult(
            tokens_per_sec_baseline=baseline,
            tokens_per_sec_spec=spec,
            acceptance_rate=acceptance,
        )


_BASELINE_RE = re.compile(r"baseline[:\s]+([\d.]+)\s*t(?:ok(?:en)?s?)?/s", re.IGNORECASE)
_SPEC_RE = re.compile(r"speculative[:\s]+([\d.]+)\s*t(?:ok(?:en)?s?)?/s", re.IGNORECASE)
_ACCEPT_RE = re.compile(r"accept(?:ance)?(?:\s*rate)?[:\s]+([\d.]+)\s*%?", re.IGNORECASE)


class LlamaCppRunner:
    """Runs a real llama.cpp speculative-decoding binary and parses its output.

    Looks for ``llama-speculative`` first, then ``llama-server``, via
    ``shutil.which``. If neither is on PATH, raises a ``BenchRunnerError``
    with install instructions instead of letting a ``FileNotFoundError``
    traceback surface.
    """

    BINARY_NAMES = ("llama-speculative", "llama-server")

    def __init__(self, binary: Optional[str] = None) -> None:
        self._binary = binary or self._find_binary()

    def _find_binary(self) -> Optional[str]:
        for name in self.BINARY_NAMES:
            found = shutil.which(name)
            if found:
                return found
        return None

    def run(self, target: str, draft: str, prompts: Sequence[str]) -> BenchResult:
        if not self._binary:
            raise BenchRunnerError(
                "no llama.cpp binary found on PATH (looked for: {}). "
                "install llama.cpp, see README".format(", ".join(self.BINARY_NAMES))
            )
        cmd: list[str] = [self._binary, "-m", target, "-md", draft]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, check=False, timeout=300
            )
        except OSError as exc:
            raise BenchRunnerError(
                f"failed to run {self._binary}: {exc}. install llama.cpp, see README"
            ) from exc
        return self.parse_output(proc.stdout + "\n" + proc.stderr)

    @staticmethod
    def parse_output(output: str) -> BenchResult:
        baseline_m = _BASELINE_RE.search(output)
        spec_m = _SPEC_RE.search(output)
        accept_m = _ACCEPT_RE.search(output)
        if not (baseline_m and spec_m and accept_m):
            raise BenchRunnerError(
                "could not parse llama.cpp output — expected lines matching "
                "'baseline: N tok/s', 'speculative: N tok/s', 'acceptance: N%'"
            )
        accept_val = float(accept_m.group(1))
        if accept_val > 1.0:
            accept_val /= 100.0
        return BenchResult(
            tokens_per_sec_baseline=float(baseline_m.group(1)),
            tokens_per_sec_spec=float(spec_m.group(1)),
            acceptance_rate=accept_val,
        )
