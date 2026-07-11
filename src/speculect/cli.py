"""Command-line interface for speculect."""
import sys
from typing import Optional

import click

from . import __version__
from .bench import BenchRunnerError, LlamaCppRunner, MockRunner
from .compat import ModelInfo, Verdict, check_compat
from .gguf import GGUFFileInfo, scan_directory, token_hash_of
from .ollama import OllamaClient, OllamaError

EXIT_OK = 0
EXIT_NO_COMPATIBLE_CANDIDATES = 1
EXIT_INFRA_ERROR = 2


class InfraError(Exception):
    """Signal for exit code 2: no usable source of model metadata."""


def _model_info_from_gguf(file_info: GGUFFileInfo) -> ModelInfo:
    h = file_info.header
    return ModelInfo(
        name=file_info.name,
        architecture=h.architecture,
        tokenizer_model=h.tokenizer_model,
        vocab_size=h.vocab_size,
        token_hash=h.token_hash,
        parameter_count=h.parameter_count,
    )


def model_info_from_ollama_show(name: str, show_data: dict) -> ModelInfo:
    """Build a ModelInfo from a POST /api/show response.

    Ollama exposes GGUF-style metadata under ``model_info`` (same key names
    as the GGUF header: ``general.architecture``, ``tokenizer.ggml.model``,
    ``tokenizer.ggml.tokens``, ``general.parameter_count``) when the request
    includes ``"verbose": true``. Fields absent from ``model_info`` are left
    as ``None`` rather than guessed from ``details`` — that keeps compat
    verdicts honestly UNKNOWN instead of silently inferring from a coarser
    signal like ``details.family``.
    """
    model_info = show_data.get("model_info") or {}
    architecture = model_info.get("general.architecture")
    tokenizer_model = model_info.get("tokenizer.ggml.model")
    tokens = model_info.get("tokenizer.ggml.tokens")
    parameter_count = model_info.get("general.parameter_count")

    vocab_size = None
    token_hash = None
    if isinstance(tokens, list) and tokens:
        vocab_size = len(tokens)
        token_hash = token_hash_of(tokens)

    return ModelInfo(
        name=name,
        architecture=architecture if isinstance(architecture, str) else None,
        tokenizer_model=tokenizer_model if isinstance(tokenizer_model, str) else None,
        vocab_size=vocab_size,
        token_hash=token_hash,
        parameter_count=parameter_count if isinstance(parameter_count, int) else None,
    )


def _collect_models(gguf_dir: Optional[str], base_url: str) -> tuple[list[ModelInfo], list[str]]:
    """Gather ModelInfo from --gguf-dir if given, else from Ollama.

    Returns (models, warnings). Raises InfraError when Ollama is unreachable
    and no --gguf-dir was given to fall back on.
    """
    warnings: list[str] = []

    if gguf_dir:
        result = scan_directory(gguf_dir)
        for skipped in result.skipped:
            warnings.append(f"skipping {skipped.path}: {skipped.reason}")
        return [_model_info_from_gguf(f) for f in result.files], warnings

    client = OllamaClient(base_url=base_url)
    try:
        ollama_models = client.list_models()
    except OllamaError as exc:
        raise InfraError(
            f"{exc} (pass --gguf-dir to scan local GGUF files instead)"
        ) from exc

    models: list[ModelInfo] = []
    for m in ollama_models:
        try:
            show_data = client.show_model(m.name)
        except OllamaError as exc:
            warnings.append(f"skipping {m.name}: {exc}")
            continue
        models.append(model_info_from_ollama_show(m.name, show_data))
    return models, warnings


def _print_warnings(warnings: list[str]) -> None:
    for w in warnings:
        click.echo(f"warning: {w}", err=True)


@click.group()
@click.version_option(__version__, "--version", prog_name="speculect")
def main() -> None:
    """speculect — auto-pair a draft model with a target for speculative decoding."""


@main.command()
@click.option(
    "--gguf-dir",
    type=click.Path(exists=True, file_okay=False),
    default=None,
    help="Scan a directory of .gguf files instead of querying Ollama.",
)
@click.option(
    "--base-url", default="http://localhost:11434", show_default=True, help="Ollama server URL."
)
def scan(gguf_dir: Optional[str], base_url: str) -> None:
    """List models found (via --gguf-dir or Ollama) with their metadata."""
    try:
        models, warnings = _collect_models(gguf_dir, base_url)
    except InfraError as exc:
        click.echo(f"error: {exc}", err=True)
        sys.exit(EXIT_INFRA_ERROR)
        return

    _print_warnings(warnings)

    if not models:
        click.echo("no models found.")
        sys.exit(EXIT_OK)
        return

    click.echo(
        "{:<30} {:<14} {:<10} {:>10} {:>16}".format(
            "NAME", "ARCHITECTURE", "TOKENIZER", "VOCAB", "PARAMS"
        )
    )
    for m in models:
        click.echo(
            "{:<30} {:<14} {:<10} {:>10} {:>16}".format(
                m.name,
                m.architecture or "?",
                m.tokenizer_model or "?",
                m.vocab_size if m.vocab_size is not None else "?",
                f"{m.parameter_count:,}" if m.parameter_count else "?",
            )
        )
    sys.exit(EXIT_OK)


@main.command()
@click.option("--target", required=True, help="Name of the target model.")
@click.option("--gguf-dir", type=click.Path(exists=True, file_okay=False), default=None)
@click.option("--base-url", default="http://localhost:11434", show_default=True)
def pair(target: str, gguf_dir: Optional[str], base_url: str) -> None:
    """Rank candidate draft models for --target, with a verdict and reason each."""
    try:
        models, warnings = _collect_models(gguf_dir, base_url)
    except InfraError as exc:
        click.echo(f"error: {exc}", err=True)
        sys.exit(EXIT_INFRA_ERROR)
        return

    _print_warnings(warnings)

    target_model = next((m for m in models if m.name == target), None)
    if target_model is None:
        click.echo(
            f"no compatible candidates: target model not found: {target}", err=True
        )
        sys.exit(EXIT_NO_COMPATIBLE_CANDIDATES)
        return

    candidates = [m for m in models if m.name != target]
    if not candidates:
        click.echo("no other models available to pair as a draft.")
        sys.exit(EXIT_NO_COMPATIBLE_CANDIDATES)
        return

    ranked = [(c, check_compat(target_model, c)) for c in candidates]
    rank_order = {Verdict.COMPATIBLE: 0, Verdict.UNKNOWN: 1, Verdict.INCOMPATIBLE: 2}
    ranked.sort(key=lambda pair_: rank_order[pair_[1].verdict])

    click.echo(f"target: {target}")
    click.echo()
    any_compatible = False
    for candidate, result in ranked:
        if result.verdict == Verdict.COMPATIBLE:
            any_compatible = True
        click.echo(f"[{result.verdict.value}] {candidate.name} — {result.reason}")
        if result.size_warning:
            click.echo(f"    warning: {result.size_warning}")

    sys.exit(EXIT_OK if any_compatible else EXIT_NO_COMPATIBLE_CANDIDATES)


@main.command()
@click.option("--target", required=True, help="Target model name or GGUF path.")
@click.option("--draft", required=True, help="Draft model name or GGUF path.")
@click.option("--mock", is_flag=True, help="Use deterministic fake numbers instead of llama.cpp.")
@click.option("--prompt", "prompts", multiple=True, help="Prompt to benchmark (repeatable).")
def bench(target: str, draft: str, mock: bool, prompts: tuple[str, ...]) -> None:
    """Benchmark a target/draft pair — tokens/sec baseline vs speculative, acceptance rate."""
    prompt_list = list(prompts) or ["Explain speculative decoding in one sentence."]
    runner = MockRunner() if mock else LlamaCppRunner()
    try:
        result = runner.run(target, draft, prompt_list)
    except BenchRunnerError as exc:
        click.echo(f"error: {exc}", err=True)
        sys.exit(EXIT_INFRA_ERROR)
        return

    click.echo(f"baseline:    {result.tokens_per_sec_baseline:.1f} tok/s")
    click.echo(f"speculative: {result.tokens_per_sec_spec:.1f} tok/s")
    click.echo(f"speedup:     {result.speedup:.2f}x")
    click.echo(f"acceptance:  {result.acceptance_rate:.0%}")
    sys.exit(EXIT_OK)


if __name__ == "__main__":
    main()
