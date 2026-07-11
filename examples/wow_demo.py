#!/usr/bin/env python3
"""A styled terminal walkthrough of speculect — for recording a README GIF.

Every number shown here comes from actually running speculect's scan/compat/
bench logic against the fixture GGUF files in examples/fixtures/ (the same
files examples/demo.sh and the test suite use) — nothing is fabricated, only
the pacing and colors are for show. Requires `rich`:

    pip install -e ".[demo]"
    python examples/wow_demo.py               # animated, live in your terminal
    python examples/wow_demo.py --svg out.svg  # static snapshot, no animation
"""
import argparse
import sys
import time
from pathlib import Path

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import BarColumn, Progress, TextColumn
    from rich.table import Table
    from rich.text import Text
except ImportError:
    print('This demo needs "rich". Install with: pip install -e ".[demo]"', file=sys.stderr)
    sys.exit(1)

from speculect.bench import MockRunner
from speculect.compat import ModelInfo, Verdict, check_compat
from speculect.gguf import GGUFFileInfo, scan_directory

FIXTURES_DIR = Path(__file__).parent / "fixtures"
TARGET_NAME = "target-7b.gguf"
DRAFT_NAME = "draft-500m.gguf"

console = Console()


def beat(seconds: float = 0.5) -> None:
    time.sleep(seconds)


def model_info_from_gguf(file_info: GGUFFileInfo) -> ModelInfo:
    h = file_info.header
    return ModelInfo(
        name=file_info.name,
        architecture=h.architecture,
        tokenizer_model=h.tokenizer_model,
        vocab_size=h.vocab_size,
        token_hash=h.token_hash,
        parameter_count=h.parameter_count,
    )


def print_intro() -> None:
    console.print()
    console.print(
        Panel.fit(
            Text("speculect", style="bold cyan")
            + Text("  —  auto-pair a draft model for speculative decoding", style="dim"),
            border_style="cyan",
        )
    )
    beat(0.6)


def scan_and_show(animate: bool) -> list:
    if animate:
        msg = "[bold cyan]scanning examples/fixtures for GGUF models..."
        with console.status(msg, spinner="dots"):
            beat(1.2)
            result = scan_directory(str(FIXTURES_DIR))
    else:
        result = scan_directory(str(FIXTURES_DIR))

    table = Table(title="models found", border_style="grey50")
    table.add_column("NAME", style="bold")
    table.add_column("ARCH")
    table.add_column("TOKENIZER")
    table.add_column("VOCAB", justify="right")
    table.add_column("PARAMS", justify="right")
    models = [model_info_from_gguf(f) for f in result.files]
    for m in models:
        table.add_row(
            m.name,
            m.architecture or "?",
            m.tokenizer_model or "?",
            str(m.vocab_size) if m.vocab_size is not None else "?",
            f"{m.parameter_count:,}" if m.parameter_count else "?",
        )
    console.print(table)
    if animate:
        beat(0.8)
    return models


def check_candidates(models: list, animate: bool) -> None:
    target = next(m for m in models if m.name == TARGET_NAME)
    console.print()
    console.print(f"[bold]target:[/bold] {TARGET_NAME}")
    if animate:
        beat(0.4)

    verdict_style = {
        Verdict.COMPATIBLE: ("green", "COMPATIBLE"),
        Verdict.INCOMPATIBLE: ("red", "INCOMPATIBLE"),
        Verdict.UNKNOWN: ("yellow", "UNKNOWN"),
    }

    for candidate in (m for m in models if m.name != TARGET_NAME):
        if animate:
            status_msg = f"[cyan]checking {candidate.name} against {TARGET_NAME}..."
            with console.status(status_msg, spinner="dots"):
                beat(0.9)
        result = check_compat(target, candidate)
        color, label = verdict_style[result.verdict]
        body = (
            f"[bold {color}]{label}[/bold {color}]  {candidate.name}\n"
            f"[dim]{result.reason}[/dim]"
        )
        console.print(Panel(body, border_style=color))
        if animate:
            beat(0.3)


def run_bench(animate: bool) -> None:
    console.print()
    console.print(f"[bold]benchmarking[/bold] {TARGET_NAME} + {DRAFT_NAME} [dim](--mock)[/dim]")
    if animate:
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("running speculative decode...", total=100)
            while not progress.finished:
                progress.update(task, advance=4)
                time.sleep(0.03)

    result = MockRunner().run(
        TARGET_NAME, DRAFT_NAME, ["Explain speculative decoding in one sentence."]
    )
    console.print(
        Panel(
            f"baseline:    {result.tokens_per_sec_baseline:.1f} tok/s\n"
            f"speculative: {result.tokens_per_sec_spec:.1f} tok/s\n"
            f"[bold]speedup:     {result.speedup:.2f}x[/bold]\n"
            f"acceptance:  {result.acceptance_rate:.0%}",
            title="[bold green]bench result[/bold green]",
            border_style="green",
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--svg", metavar="PATH", help="skip animation, export a static SVG snapshot to PATH"
    )
    args = parser.parse_args()
    animate = args.svg is None

    global console
    if args.svg:
        console = Console(record=True, width=100)

    print_intro()
    models = scan_and_show(animate)
    check_candidates(models, animate)
    run_bench(animate)
    console.print()
    console.print(
        "[bold green]done[/bold green] — real speedup needs a real llama.cpp binary. "
        "[dim]See README.[/dim]"
    )

    if args.svg:
        console.save_svg(args.svg, title="speculect")


if __name__ == "__main__":
    main()
