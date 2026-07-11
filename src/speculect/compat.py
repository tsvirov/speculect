"""Draft/target compatibility rules for speculative decoding.

Rules are applied in order (each requires the previous ones to have already
matched or be inconclusive):

  1. same ``architecture``
  2. same ``tokenizer.ggml.model``
  3. equal vocab size
  4. if both sides have a token hash, it must match

Plus a size heuristic: a draft model should be roughly <= 1/6th of the
target's parameter count, or the speedup from speculative decoding is
unlikely to materialize.

A verdict of UNKNOWN is returned whenever there isn't enough metadata to
decide — speculect never silently guesses COMPATIBLE.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional

SIZE_RATIO_THRESHOLD = 1.0 / 6.0


class Verdict(Enum):
    COMPATIBLE = "COMPATIBLE"
    INCOMPATIBLE = "INCOMPATIBLE"
    UNKNOWN = "UNKNOWN"


@dataclass
class ModelInfo:
    """Normalized metadata used for compatibility checks, regardless of
    whether it came from a GGUF file header or an Ollama /api/show response.
    """

    name: str
    architecture: Optional[str] = None
    tokenizer_model: Optional[str] = None
    vocab_size: Optional[int] = None
    token_hash: Optional[str] = None
    parameter_count: Optional[int] = None


@dataclass
class CompatResult:
    verdict: Verdict
    reason: str
    size_warning: Optional[str] = None


def check_compat(target: ModelInfo, draft: ModelInfo) -> CompatResult:
    missing: list[str] = []

    if target.architecture is None or draft.architecture is None:
        missing.append("architecture")
    elif target.architecture != draft.architecture:
        return CompatResult(
            Verdict.INCOMPATIBLE,
            f"architecture mismatch: target={target.architecture!r} draft={draft.architecture!r}",
        )

    if target.tokenizer_model is None or draft.tokenizer_model is None:
        missing.append("tokenizer.ggml.model")
    elif target.tokenizer_model != draft.tokenizer_model:
        return CompatResult(
            Verdict.INCOMPATIBLE,
            f"tokenizer mismatch: target={target.tokenizer_model!r} "
            f"draft={draft.tokenizer_model!r}",
        )

    if target.vocab_size is None or draft.vocab_size is None:
        missing.append("vocab size")
    elif target.vocab_size != draft.vocab_size:
        return CompatResult(
            Verdict.INCOMPATIBLE,
            f"vocab size mismatch: target={target.vocab_size} draft={draft.vocab_size}",
        )

    if target.token_hash and draft.token_hash and target.token_hash != draft.token_hash:
        return CompatResult(
            Verdict.INCOMPATIBLE,
            f"token hash mismatch: first {256} tokens differ between target and draft",
        )

    if missing:
        return CompatResult(
            Verdict.UNKNOWN,
            "missing metadata to decide: {}".format(", ".join(missing)),
        )

    result = CompatResult(
        Verdict.COMPATIBLE,
        "architecture, tokenizer, vocab size" + (
            ", and token hash all match" if (target.token_hash and draft.token_hash) else " match"
        ),
    )

    if target.parameter_count and draft.parameter_count:
        ratio = draft.parameter_count / target.parameter_count
        if ratio > SIZE_RATIO_THRESHOLD:
            result.size_warning = (
                f"draft is {ratio:.0%} of target's parameter count "
                f"(target={target.parameter_count:,}, draft={draft.parameter_count:,}); "
                f"should be <= {SIZE_RATIO_THRESHOLD:.0%} for a meaningful speedup"
            )

    return result
