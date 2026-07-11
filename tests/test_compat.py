from speculect.compat import ModelInfo, Verdict, check_compat


def _model(**overrides):
    base = dict(
        name="model",
        architecture="llama",
        tokenizer_model="llama",
        vocab_size=32000,
        token_hash="deadbeef",
        parameter_count=1_000_000_000,
    )
    base.update(overrides)
    return ModelInfo(**base)


def test_compatible_when_everything_matches():
    target = _model(name="target")
    draft = _model(name="draft", parameter_count=100_000_000)  # 1/10th, under threshold
    result = check_compat(target, draft)
    assert result.verdict == Verdict.COMPATIBLE
    assert result.size_warning is None


def test_incompatible_architecture_mismatch():
    target = _model(architecture="llama")
    draft = _model(architecture="gemma")
    result = check_compat(target, draft)
    assert result.verdict == Verdict.INCOMPATIBLE
    assert "architecture" in result.reason


def test_incompatible_tokenizer_mismatch():
    target = _model(tokenizer_model="llama")
    draft = _model(tokenizer_model="gpt2")
    result = check_compat(target, draft)
    assert result.verdict == Verdict.INCOMPATIBLE
    assert "tokenizer" in result.reason


def test_incompatible_vocab_size_mismatch():
    target = _model(vocab_size=32000)
    draft = _model(vocab_size=32001)
    result = check_compat(target, draft)
    assert result.verdict == Verdict.INCOMPATIBLE
    assert "vocab size" in result.reason


def test_incompatible_token_hash_mismatch():
    target = _model(token_hash="aaaa")
    draft = _model(token_hash="bbbb")
    result = check_compat(target, draft)
    assert result.verdict == Verdict.INCOMPATIBLE
    assert "token hash" in result.reason


def test_compatible_when_token_hash_missing_on_one_side():
    # rule 4 only applies when BOTH sides have a hash
    target = _model(token_hash=None)
    draft = _model(token_hash="bbbb")
    result = check_compat(target, draft)
    assert result.verdict == Verdict.COMPATIBLE


def test_unknown_when_architecture_missing():
    target = _model(architecture=None)
    draft = _model()
    result = check_compat(target, draft)
    assert result.verdict == Verdict.UNKNOWN
    assert "architecture" in result.reason


def test_unknown_when_tokenizer_missing():
    target = _model()
    draft = _model(tokenizer_model=None)
    result = check_compat(target, draft)
    assert result.verdict == Verdict.UNKNOWN
    assert "tokenizer.ggml.model" in result.reason


def test_unknown_when_vocab_size_missing():
    target = _model(vocab_size=None)
    draft = _model()
    result = check_compat(target, draft)
    assert result.verdict == Verdict.UNKNOWN
    assert "vocab size" in result.reason


def test_unknown_lists_all_missing_fields():
    target = _model(architecture=None, tokenizer_model=None, vocab_size=None)
    draft = _model()
    result = check_compat(target, draft)
    assert result.verdict == Verdict.UNKNOWN
    assert "architecture" in result.reason
    assert "tokenizer.ggml.model" in result.reason
    assert "vocab size" in result.reason


def test_never_silently_reports_compatible_on_missing_metadata():
    # regression guard: no metadata at all must never resolve to COMPATIBLE
    target = ModelInfo(name="t")
    draft = ModelInfo(name="d")
    result = check_compat(target, draft)
    assert result.verdict == Verdict.UNKNOWN


def test_size_warning_when_draft_too_large():
    target = _model(parameter_count=1_000_000_000)
    draft = _model(parameter_count=500_000_000)  # 1/2, over the 1/6 threshold
    result = check_compat(target, draft)
    assert result.verdict == Verdict.COMPATIBLE
    assert result.size_warning is not None
    assert "speedup" in result.size_warning


def test_size_warning_absent_at_exact_threshold_boundary():
    target = _model(parameter_count=6_000_000_000)
    draft = _model(parameter_count=1_000_000_000)  # exactly 1/6
    result = check_compat(target, draft)
    assert result.verdict == Verdict.COMPATIBLE
    assert result.size_warning is None


def test_size_warning_absent_when_parameter_counts_unknown():
    target = _model(parameter_count=None)
    draft = _model(parameter_count=None)
    result = check_compat(target, draft)
    assert result.verdict == Verdict.COMPATIBLE
    assert result.size_warning is None


def test_no_division_by_zero_when_target_params_zero():
    target = _model(parameter_count=0)
    draft = _model(parameter_count=100)
    # target.parameter_count is falsy (0), so the size heuristic is skipped
    # entirely rather than dividing by zero.
    result = check_compat(target, draft)
    assert result.verdict == Verdict.COMPATIBLE
    assert result.size_warning is None
