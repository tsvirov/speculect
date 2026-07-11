import pytest

from speculect.bench import BenchRunnerError, LlamaCppRunner, MockRunner


def test_mock_runner_deterministic():
    runner = MockRunner()
    r1 = runner.run("target.gguf", "draft.gguf", ["hello"])
    r2 = runner.run("target.gguf", "draft.gguf", ["hello"])
    assert r1 == r2
    assert r1.tokens_per_sec_spec > r1.tokens_per_sec_baseline
    assert 0.0 <= r1.acceptance_rate <= 1.0


def test_mock_runner_requires_prompts():
    runner = MockRunner()
    with pytest.raises(BenchRunnerError):
        runner.run("target.gguf", "draft.gguf", [])


def test_bench_result_speedup():
    from speculect.bench import BenchResult

    result = BenchResult(
        tokens_per_sec_baseline=20.0, tokens_per_sec_spec=40.0, acceptance_rate=0.7
    )
    assert result.speedup == 2.0


def test_bench_result_speedup_no_division_by_zero():
    from speculect.bench import BenchResult

    result = BenchResult(tokens_per_sec_baseline=0.0, tokens_per_sec_spec=40.0, acceptance_rate=0.7)
    assert result.speedup == 0.0


def test_llamacpp_runner_missing_binary_gives_clear_error(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: None)
    runner = LlamaCppRunner()
    with pytest.raises(BenchRunnerError, match="install llama.cpp"):
        runner.run("target.gguf", "draft.gguf", ["hello"])


def test_llamacpp_runner_uses_explicit_binary_path():
    runner = LlamaCppRunner(binary="/nonexistent/llama-speculative")
    with pytest.raises(BenchRunnerError, match="failed to run"):
        runner.run("target.gguf", "draft.gguf", ["hello"])


def test_llamacpp_parse_output_percent_form():
    output = "baseline: 18.2 tok/s\nspeculative: 39.9 tok/s\nacceptance rate: 68%\n"
    result = LlamaCppRunner.parse_output(output)
    assert result.tokens_per_sec_baseline == 18.2
    assert result.tokens_per_sec_spec == 39.9
    assert result.acceptance_rate == 0.68


def test_llamacpp_parse_output_fraction_form():
    output = "baseline: 18.2 tokens/s\nspeculative: 39.9 tokens/s\naccept: 0.68\n"
    result = LlamaCppRunner.parse_output(output)
    assert result.acceptance_rate == 0.68


def test_llamacpp_parse_output_unparseable_raises_clear_error():
    with pytest.raises(BenchRunnerError, match="could not parse"):
        LlamaCppRunner.parse_output("garbage output with no numbers")


def test_llamacpp_runner_finds_binary_via_which(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: f"/usr/local/bin/{name}")
    runner = LlamaCppRunner()
    assert runner._binary == "/usr/local/bin/llama-speculative"


def test_llamacpp_runner_falls_back_to_second_binary_name(monkeypatch):
    def fake_which(name):
        return "/usr/local/bin/llama-server" if name == "llama-server" else None

    monkeypatch.setattr("shutil.which", fake_which)
    runner = LlamaCppRunner()
    assert runner._binary == "/usr/local/bin/llama-server"
