import pytest
from click.testing import CliRunner

from speculect.cli import main


@pytest.fixture
def runner():
    return CliRunner()


def test_version(runner):
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "speculect" in result.output


def test_scan_gguf_dir_exit_0(runner, tmp_path, gguf_builder, model_metadata):
    (tmp_path / "a.gguf").write_bytes(gguf_builder(metadata=model_metadata(architecture="llama")))
    result = runner.invoke(main, ["scan", "--gguf-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "a.gguf" in result.output
    assert "llama" in result.output


def test_scan_gguf_dir_empty_exit_0(runner, tmp_path):
    result = runner.invoke(main, ["scan", "--gguf-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "no models found" in result.output


def test_scan_gguf_dir_reports_corrupt_file_as_warning(
    runner, tmp_path, gguf_builder, model_metadata
):
    (tmp_path / "good.gguf").write_bytes(gguf_builder(metadata=model_metadata()))
    (tmp_path / "bad.gguf").write_bytes(b"NOTGGUF")
    result = runner.invoke(main, ["scan", "--gguf-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "warning" in result.output
    assert "bad.gguf" in result.output


def test_scan_infra_error_exit_2(runner, monkeypatch):
    from speculect.ollama import OllamaError

    def fake_list_models(self):
        raise OllamaError("cannot reach Ollama at http://localhost:11434: refused")

    monkeypatch.setattr("speculect.ollama.OllamaClient.list_models", fake_list_models)
    result = runner.invoke(main, ["scan"])
    assert result.exit_code == 2
    assert "error" in result.output


def test_pair_no_compatible_candidates_exit_1(runner, tmp_path, gguf_builder, model_metadata):
    (tmp_path / "solo.gguf").write_bytes(gguf_builder(metadata=model_metadata()))
    result = runner.invoke(main, ["pair", "--target", "solo.gguf", "--gguf-dir", str(tmp_path)])
    assert result.exit_code == 1
    assert "no other models" in result.output


def test_pair_target_not_found_exit_1(runner, tmp_path, gguf_builder, model_metadata):
    (tmp_path / "a.gguf").write_bytes(gguf_builder(metadata=model_metadata()))
    result = runner.invoke(main, ["pair", "--target", "missing.gguf", "--gguf-dir", str(tmp_path)])
    assert result.exit_code == 1
    assert "not found" in result.output


def test_pair_finds_compatible_draft_exit_0(runner, tmp_path, gguf_builder, model_metadata):
    (tmp_path / "target.gguf").write_bytes(
        gguf_builder(metadata=model_metadata(architecture="llama", parameter_count=6_000_000_000))
    )
    (tmp_path / "draft.gguf").write_bytes(
        gguf_builder(metadata=model_metadata(architecture="llama", parameter_count=500_000_000))
    )
    result = runner.invoke(main, ["pair", "--target", "target.gguf", "--gguf-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "COMPATIBLE" in result.output
    assert "draft.gguf" in result.output


def test_pair_only_incompatible_candidates_exit_1(runner, tmp_path, gguf_builder, model_metadata):
    (tmp_path / "target.gguf").write_bytes(
        gguf_builder(metadata=model_metadata(architecture="llama"))
    )
    (tmp_path / "draft.gguf").write_bytes(
        gguf_builder(metadata=model_metadata(architecture="gemma"))
    )
    result = runner.invoke(main, ["pair", "--target", "target.gguf", "--gguf-dir", str(tmp_path)])
    assert result.exit_code == 1
    assert "INCOMPATIBLE" in result.output


def test_pair_infra_error_exit_2(runner, monkeypatch):
    from speculect.ollama import OllamaError

    def fake_list_models(self):
        raise OllamaError("cannot reach Ollama at http://localhost:11434: refused")

    monkeypatch.setattr("speculect.ollama.OllamaClient.list_models", fake_list_models)
    result = runner.invoke(main, ["pair", "--target", "anything"])
    assert result.exit_code == 2


def test_bench_mock_exit_0(runner):
    result = runner.invoke(main, ["bench", "--target", "t.gguf", "--draft", "d.gguf", "--mock"])
    assert result.exit_code == 0
    assert "baseline" in result.output
    assert "speculative" in result.output
    assert "speedup" in result.output


def test_bench_no_binary_exit_2(runner, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: None)
    result = runner.invoke(main, ["bench", "--target", "t.gguf", "--draft", "d.gguf"])
    assert result.exit_code == 2
    assert "install llama.cpp" in result.output


def test_bench_custom_prompts(runner):
    result = runner.invoke(
        main,
        [
            "bench", "--target", "t.gguf", "--draft", "d.gguf",
            "--mock", "--prompt", "hi", "--prompt", "there",
        ],
    )
    assert result.exit_code == 0


def test_scan_with_ollama_mocked(runner, monkeypatch):
    def fake_list_models(self):
        from speculect.ollama import OllamaModel

        return [OllamaModel(name="gemma4:latest", family="gemma4")]

    def fake_show_model(self, name, verbose=True):
        return {
            "model_info": {
                "general.architecture": "gemma4",
                "tokenizer.ggml.model": "llama",
                "tokenizer.ggml.tokens": ["<pad>", "<eos>"],
                "general.parameter_count": 7996157674,
            }
        }

    monkeypatch.setattr("speculect.ollama.OllamaClient.list_models", fake_list_models)
    monkeypatch.setattr("speculect.ollama.OllamaClient.show_model", fake_show_model)
    result = runner.invoke(main, ["scan"])
    assert result.exit_code == 0
    assert "gemma4:latest" in result.output
    assert "gemma4" in result.output


def test_scan_ollama_show_failure_becomes_warning_not_crash(runner, monkeypatch):
    from speculect.ollama import OllamaError, OllamaModel

    def fake_list_models(self):
        return [OllamaModel(name="broken:latest")]

    def fake_show_model(self, name, verbose=True):
        raise OllamaError(f"model not found in Ollama: {name}")

    monkeypatch.setattr("speculect.ollama.OllamaClient.list_models", fake_list_models)
    monkeypatch.setattr("speculect.ollama.OllamaClient.show_model", fake_show_model)
    result = runner.invoke(main, ["scan"])
    assert result.exit_code == 0
    assert "warning" in result.output
    assert "no models found" in result.output
