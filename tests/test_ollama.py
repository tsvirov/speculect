import httpx
import pytest

from speculect.ollama import OllamaClient, OllamaError


def _client(handler):
    return OllamaClient(
        base_url="http://test",
        client=httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test"),
    )


def test_list_models_success():
    def handler(request):
        assert request.url.path == "/api/tags"
        return httpx.Response(
            200,
            json={
                "models": [
                    {
                        "name": "gemma4:latest",
                        "size": 9608350718,
                        "details": {
                            "family": "gemma4",
                            "families": ["gemma4"],
                            "parameter_size": "8.0B",
                            "quantization_level": "Q4_K_M",
                        },
                    }
                ]
            },
        )

    client = _client(handler)
    models = client.list_models()
    assert len(models) == 1
    m = models[0]
    assert m.name == "gemma4:latest"
    assert m.family == "gemma4"
    assert m.families == ["gemma4"]
    assert m.parameter_size == "8.0B"
    assert m.quantization == "Q4_K_M"
    assert m.size_bytes == 9608350718


def test_list_models_empty():
    def handler(request):
        return httpx.Response(200, json={"models": []})

    client = _client(handler)
    assert client.list_models() == []


def test_list_models_server_error():
    def handler(request):
        return httpx.Response(500, text="internal error")

    client = _client(handler)
    with pytest.raises(OllamaError, match="HTTP 500"):
        client.list_models()


def test_list_models_unreachable_server():
    def handler(request):
        raise httpx.ConnectError("connection refused", request=request)

    client = _client(handler)
    with pytest.raises(OllamaError, match="cannot reach Ollama"):
        client.list_models()


def test_show_model_success_sends_verbose_true():
    seen = {}

    def handler(request):
        seen["body"] = request.read()
        return httpx.Response(
            200,
            json={
                "model_info": {
                    "general.architecture": "gemma4",
                    "tokenizer.ggml.model": "llama",
                    "tokenizer.ggml.tokens": ["<pad>", "<eos>", "hi"],
                    "general.parameter_count": 7996157674,
                }
            },
        )

    client = _client(handler)
    data = client.show_model("gemma4:latest")
    assert b'"verbose":true' in seen["body"] or b'"verbose": true' in seen["body"]
    assert data["model_info"]["general.architecture"] == "gemma4"


def test_show_model_not_found():
    def handler(request):
        return httpx.Response(404, json={"error": "not found"})

    client = _client(handler)
    with pytest.raises(OllamaError, match="model not found"):
        client.show_model("nonexistent:latest")


def test_show_model_unreachable_server():
    def handler(request):
        raise httpx.ConnectError("connection refused", request=request)

    client = _client(handler)
    with pytest.raises(OllamaError, match="cannot reach Ollama"):
        client.show_model("anything")


def test_client_context_manager_closes():
    def handler(request):
        return httpx.Response(200, json={"models": []})

    with _client(handler) as client:
        client.list_models()
    # closing twice must not raise
    client.close()


def test_list_models_invalid_json_raises_ollama_error():
    def handler(request):
        return httpx.Response(
            200, content=b"not json{{{", headers={"content-type": "application/json"}
        )

    client = _client(handler)
    with pytest.raises(OllamaError, match="invalid JSON"):
        client.list_models()


def test_show_model_invalid_json_raises_ollama_error():
    def handler(request):
        return httpx.Response(
            200, content=b"not json{{{", headers={"content-type": "application/json"}
        )

    client = _client(handler)
    with pytest.raises(OllamaError, match="invalid JSON"):
        client.show_model("anything")


def test_list_models_entry_missing_details_key():
    def handler(request):
        return httpx.Response(200, json={"models": [{"name": "bare:latest", "size": 123}]})

    client = _client(handler)
    models = client.list_models()
    assert len(models) == 1
    assert models[0].name == "bare:latest"
    assert models[0].family is None
    assert models[0].families == []
