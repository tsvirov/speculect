"""Minimal Ollama HTTP client.

The httpx client is injected via the constructor so tests can pass an
``httpx.Client(transport=httpx.MockTransport(...))`` and never touch a real
network socket.
"""
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_TIMEOUT = 5.0


class OllamaError(Exception):
    """Raised for any failure talking to an Ollama server."""


@dataclass
class OllamaModel:
    name: str
    family: Optional[str] = None
    families: list[str] = field(default_factory=list)
    parameter_size: Optional[str] = None
    quantization: Optional[str] = None
    size_bytes: Optional[int] = None


class OllamaClient:
    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        client: Optional[httpx.Client] = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = client or httpx.Client(base_url=self.base_url, timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "OllamaClient":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def _get(self, path: str) -> httpx.Response:
        try:
            return self._client.get(path)
        except httpx.RequestError as exc:
            raise OllamaError(
                f"cannot reach Ollama at {self.base_url}: {exc}"
            ) from exc

    def _post(self, path: str, json: dict[str, Any]) -> httpx.Response:
        try:
            return self._client.post(path, json=json)
        except httpx.RequestError as exc:
            raise OllamaError(
                f"cannot reach Ollama at {self.base_url}: {exc}"
            ) from exc

    def list_models(self) -> list[OllamaModel]:
        resp = self._get("/api/tags")
        if resp.status_code != 200:
            raise OllamaError(
                f"Ollama returned HTTP {resp.status_code} for GET /api/tags"
            )
        data = resp.json()
        models = []
        for entry in data.get("models", []):
            details = entry.get("details") or {}
            models.append(
                OllamaModel(
                    name=entry.get("name", ""),
                    family=details.get("family"),
                    families=details.get("families") or [],
                    parameter_size=details.get("parameter_size"),
                    quantization=details.get("quantization_level"),
                    size_bytes=entry.get("size"),
                )
            )
        return models

    def show_model(self, name: str, verbose: bool = True) -> dict[str, Any]:
        """Fetch model details, including ``model_info`` (GGUF-style metadata).

        ``verbose=True`` asks Ollama to include the full
        ``tokenizer.ggml.tokens`` array in ``model_info`` — without it,
        Ollama omits large arrays and speculect can't compute a token hash.
        """
        resp = self._post("/api/show", json={"name": name, "verbose": verbose})
        if resp.status_code == 404:
            raise OllamaError(f"model not found in Ollama: {name}")
        if resp.status_code != 200:
            raise OllamaError(
                f"Ollama returned HTTP {resp.status_code} for POST /api/show ({name})"
            )
        return resp.json()
