"""Unified LLM client with mock, Ollama, and IBM watsonx providers.

The agents call ``LLMClient(provider).generate(prompt, system_prompt)`` and
get back a plain string. Each provider is wrapped in try/except so a missing
SDK or unreachable host degrades gracefully instead of crashing the app.

For structured outputs, callers can pass ``response_format="json"`` to coax
Ollama into producing parseable JSON (via the model's native JSON mode), or
use the higher-level :meth:`LLMClient.generate_validated` helper which also
validates the response against a Pydantic model and falls back cleanly when
the model produces malformed output.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple, Type, TypeVar

import requests

from src import config


SUPPORTED_PROVIDERS = ("mock", "ollama", "watsonx")

ResponseFormat = Literal["text", "json"]


@dataclass
class LLMResult:
    """Lightweight container so callers can detect mock fallbacks."""

    text: str
    provider_used: str
    fell_back: bool = False
    error: Optional[str] = None
    info: list[str] = field(default_factory=list)


# Bound TypeVar for ``generate_validated``; must be a Pydantic BaseModel.
TModel = TypeVar("TModel")


class LLMClient:
    """Provider-agnostic LLM client.

    Parameters
    ----------
    provider : str
        ``"mock"``, ``"ollama"``, or ``"watsonx"``.
    ollama_model : str | None
        Override the model id sent to Ollama.
    watsonx_model : str | None
        Override the model id sent to watsonx.
    timeout_seconds : int
        Network timeout for Ollama and watsonx requests.
    """

    def __init__(
        self,
        provider: str = "mock",
        ollama_model: Optional[str] = None,
        watsonx_model: Optional[str] = None,
        timeout_seconds: int = 60,
    ) -> None:
        provider = (provider or "mock").strip().lower()
        if provider not in SUPPORTED_PROVIDERS:
            provider = "mock"
        self.provider = provider
        self.ollama_model = ollama_model or config.OLLAMA_MODEL
        self.watsonx_model = watsonx_model or config.WATSONX_MODEL_ID
        self.timeout_seconds = timeout_seconds

    # --- Public API ----------------------------------------------------------

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.1,
        response_format: ResponseFormat = "text",
    ) -> LLMResult:
        """Generate text. Falls back to mock on any provider failure.

        Parameters
        ----------
        response_format:
            ``"text"`` (default) leaves the response unconstrained.
            ``"json"`` asks the provider to return parseable JSON. For Ollama
            this maps to the native ``format="json"`` option; for watsonx it
            is currently best-effort (we add a JSON-only system suffix but
            the model is still free to deviate).
        """
        if self.provider == "mock":
            return LLMResult(text=_mock_generate(prompt, system_prompt), provider_used="mock")

        if self.provider == "ollama":
            try:
                text = self._ollama_generate(prompt, system_prompt, temperature, response_format)
                return LLMResult(text=text, provider_used="ollama")
            except Exception as exc:
                return LLMResult(
                    text=_mock_generate(prompt, system_prompt),
                    provider_used="mock",
                    fell_back=True,
                    error=f"Ollama error: {exc}",
                )

        if self.provider == "watsonx":
            try:
                text = self._watsonx_generate(prompt, system_prompt, temperature, response_format)
                return LLMResult(text=text, provider_used="watsonx")
            except Exception as exc:
                return LLMResult(
                    text=_mock_generate(prompt, system_prompt),
                    provider_used="mock",
                    fell_back=True,
                    error=f"watsonx error: {exc}",
                )

        return LLMResult(text=_mock_generate(prompt, system_prompt), provider_used="mock")

    def generate_validated(
        self,
        prompt: str,
        system_prompt: str,
        model_cls: Type[TModel],
        temperature: float = 0.1,
    ) -> Tuple[Optional[TModel], LLMResult]:
        """Ask for JSON, validate against a Pydantic model.

        Returns ``(parsed_model_or_None, llm_result)``. Callers should treat
        ``None`` as a signal to fall back to deterministic logic. The
        ``LLMResult`` is always returned so the caller can inspect
        ``provider_used`` / ``fell_back`` for telemetry.

        ``model_cls`` must be a Pydantic ``BaseModel`` subclass; we don't
        import pydantic at module scope so the rest of the app keeps loading
        even if pydantic is missing.
        """
        result = self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            response_format="json",
        )
        if result.fell_back or not result.text:
            return None, result

        # Try to extract a JSON object from the response. Ollama's JSON mode
        # is reliable but not perfect on small models (occasional leading/
        # trailing prose), so we fall back to the lenient extractor.
        raw_text = result.text.strip()
        candidates: list[str] = [raw_text]
        snippet = _extract_json_snippet(raw_text)
        if snippet and snippet != raw_text:
            candidates.append(snippet)

        for candidate in candidates:
            try:
                payload = json.loads(candidate)
            except Exception:
                continue
            try:
                parsed = model_cls.model_validate(payload)  # type: ignore[attr-defined]
                return parsed, result
            except Exception:
                # Pydantic ValidationError or model_cls without model_validate.
                continue

        return None, result

    # --- Provider implementations -------------------------------------------

    def _ollama_generate(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float,
        response_format: ResponseFormat = "text",
    ) -> str:
        # Try the official ollama package first; fall back to raw HTTP.
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        json_mode = response_format == "json"

        try:
            import ollama  # type: ignore

            chat_kwargs: dict = {
                "model": self.ollama_model,
                "messages": messages,
                "options": {"temperature": float(temperature)},
            }
            if json_mode:
                chat_kwargs["format"] = "json"
            resp = ollama.chat(**chat_kwargs)
            content = resp.get("message", {}).get("content")
            if content:
                return str(content)
        except ImportError:
            pass
        except Exception:
            # Fall through to HTTP fallback.
            pass

        url = config.OLLAMA_HOST.rstrip("/") + "/api/chat"
        payload: dict = {
            "model": self.ollama_model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": float(temperature)},
        }
        if json_mode:
            payload["format"] = "json"
        resp = requests.post(url, json=payload, timeout=self.timeout_seconds)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict) and "message" in data:
            return str(data["message"].get("content", ""))
        # Fall back to /api/generate
        gen_url = config.OLLAMA_HOST.rstrip("/") + "/api/generate"
        gen_payload: dict = {
            "model": self.ollama_model,
            "prompt": (system_prompt + "\n\n" + prompt) if system_prompt else prompt,
            "stream": False,
            "options": {"temperature": float(temperature)},
        }
        if json_mode:
            gen_payload["format"] = "json"
        resp = requests.post(gen_url, json=gen_payload, timeout=self.timeout_seconds)
        resp.raise_for_status()
        data = resp.json()
        return str(data.get("response", ""))

    def _watsonx_generate(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float,
        response_format: ResponseFormat = "text",
    ) -> str:
        if not config.WATSONX_API_KEY or not config.WATSONX_PROJECT_ID:
            raise RuntimeError(
                "Set WATSONX_API_KEY and WATSONX_PROJECT_ID in .env to use watsonx."
            )
        # Imported lazily so the rest of the app runs without ibm-watsonx-ai installed.
        try:
            from ibm_watsonx_ai import Credentials  # type: ignore
            from ibm_watsonx_ai.foundation_models import ModelInference  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "ibm-watsonx-ai is not installed. pip install ibm-watsonx-ai"
            ) from exc

        creds = Credentials(url=config.WATSONX_URL, api_key=config.WATSONX_API_KEY)
        model = ModelInference(
            model_id=self.watsonx_model,
            credentials=creds,
            project_id=config.WATSONX_PROJECT_ID,
            params={
                "decoding_method": "greedy",
                "max_new_tokens": 1024,
                "temperature": float(temperature),
            },
        )

        # The watsonx SDK doesn't have a first-class structured-output mode
        # like Ollama. When the caller asks for JSON we append a strong hint
        # to the system prompt; the parse_json_block helper at call sites
        # already handles models that wrap their response in prose.
        effective_system = system_prompt
        if response_format == "json":
            json_hint = (
                "Respond with a single JSON object only. "
                "Do not include any explanation, markdown, or prose."
            )
            effective_system = (
                f"{system_prompt}\n\n{json_hint}".strip() if system_prompt else json_hint
            )

        full_prompt = (
            f"<|system|>\n{effective_system}\n<|user|>\n{prompt}\n<|assistant|>\n"
            if effective_system
            else prompt
        )
        response = model.generate(prompt=full_prompt)
        if isinstance(response, dict):
            results = response.get("results") or []
            if results:
                return str(results[0].get("generated_text", ""))
        # Some SDK versions return the raw string directly.
        return str(response)


# --- Mock implementation ------------------------------------------------------


def _mock_generate(prompt: str, system_prompt: str) -> str:
    """Return a deterministic placeholder response.

    The real Mock Demo Mode logic lives inside each agent (which inspects the
    user question directly). This function exists so callers that ask the
    LLM in mock mode still receive a plausible string instead of an error.
    """
    note = (
        "[Mock LLM] This response is generated locally without calling an external model. "
        "Switch the provider to Ollama or IBM watsonx to use a real LLM."
    )
    if system_prompt:
        return f"{note}\n\nSystem hint: {system_prompt[:140]}\nUser: {prompt[:140]}"
    return f"{note}\n\nUser: {prompt[:200]}"


# --- Convenience helpers ------------------------------------------------------


def _extract_json_snippet(text: str) -> Optional[str]:
    """Strip markdown fences and return the outer ``{...}`` block, if any."""
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return cleaned[start : end + 1]


def parse_json_block(text: str) -> Optional[dict]:
    """Best-effort JSON extraction from an LLM response."""
    snippet = _extract_json_snippet(text)
    if snippet is None:
        return None
    try:
        return json.loads(snippet)
    except Exception:
        return None
