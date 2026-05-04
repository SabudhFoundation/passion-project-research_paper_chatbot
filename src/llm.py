"""Provider-agnostic LLM client factory and provider adapters."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

import requests


@dataclass
class LLMResponse:
    text: str
    raw: Any | None = None


class BaseLLMClient:
    def generate_content(self, prompt: str) -> LLMResponse:
        raise NotImplementedError


class GeminiClient(BaseLLMClient):
    def __init__(self, model_name: str, temperature: float, api_key: Optional[str] = None) -> None:
        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise ImportError(
                "Google Gemini support requires the 'google-generativeai' package."
            ) from exc

        api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY or GOOGLE_API_KEY not set in environment / .env")

        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(
            model_name=model_name,
            generation_config={"temperature": temperature},
        )

    def generate_content(self, prompt: str) -> LLMResponse:
        response = self._model.generate_content(prompt)
        return LLMResponse(text=getattr(response, "text", str(response)), raw=response)


class GroqClient(BaseLLMClient):
    def __init__(self, model_name: str, temperature: float, api_key: Optional[str] = None) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.api_key = api_key or os.getenv("GROQ_API_KEY") or os.getenv("GROQ_API_TOKEN")
        if not self.api_key:
            raise EnvironmentError("GROQ_API_KEY not set in environment / .env")

    def generate_content(self, prompt: str) -> LLMResponse:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
            },
            timeout=120,
        )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            detail = None
            try:
                payload = response.json()
                if isinstance(payload, dict):
                    detail = payload.get("error") or payload.get("message")
            except ValueError:
                detail = response.text.strip() or None

            message = f"Groq API request failed with status {response.status_code}"
            if detail:
                message = f"{message}: {detail}"
            raise RuntimeError(message) from exc

        payload = response.json()
        if isinstance(payload, dict) and payload.get("error"):
            raise RuntimeError(f"Groq API error: {payload['error']}")

        try:
            text = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected Groq response format: {payload!r}") from exc

        return LLMResponse(text=text, raw=payload)


def normalize_provider_name(provider: Optional[str], model_name: str) -> str:
    if provider:
        provider = provider.strip().lower()
    else:
        provider = os.getenv("LLM_PROVIDER", "").strip().lower()

    if not provider and ":" in model_name:
        provider, model_name = model_name.split(":", 1)

    aliases = {
        "google": "gemini",
        "gemini": "gemini",
        "groq": "groq",
    }

    if provider:
        return aliases.get(provider, provider)

    return "gemini"


def build_llm(model_name: str, temperature: float, provider: Optional[str] = None, **kwargs: Any) -> BaseLLMClient:
    resolved_provider = normalize_provider_name(provider, model_name)
    model_name = model_name.split(":", 1)[1] if ":" in model_name and resolved_provider != "gemini" else model_name

    if resolved_provider == "gemini":
        return GeminiClient(model_name=model_name, temperature=temperature, api_key=kwargs.get("api_key"))
    if resolved_provider == "groq":
        return GroqClient(model_name=model_name, temperature=temperature, api_key=kwargs.get("api_key"))

    raise ValueError(f"Unsupported LLM provider: {resolved_provider}")


def get_llm_response_text(client: BaseLLMClient, prompt: str) -> str:
    return client.generate_content(prompt).text