"""
llm.py

Provider-agnostic LLM client factory.

Supports:
- text-only calls:
    generate_content(prompt)

- optional image/vision calls:
    generate_content(prompt, images=[image_path])

Important:
- Text-only calls continue to work exactly like before.
- Image calls require a vision-capable model.
"""

from __future__ import annotations

import base64
import mimetypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import requests


@dataclass
class LLMResponse:
    text: str
    raw: Any | None = None


def image_path_to_data_url(image_path: str | Path) -> str:
    """
    Convert local image path to a base64 data URL.

    Used by Groq and Hugging Face OpenAI-compatible chat APIs.
    """
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    mime_type, _ = mimetypes.guess_type(str(path))
    mime_type = mime_type or "image/png"

    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def normalize_images(images: Optional[Sequence[str | Path]]) -> list[Path]:
    if not images:
        return []

    normalized: list[Path] = []

    for image in images:
        path = Path(image)

        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")

        normalized.append(path)

    return normalized


def extract_gemini_text(response: Any) -> str:
    """
    Safely extract text from Gemini response.
    """
    try:
        return response.text.strip()
    except Exception:
        pass

    try:
        parts = response.candidates[0].content.parts
        return "\n".join(
            str(getattr(part, "text", ""))
            for part in parts
            if getattr(part, "text", "")
        ).strip()
    except Exception:
        return str(response)


class BaseLLMClient:
    # Indicates whether this provider client can technically send image payloads.
    # Do NOT use this to decide whether a selected model supports vision.
    # Use config.MODEL_CAPABILITIES[model_name]["supports_vision"] instead.
    supports_vision: bool = False

    def generate_content(
        self,
        prompt: str,
        images: Optional[Sequence[str | Path]] = None,
    ) -> LLMResponse:
        raise NotImplementedError


class GeminiClient(BaseLLMClient):
    supports_vision = True

    def __init__(
        self,
        model_name: str,
        temperature: float,
        api_key: Optional[str] = None,
    ) -> None:
        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise ImportError(
                "Google Gemini support requires the 'google-generativeai' package."
            ) from exc

        api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

        if not api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY or GOOGLE_API_KEY not set in environment / .env"
            )

        genai.configure(api_key=api_key)

        self.model_name = model_name
        self.temperature = temperature
        self._model = genai.GenerativeModel(
            model_name=model_name,
            generation_config={"temperature": temperature},
        )

    def generate_content(
        self,
        prompt: str,
        images: Optional[Sequence[str | Path]] = None,
    ) -> LLMResponse:
        image_paths = normalize_images(images)

        if not image_paths:
            response = self._model.generate_content(prompt)
            return LLMResponse(text=extract_gemini_text(response), raw=response)

        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError(
                "Vision input requires Pillow. Install it with: pip install Pillow"
            ) from exc

        opened_images = []

        try:
            for image_path in image_paths:
                image = Image.open(image_path)
                image.load()
                opened_images.append(image)

            response = self._model.generate_content([prompt, *opened_images])
            return LLMResponse(text=extract_gemini_text(response), raw=response)

        finally:
            for image in opened_images:
                try:
                    image.close()
                except Exception:
                    pass


class GroqClient(BaseLLMClient):
    """
    Groq text + optional vision client.

    Image input works only with Groq vision-capable models.
    If you pass images to a text-only Groq model, Groq will reject the request.
    """

    supports_vision = True

    def __init__(
        self,
        model_name: str,
        temperature: float,
        api_key: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.api_key = api_key or os.getenv("GROQ_API_KEY") or os.getenv("GROQ_API_TOKEN")

        if not self.api_key:
            raise EnvironmentError("GROQ_API_KEY not set in environment / .env")

    def build_messages(
        self,
        prompt: str,
        images: Optional[Sequence[str | Path]] = None,
    ) -> list[dict[str, Any]]:
        image_paths = normalize_images(images)

        if not image_paths:
            return [{"role": "user", "content": prompt}]

        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]

        for image_path in image_paths:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_path_to_data_url(image_path),
                    },
                }
            )

        return [{"role": "user", "content": content}]

    def generate_content(
        self,
        prompt: str,
        images: Optional[Sequence[str | Path]] = None,
    ) -> LLMResponse:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model_name,
                "messages": self.build_messages(prompt, images),
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


class HuggingFaceClient(BaseLLMClient):
    """
    Hugging Face text + optional vision client.

    Image input works only if the selected HF model/provider supports
    multimodal chat-completion.
    """

    supports_vision = True

    def __init__(
        self,
        model_name: str,
        temperature: float,
        api_key: Optional[str] = None,
    ) -> None:
        try:
            from huggingface_hub import InferenceClient
        except ImportError as exc:
            raise ImportError("Run: pip install huggingface_hub") from exc

        self.temperature = temperature
        self.model_name = model_name

        token = api_key or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_TOKEN")

        if not token:
            raise EnvironmentError("HF_TOKEN not set in .env")

        self._client = InferenceClient(
            model=model_name,
            token=token,
        )

    def build_messages(
        self,
        prompt: str,
        images: Optional[Sequence[str | Path]] = None,
    ) -> list[dict[str, Any]]:
        image_paths = normalize_images(images)

        if not image_paths:
            return [{"role": "user", "content": prompt}]

        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]

        for image_path in image_paths:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_path_to_data_url(image_path),
                    },
                }
            )

        return [{"role": "user", "content": content}]

    def generate_content(
        self,
        prompt: str,
        images: Optional[Sequence[str | Path]] = None,
    ) -> LLMResponse:
        response = self._client.chat_completion(
            messages=self.build_messages(prompt, images),
            max_tokens=1024,
            temperature=self.temperature,
        )

        text = response.choices[0].message.content

        if isinstance(text, list):
            text = "\n".join(
                str(item.get("text", item))
                if isinstance(item, dict)
                else str(item)
                for item in text
            )

        return LLMResponse(text=str(text).strip(), raw=response)


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
        "huggingface": "huggingface",
        "hf": "huggingface",
    }

    if provider:
        return aliases.get(provider, provider)

    return "gemini"


def build_llm(
    model_name: str,
    temperature: float,
    provider: Optional[str] = None,
    **kwargs: Any,
) -> BaseLLMClient:
    resolved_provider = normalize_provider_name(provider, model_name)

    if resolved_provider == "gemini":
        return GeminiClient(
            model_name=model_name,
            temperature=temperature,
            api_key=kwargs.get("api_key"),
        )

    if resolved_provider == "groq":
        return GroqClient(
            model_name=model_name,
            temperature=temperature,
            api_key=kwargs.get("api_key"),
        )

    if resolved_provider == "huggingface":
        return HuggingFaceClient(
            model_name=model_name,
            temperature=temperature,
            api_key=kwargs.get("api_key"),
        )

    raise ValueError(f"Unsupported LLM provider: {resolved_provider}")


def get_llm_response_text(
    client: BaseLLMClient,
    prompt: str,
    images: Optional[Sequence[str | Path]] = None,
) -> str:
    return client.generate_content(prompt, images=images).text