"""Cloud inference adapters for OpenAI, Anthropic, and Gemini."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..exceptions import (
    FatTummyAuthenticationError,
    FatTummyDependencyError,
    FatTummyNetworkError,
    FatTummyUnsupportedBackendError,
)


@dataclass
class BaseCloudAdapter:
    """Base class for cloud text generation adapters."""

    api_key: str
    model_name: Optional[str] = None
    timeout: float = 60.0
    max_tokens: int = 1024

    def __post_init__(self) -> None:
        if not self.api_key:
            raise FatTummyAuthenticationError(f"API key is required for {self.__class__.__name__}.")

    def generate(self, prompt: str) -> str:
        """Generate text from a prompt."""
        raise NotImplementedError

    def _execute_with_retry(self, call_fn, max_retries: int = 3, initial_backoff: float = 1.0):
        """Execute a callable with exponential backoff on exceptions."""
        import time
        backoff = initial_backoff
        for attempt in range(max_retries):
            try:
                return call_fn()
            except Exception as exc:
                if attempt == max_retries - 1:
                    raise
                print(f"FatTummy: Retrying cloud call due to: {exc}. Attempt {attempt + 1}/{max_retries}")
                time.sleep(backoff)
                backoff *= 2


class OpenAIAdapter(BaseCloudAdapter):
    """OpenAI chat completions adapter."""

    def __post_init__(self) -> None:
        super().__post_init__()
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise FatTummyDependencyError("OpenAI support requires: pip install openai") from exc
        self.client = OpenAI(api_key=self.api_key, timeout=self.timeout)
        self.model_name = self.model_name or "gpt-4o-mini"

    def generate(self, prompt: str) -> str:
        """Generate a response using OpenAI with retries."""
        def make_call():
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens,
            )
            content = response.choices[0].message.content
            return content or ""

        try:
            return self._execute_with_retry(make_call)
        except Exception as exc:
            raise FatTummyNetworkError("openai_generate", original_error=exc) from exc


class AnthropicAdapter(BaseCloudAdapter):
    """Anthropic Messages API adapter."""

    def __post_init__(self) -> None:
        super().__post_init__()
        try:
            import anthropic
        except ImportError as exc:
            raise FatTummyDependencyError("Anthropic support requires: pip install anthropic") from exc
        self.client = anthropic.Anthropic(api_key=self.api_key, timeout=self.timeout)
        self.model_name = self.model_name or "claude-3-5-haiku-latest"

    def generate(self, prompt: str) -> str:
        """Generate a response using Anthropic with retries."""
        def make_call():
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            parts = [getattr(part, "text", "") for part in response.content]
            return "".join(parts)

        try:
            return self._execute_with_retry(make_call)
        except Exception as exc:
            raise FatTummyNetworkError("anthropic_generate", original_error=exc) from exc


class GeminiAdapter(BaseCloudAdapter):
    """Google Gemini generate-content adapter."""

    def __post_init__(self) -> None:
        super().__post_init__()
        try:
            from google import genai
        except ImportError as exc:
            raise FatTummyDependencyError("Gemini support requires: pip install google-genai") from exc
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = self.model_name or "gemini-1.5-flash"

    def generate(self, prompt: str) -> str:
        """Generate a response using Gemini with retries."""
        def make_call():
            response = self.client.models.generate_content(model=self.model_name, contents=prompt)
            return getattr(response, "text", "") or ""

        try:
            return self._execute_with_retry(make_call)
        except Exception as exc:
            raise FatTummyNetworkError("gemini_generate", original_error=exc) from exc


def get_cloud_adapter(
    engine_name: str,
    api_key: str,
    model_name: Optional[str] = None,
    timeout: float = 60.0,
) -> BaseCloudAdapter:
    """Create a cloud adapter for a supported provider."""
    normalized = engine_name.lower().strip()
    adapters = {
        "openai": OpenAIAdapter,
        "anthropic": AnthropicAdapter,
        "claude": AnthropicAdapter,
        "gemini": GeminiAdapter,
        "google": GeminiAdapter,
    }
    adapter_cls = adapters.get(normalized)
    if adapter_cls is None:
        raise FatTummyUnsupportedBackendError(f"Unknown cloud engine '{engine_name}'.")
    return adapter_cls(api_key=api_key, model_name=model_name, timeout=timeout)
