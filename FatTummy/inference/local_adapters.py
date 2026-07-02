"""Local inference adapters for Hugging Face Transformers and Ollama."""

from __future__ import annotations

import json
import shutil
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Optional

from ..exceptions import (
    FatTummyDependencyError,
    FatTummyNetworkError,
    FatTummyOOMError,
    FatTummyUnsupportedBackendError,
)


@dataclass
class LocalAdapterBase:
    """Base class for local generation backends."""

    model_name: str
    timeout: float = 120.0

    def generate(self, prompt: str) -> str:
        """Generate text from a prompt."""
        raise NotImplementedError


class HuggingFaceAdapter(LocalAdapterBase):
    """Lazy Hugging Face Transformers causal language-model adapter."""

    def __init__(self, model_name: str, token: Optional[str] = None, timeout: float = 120.0, quantization: Optional[str] = None) -> None:
        super().__init__(model_name=model_name, timeout=timeout)
        self.token = token
        self.tokenizer: Any = None
        self.model: Any = None
        self.quantization = quantization

    def _load(self) -> None:
        """Load tokenizer and model only when inference or training needs them."""
        if self.model is not None and self.tokenizer is not None:
            return
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise FatTummyDependencyError(
                "Hugging Face inference requires: pip install transformers torch"
            ) from exc
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=self.token)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            kwargs = {
                "device_map": "auto",
                "torch_dtype": "auto",
                "token": self.token,
            }
            if self.quantization in {"4bit", "4-bit"}:
                try:
                    from transformers import BitsAndBytesConfig
                    kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype="float16",
                    )
                except ImportError:
                    print("FatTummy: bitsandbytes is required for 4-bit quantization. Loading standard precision instead.")
            elif self.quantization in {"8bit", "8-bit"}:
                try:
                    from transformers import BitsAndBytesConfig
                    kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_8bit=True,
                    )
                except ImportError:
                    print("FatTummy: bitsandbytes is required for 8-bit quantization. Loading standard precision instead.")

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **kwargs
            )
            self.model.eval()
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                raise FatTummyOOMError(original_error=exc) from exc
            raise
        except Exception as exc:
            raise FatTummyNetworkError("huggingface_model_load", original_error=exc) from exc

    def generate(self, prompt: str) -> str:
        """Generate text with safe defaults."""
        self._load()
        try:
            import torch
        except ImportError as exc:
            raise FatTummyDependencyError("Hugging Face inference requires: pip install torch") from exc

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            device = getattr(self.model, "device", None)
            if device is not None:
                inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.95,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                raise FatTummyOOMError(original_error=exc) from exc
            raise


class OllamaAdapter(LocalAdapterBase):
    """Ollama local HTTP adapter with model availability checks."""

    def __init__(self, model_name: str, timeout: float = 120.0, auto_pull: bool = True) -> None:
        super().__init__(model_name=model_name, timeout=timeout)
        self.auto_pull = auto_pull
        self._verify_installation()
        if auto_pull:
            self._ensure_model_pulled()

    def _verify_installation(self) -> None:
        """Ensure the ollama CLI is available."""
        if shutil.which("ollama") is None:
            raise FatTummyDependencyError(
                "Ollama is not installed or not on PATH. Install it from https://ollama.com/download."
            )

    def _run_ollama(self, args: list[str]) -> str:
        """Run an ollama command with timeout handling."""
        try:
            completed = subprocess.run(
                ["ollama", *args],
                check=True,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            return completed.stdout
        except subprocess.TimeoutExpired as exc:
            raise FatTummyNetworkError("ollama_cli_timeout", original_error=exc) from exc
        except subprocess.CalledProcessError as exc:
            raise FatTummyNetworkError("ollama_cli", original_error=exc.stderr or exc) from exc

    def _ensure_model_pulled(self) -> None:
        """Pull the configured model when it is missing locally."""
        listing = self._run_ollama(["list"])
        if self.model_name not in listing:
            self._run_ollama(["pull", self.model_name])

    def generate(self, prompt: str) -> str:
        """Generate text through Ollama's local HTTP API."""
        payload = json.dumps({"model": self.model_name, "prompt": prompt, "stream": False}).encode("utf-8")
        request = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
            return result.get("response", "")
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise FatTummyNetworkError("ollama_generate", original_error=exc) from exc


def get_local_adapter(
    engine_name: str,
    model_name: str,
    token: Optional[str] = None,
    timeout: float = 120.0,
    quantization: Optional[str] = None,
) -> LocalAdapterBase:
    """Create a local adapter for a supported backend."""
    normalized = engine_name.lower().strip()
    if normalized == "hf":
        return HuggingFaceAdapter(model_name=model_name, token=token, timeout=timeout, quantization=quantization)
    if normalized == "ollama":
        return OllamaAdapter(model_name=model_name, timeout=timeout)
    raise FatTummyUnsupportedBackendError(f"Unknown local engine '{engine_name}'.")
