"""Builder-style runtime engine for FatTummy."""

from __future__ import annotations

import os
from typing import Any, List, Optional, Sequence

from .exceptions import (
    FatTummyAuthenticationError,
    FatTummyConfigurationError,
    FatTummyDatasetError,
    FatTummyDependencyError,
    FatTummyUnsupportedBackendError,
)
from .inference.cloud_adapters import get_cloud_adapter
from .inference.local_adapters import HuggingFaceAdapter, get_local_adapter
from .installer import ensure_installed
from .data import resolve_datasets
from .data.loader import _normalize_hf_token
from .predictor import predict
from .tuning.trainer import FatTummyTrainer

CLOUD_ENGINES = {"openai", "anthropic", "claude", "gemini", "google"}
LOCAL_ENGINES = {"hf", "ollama"}
NATIVE_ENGINES = {"mooe", "lion", "spacebyte"}
SUPPORTED_ENGINES = CLOUD_ENGINES | LOCAL_ENGINES | NATIVE_ENGINES


class FatTummyEngine:
    """Mutable builder that coordinates datasets, backends, inference, and tuning."""

    def __init__(self, api_only: bool = False) -> None:
        self._engine_name: Optional[str] = None
        self._param: Optional[str] = None
        self._data_sources: List[Any] = []
        self._model_type: Any = None
        self._api_key: Optional[str] = None
        self._hf_token: Optional[str] = None
        self._action: Optional[str] = None
        self._dataset_modes: List[str] = []
        self._temperature = 1.0
        self._token_limit = 512
        self._epochs = 3
        self._timeout = 120.0
        self._quantization: Optional[str] = None

        # Advanced training knobs
        self._optimizer: str = "adamw"
        self._use_spacebyte: bool = False
        self._lr_scheduler: str = "none"
        self._weight_decay: float = 0.01
        self._warmup_steps: int = 0
        self._clip_grad_norm: Optional[float] = None

        self._compiled = False
        self._adapter: Any = None
        self._model_instance: Any = None

        ensure_installed(api_only=api_only)

    def engine(self, name: str) -> "FatTummyEngine":
        """Select an engine: hf, ollama, openai, anthropic, gemini, mooe, lion, or spacebyte."""
        normalized = name.lower().strip()
        aliases = {"huggingface": "hf", "claude": "anthropic", "google": "gemini"}
        normalized = aliases.get(normalized, normalized)
        if normalized not in SUPPORTED_ENGINES:
            supported = ", ".join(sorted(SUPPORTED_ENGINES))
            raise FatTummyUnsupportedBackendError(f"Unknown engine '{name}'. Supported engines: {supported}.")
        self._engine_name = normalized
        self._compiled = False
        return self

    def modelbuild(self, scale: str) -> "FatTummyEngine":
        """Set the native model scale hint, for example 'tiny', 'small', or 'medium'."""
        self._param = scale
        self._compiled = False
        return self

    def data(self, *sources: Any) -> "FatTummyEngine":
        """Register one or more dataset objects, raw text values, or dataset source identifiers."""
        for source in sources:
            if isinstance(source, (str, os.PathLike)):
                # Resolve local dataset files and Hugging Face dataset repo IDs when possible.
                try:
                    datasets, modes = resolve_datasets(str(source), token=self._hf_token)
                except FatTummyDatasetError:
                    self._data_sources.append(source)
                else:
                    self._dataset_modes.extend(modes)
                    self._data_sources.extend(datasets)
                continue

            if isinstance(source, Sequence) and not isinstance(source, (str, bytes, os.PathLike)):
                for item in source:
                    self.data(item)
                continue

            self._data_sources.append(source)
        return self

    def type(self, arch: Any) -> "FatTummyEngine":
        """Set the model architecture/name and initialize lightweight backend state."""
        self._model_type = arch
        self._compiled = False
        self._compile_and_initialize()
        return self

    def key(self, api_key: str) -> "FatTummyEngine":
        """Set API key for cloud engines."""
        self._api_key = api_key
        self._adapter = None
        return self

    def hf_login(self, token: str) -> "FatTummyEngine":
        """Store a Hugging Face token and attempt a non-fatal hub login."""
        normalized_token = _normalize_hf_token(token)
        self._hf_token = normalized_token
        if normalized_token:
            try:
                from huggingface_hub import login
            except ImportError:
                print("FatTummy: huggingface_hub is not installed; token stored for later use.")
            else:
                login(token=normalized_token, add_to_git_credential=False)
                print("FatTummy: Hugging Face login successful.")
        return self

    def action(self, name: str) -> "FatTummyEngine":
        """Set the user goal: make, finetune, api, chat, or predict."""
        normalized = name.lower().strip()
        if normalized not in {"make", "finetune", "api", "chat", "predict"}:
            raise FatTummyConfigurationError("Action must be one of: make, finetune, api, chat, predict.")
        self._action = normalized
        return self

    def temp(self, value: float) -> "FatTummyEngine":
        """Set generation temperature."""
        if value <= 0:
            raise FatTummyConfigurationError("Temperature must be positive.")
        self._temperature = float(value)
        return self

    def token_limit(self, value: int) -> "FatTummyEngine":
        """Set maximum generated token count."""
        if value <= 0:
            raise FatTummyConfigurationError("Token limit must be positive.")
        self._token_limit = int(value)
        return self

    def timeout(self, value: float) -> "FatTummyEngine":
        """Set backend timeout in seconds."""
        if value <= 0:
            raise FatTummyConfigurationError("Timeout must be positive.")
        self._timeout = float(value)
        return self

    def epochs(self, value: int) -> "FatTummyEngine":
        """Set the default epoch count for fine-tuning."""
        if value <= 0:
            raise FatTummyConfigurationError("Epochs must be positive.")
        self._epochs = int(value)
        return self

    def quantize(self, mode: str) -> "FatTummyEngine":
        """Set the quantization mode for local Hugging Face models (e.g. '4bit' or '8bit')."""
        self._quantization = mode
        self._compiled = False
        return self

    def optimizer(self, name: str) -> "FatTummyEngine":
        """Choose the training optimizer: 'adamw' (default) or 'lion' (auto-installed)."""
        normalized = name.lower().strip()
        if normalized not in {"adamw", "lion"}:
            from .exceptions import FatTummyConfigurationError
            raise FatTummyConfigurationError("optimizer must be 'adamw' or 'lion'.")
        self._optimizer = normalized
        return self

    def spacebyte(self, enabled: bool = True) -> "FatTummyEngine":
        """Enable SpaceByte raw UTF-8 byte tokenisation (no BPE, vocab_size=256)."""
        self._use_spacebyte = bool(enabled)
        return self

    def lr_scheduler(self, name: str) -> "FatTummyEngine":
        """Set LR scheduler: 'cosine', 'linear', or 'none' (default)."""
        normalized = name.lower().strip()
        if normalized not in {"cosine", "linear", "none"}:
            from .exceptions import FatTummyConfigurationError
            raise FatTummyConfigurationError("lr_scheduler must be 'cosine', 'linear', or 'none'.")
        self._lr_scheduler = normalized
        return self

    def weight_decay(self, value: float) -> "FatTummyEngine":
        """Set L2 weight decay coefficient (default 0.01)."""
        self._weight_decay = float(value)
        return self

    def warmup(self, steps: int) -> "FatTummyEngine":
        """Set the number of linear warmup optimiser steps (default 0)."""
        self._warmup_steps = max(0, int(steps))
        return self

    def clip_grad(self, max_norm: float) -> "FatTummyEngine":
        """Clip gradient norm during training (e.g. 1.0). Pass 0 to disable."""
        self._clip_grad_norm = float(max_norm) if max_norm > 0 else None
        return self

    def _default_engine(self) -> str:
        """Infer a reasonable engine when the user omitted one."""
        if self._engine_name:
            return self._engine_name

        model_type_str = ""
        if isinstance(self._model_type, str):
            model_type_str = self._model_type.lower()
        elif self._model_type is not None:
            model_type_str = getattr(self._model_type, "__name__", "").lower()

        if model_type_str in {"mooe", "lion", "spacebyte"}:
            return model_type_str
        return "hf"

    def _compile_and_initialize(self) -> None:
        """Initialize lightweight backend state without downloading large assets."""
        if self._compiled:
            return
        engine_name = self._default_engine()
        self._engine_name = engine_name
        self._validate_combination(engine_name, self._model_type)

        if engine_name in LOCAL_ENGINES:
            if not isinstance(self._model_type, str):
                raise FatTummyConfigurationError(f"Engine '{engine_name}' requires a string model name.")
            self._adapter = get_local_adapter(
                engine_name,
                self._model_type,
                token=self._hf_token,
                timeout=self._timeout,
                quantization=self._quantization,
            )
        elif engine_name in NATIVE_ENGINES:
            self._model_instance = self._build_native_model(engine_name)
        else:
            self._adapter = None
        self._compiled = True

    def _validate_combination(self, engine_name: str, model_type: Any) -> None:
        """Reject invalid engine/model combinations with a user-facing error."""
        if engine_name in CLOUD_ENGINES and not (model_type is None or isinstance(model_type, str)):
            raise FatTummyUnsupportedBackendError("Cloud engines require a provider model-name string.")
        if engine_name == "ollama" and (not isinstance(model_type, str) or "/" in model_type):
            raise FatTummyUnsupportedBackendError("Ollama models should use local Ollama names such as 'llama3.2'.")
        if engine_name == "hf" and not isinstance(model_type, str):
            raise FatTummyUnsupportedBackendError("Hugging Face engine requires a model repo/name string.")

    def _build_native_model(self, engine_name: str) -> Any:
        """Instantiate a native experimental model variant."""
        try:
            from .models.mooe import MOOE, MOOEConfig
        except ImportError as exc:
            raise FatTummyDependencyError("Native models require PyTorch. Install it with: pip install torch") from exc

        scale = (self._param or "tiny").lower()
        legacy_large_aliases = {"1b", "8b", "10b"}
        if any(alias in scale for alias in legacy_large_aliases):
            print(
                "FatTummy: large scale labels are disabled in the prototype; "
                "using the CPU-friendly 'small' preset."
            )
            scale = "small"
        configs = {
            "tiny": dict(hidden_size=128, intermediate_size=512, num_layers=2, num_experts=4, top_k=2),
            "small": dict(hidden_size=256, intermediate_size=1024, num_layers=4, num_experts=4, top_k=2),
            "medium": dict(hidden_size=384, intermediate_size=1536, num_layers=6, num_experts=6, top_k=2),
        }
        selected = next((value for key, value in configs.items() if key in scale), configs["tiny"])
        if engine_name == "lion":
            selected = {**selected, "num_experts": max(2, selected["num_experts"] // 2), "top_k": 1}
        elif engine_name == "spacebyte":
            selected = {**selected, "vocab_size": 256}
        return MOOE(MOOEConfig(**selected))

    def predict(self, values: Sequence[float], steps: int = 1) -> List[float]:
        """Predict future values from a numeric time series using the adaptive forecasting stack."""
        if steps < 0:
            raise FatTummyConfigurationError("steps must be non-negative.")
        return predict(list(values), steps=steps)

    def generate(self, prompt: str) -> str:
        """Generate text through the configured backend."""
        self._compile_and_initialize()
        if self._engine_name in CLOUD_ENGINES:
            if not self._api_key:
                raise FatTummyAuthenticationError(f"API key required for '{self._engine_name}'. Use ft.key(...).")
            if self._adapter is None:
                self._adapter = get_cloud_adapter(
                    self._engine_name,
                    self._api_key,
                    model_name=self._model_type if isinstance(self._model_type, str) else None,
                    timeout=self._timeout,
                )
            return self._adapter.generate(prompt)
        if self._adapter is not None:
            return self._adapter.generate(prompt)
        return self._generate_native(prompt)

    def _generate_native(self, prompt: str) -> str:
        """Generate with the lightweight native model using byte-level fallback tokenization."""
        if self._model_instance is None:
            raise FatTummyConfigurationError("No native model is initialized.")
        try:
            import torch
        except ImportError as exc:
            raise FatTummyDependencyError("Native generation requires PyTorch. Install it with: pip install torch") from exc
        vocab = self._model_instance.config.vocab_size
        token_ids = [ord(char) % vocab for char in prompt] or [0]
        input_ids = torch.tensor([token_ids], dtype=torch.long)
        output = self._model_instance.generate(input_ids, max_new_tokens=min(self._token_limit, 32))[0].tolist()
        return "".join(chr(token % 256) for token in output)

    def chat(self) -> None:
        """Start an interactive terminal chat session."""
        print(
            "FatTummy Chat session started. Type 'exit' to quit. "
            f"[Engine: {self._engine_name or self._default_engine()} | Temp: {self._temperature}]"
        )
        self._compile_and_initialize()
        while True:
            try:
                user_input = input("You: ")
                if user_input.lower() in {"exit", "quit"}:
                    break
                print(f"FatTummy: {self.generate(user_input)}")
            except (KeyboardInterrupt, EOFError):
                break
            except Exception as exc:
                print(f"FatTummy Error: {exc}")

    def finetune(self, epochs: Optional[int] = None) -> None:
        """Fine-tune a local Hugging Face or native model."""
        if epochs is None:
            epochs = self._epochs
        self._compile_and_initialize()
        if not self._data_sources:
            raise FatTummyConfigurationError("Fine-tuning requires a dataset. Use ft.data(...).")
        dataset = self._data_sources[0] if len(self._data_sources) == 1 else self._data_sources

        if isinstance(self._adapter, HuggingFaceAdapter):
            self._adapter._load()
            trainer = FatTummyTrainer(
                self._adapter.model,
                dataset,
                tokenizer=self._adapter.tokenizer,
                epochs=epochs,
                optimizer=self._optimizer,
                use_spacebyte=self._use_spacebyte,
                lr_scheduler=self._lr_scheduler,
                weight_decay=self._weight_decay,
                warmup_steps=self._warmup_steps,
                clip_grad_norm=self._clip_grad_norm,
            )
        elif self._model_instance is not None:
            trainer = FatTummyTrainer(
                self._model_instance,
                dataset,
                epochs=epochs,
                optimizer=self._optimizer,
                use_spacebyte=self._use_spacebyte,
                lr_scheduler=self._lr_scheduler,
                weight_decay=self._weight_decay,
                warmup_steps=self._warmup_steps,
                clip_grad_norm=self._clip_grad_norm,
            )
        else:
            raise FatTummyUnsupportedBackendError("Fine-tuning requires local hf or a native model.")
        trainer.finetune()

    def push_to_hub(self, repo_id: str) -> None:
        """Push a Hugging Face-compatible local model to the Hub."""
        self._compile_and_initialize()
        target = self._model_instance
        if isinstance(self._adapter, HuggingFaceAdapter):
            self._adapter._load()
            target = self._adapter.model
        if target is None or not hasattr(target, "push_to_hub"):
            raise FatTummyUnsupportedBackendError("This model cannot be pushed to the Hugging Face Hub.")
        target.push_to_hub(repo_id)
