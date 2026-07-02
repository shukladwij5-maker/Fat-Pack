"""Exception hierarchy for FatTummy."""

from __future__ import annotations

from typing import Optional


class FatTummyBaseException(Exception):
    """Base exception class for all FatTummy framework errors."""


class FatTummyConfigurationError(FatTummyBaseException):
    """Raised when the builder state is incomplete or internally inconsistent."""


class FatTummyDatasetError(FatTummyBaseException):
    """Raised when a dataset source cannot be validated or loaded."""


class FatTummyAuthenticationError(FatTummyBaseException):
    """Raised when a provider requires a missing or invalid credential."""


class FatTummyDependencyError(FatTummyBaseException):
    """Raised when an optional backend dependency is not installed."""


class FatTummyUnsupportedBackendError(FatTummyBaseException):
    """Raised when a requested engine/model combination is not supported."""


class FatTummyOOMError(FatTummyBaseException):
    """Raised when CUDA, CPU, or TPU memory is exhausted."""

    def __init__(self, original_error: Optional[object] = None) -> None:
        message = (
            "Hardware memory was exhausted. Reduce batch size, sequence length, "
            "or model size before retrying."
        )
        if original_error:
            message += f"\nOriginal error: {original_error}"
        super().__init__(message)


class FatTummyDriverError(FatTummyBaseException):
    """Raised when hardware driver/runtime support is unavailable."""

    def __init__(self, hardware_target: str, original_error: Optional[object] = None) -> None:
        message = f"Driver error for target '{hardware_target}'. "
        if hardware_target == "tpu":
            message += "Install a compatible torch_xla/libtpu runtime."
        elif hardware_target == "gpu":
            message += "Install CUDA drivers that match the PyTorch build."
        else:
            message += "Check that the requested hardware runtime is available."
        if original_error:
            message += f"\nOriginal error: {original_error}"
        super().__init__(message)


class FatTummyNetworkError(FatTummyBaseException):
    """Raised for Hugging Face, Ollama, or cloud provider network failures."""

    def __init__(
        self,
        operation: str,
        model_path: Optional[str] = None,
        original_error: Optional[object] = None,
    ) -> None:
        message = f"Network or provider failure during '{operation}'."
        if model_path:
            message += f"\nLocal recovery path: {model_path}"
        if original_error:
            message += f"\nOriginal error: {original_error}"
        super().__init__(message)
