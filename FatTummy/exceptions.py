class FatTummyBaseException(Exception):
    """Base exception class for FatTummy framework."""
    pass

class FatTummyOOMError(FatTummyBaseException):
    """Raised when CUDA out of memory or TPU HBM allocation fails."""
    def __init__(self, original_error=None):
        message = (
            "Hardware Memory Exhausted. "
            "Consider reducing the batch size or using '.param(\"quantize_4bit\")'.\n"
        )
        if original_error:
            message += f"Original error: {original_error}"
        super().__init__(message)

class FatTummyDriverError(FatTummyBaseException):
    """Raised when libtpu.so is missing or CUDA versions mismatch on runtime."""
    def __init__(self, hardware_target, original_error=None):
        message = f"Driver Error for target {hardware_target}. "
        if hardware_target == "tpu":
            message += "Ensure libtpu.so is accessible and the correct PyTorch XLA version is installed."
        elif hardware_target == "gpu":
            message += "Ensure CUDA drivers match the PyTorch version."
        if original_error:
            message += f"\nOriginal error: {original_error}"
        super().__init__(message)

class FatTummyNetworkError(FatTummyBaseException):
    """Catches HF Hub or Cloud API dropouts, dumps weights locally, and prints recovery notice."""
    def __init__(self, operation, model_path=None, original_error=None):
        message = f"Network connection lost during '{operation}'. "
        if model_path:
            message += f"\nModel weights have been backed up locally to {model_path}."
        if original_error:
            message += f"\nOriginal error: {original_error}"
        super().__init__(message)
