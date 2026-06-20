# Global singleton engine instance
_default_engine = None


def _ensure_engine():
    global _default_engine
    if _default_engine is None:
        from .engine import FatTummyEngine
        _default_engine = FatTummyEngine()
    return _default_engine


def build(interactive=True):
    """Initialize the builder. Interactive mode launches the terminal wizard."""
    global _default_engine
    if interactive:
        from .interactive import run_wizard
        _default_engine = run_wizard()
        return _default_engine
    from .engine import FatTummyEngine
    _default_engine = FatTummyEngine()
    return _default_engine


def modelbuild(scale: str):
    """Sets the model parameter scale globally."""
    return _ensure_engine().modelbuild(scale)


def type(arch):
    """Sets the global model type and validates targets."""
    return _ensure_engine().type(arch)


def data(*sources):
    """Ingests multiple data sources into the global engine."""
    return _ensure_engine().data(*sources)


def temp(value: float):
    """Sets the temperature globally."""
    return _ensure_engine().temp(value)


def chat():
    """Starts the terminal chat interface using the globally built engine."""
    return _ensure_engine().chat()


def finetune(epochs: int = 3):
    """Starts finetuning using the globally configured settings."""
    return _ensure_engine().finetune(epochs)


def engine(name: str):
    """Sets the underlying engine globally (hf, ollama, openai, etc)."""
    return _ensure_engine().engine(name)


def key(api_key: str):
    """Sets API key globally."""
    return _ensure_engine().key(api_key)


def hf_login(token: str):
    """Logs in to HuggingFace Hub globally."""
    return _ensure_engine().hf_login(token)


def __getattr__(name):
    if name == "MOOE":
        from .models.mooe import MOOE
        return MOOE
    raise AttributeError(f"module 'FatTummy' has no attribute {name!r}")


# Expose constants and global functions
__all__ = [
    "build",
    "modelbuild",
    "type",
    "data",
    "temp",
    "chat",
    "finetune",
    "engine",
    "key",
    "hf_login",
    "MOOE",
]
