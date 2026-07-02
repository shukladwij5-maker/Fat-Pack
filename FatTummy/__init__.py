# Global singleton engine instance
_default_engine = None
__version__ = "0.2.3"


def _restore_public_api():
    """Keep builder helpers available after Python attaches submodules to the package."""
    globals()["build"] = _build_func
    globals()["modelbuild"] = _modelbuild_func
    globals()["type"] = _type_func
    globals()["data"] = _data_func
    globals()["temp"] = _temp_func
    globals()["chat"] = _chat_func
    globals()["generate"] = _generate_func
    globals()["finetune"] = _finetune_func
    globals()["engine"] = _engine_func
    globals()["key"] = _key_func
    globals()["hf_login"] = _hf_login_func
    globals()["quantize"] = _quantize_func


def _ensure_engine():
    global _default_engine
    if _default_engine is None:
        from .engine import FatTummyEngine
        _default_engine = FatTummyEngine()
        _restore_public_api()
    return _default_engine


def _build_func(interactive=True):
    """Initialize the builder. Interactive mode launches the terminal wizard."""
    global _default_engine
    if interactive:
        from .interactive import run_wizard
        _default_engine = run_wizard()
        _restore_public_api()
        return _default_engine
    from .engine import FatTummyEngine
    _default_engine = FatTummyEngine()
    _restore_public_api()
    return _default_engine


def _modelbuild_func(scale: str):
    """Sets the model parameter scale globally."""
    return _ensure_engine().modelbuild(scale)


def _type_func(arch):
    """Sets the global model type and validates targets."""
    return _ensure_engine().type(arch)


def _data_func(*sources):
    """Ingests multiple data sources into the global engine."""
    return _ensure_engine().data(*sources)


def _temp_func(value: float):
    """Sets the temperature globally."""
    return _ensure_engine().temp(value)


def _chat_func():
    """Starts the terminal chat interface using the globally built engine."""
    return _ensure_engine().chat()


def _generate_func(prompt: str):
    """Generate text using the globally built engine."""
    return _ensure_engine().generate(prompt)


def _finetune_func(epochs: int = 3):
    """Starts finetuning using the globally configured settings."""
    return _ensure_engine().finetune(epochs)


def _engine_func(name: str):
    """Sets the underlying engine globally (hf, ollama, openai, etc)."""
    return _ensure_engine().engine(name)


def _key_func(api_key: str):
    """Sets API key globally."""
    return _ensure_engine().key(api_key)


def _hf_login_func(token: str):
    """Logs in to HuggingFace Hub globally."""
    return _ensure_engine().hf_login(token)


def _quantize_func(mode: str):
    """Sets the quantization mode globally."""
    return _ensure_engine().quantize(mode)


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
    "generate",
    "finetune",
    "engine",
    "key",
    "hf_login",
    "quantize",
    "MOOE",
    "__version__",
]

_restore_public_api()
