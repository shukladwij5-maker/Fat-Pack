import sys

# Global singleton engine instance
_default_engine = None
__version__ = "0.3.1"

# Make the package available under common lowercase and mixed-case aliases.
sys.modules.setdefault("fattummy", sys.modules[__name__])
sys.modules.setdefault("Fattummy", sys.modules[__name__])


def _restore_public_api():
    """Keep builder helpers available after Python attaches submodules to the package."""
    globals()["build"] = _build_func
    globals()["modelbuild"] = _modelbuild_func
    globals()["type"] = _type_func
    globals()["data"] = _data_func
    globals()["temp"] = _temp_func
    globals()["chat"] = _chat_func
    globals()["generate"] = _generate_func
    globals()["predict"] = _predict_func
    globals()["predict_csv"] = _predict_csv_func
    globals()["finetune"] = _finetune_func
    globals()["engine"] = _engine_func
    globals()["key"] = _key_func
    globals()["hf_login"] = _hf_login_func
    globals()["quantize"] = _quantize_func
    globals()["optimizer"] = _optimizer_func
    globals()["spacebyte"] = _spacebyte_func
    globals()["lr_scheduler"] = _lr_scheduler_func
    globals()["weight_decay"] = _weight_decay_func
    globals()["warmup"] = _warmup_func
    globals()["clip_grad"] = _clip_grad_func


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


def _predict_func(values, steps: int = 1):
    """Predict future values from a numeric sequence using an adaptive forecasting strategy."""
    from .predictor import predict
    return predict(values, steps=steps)


def _predict_csv_func(csv_path: str, target_column=None, steps: int = 1, model: str = "auto", date_column=None):
    """Predict future values from a CSV file using the forecasting helper."""
    from .predictor import predict_csv
    return predict_csv(csv_path, target_column=target_column, steps=steps, model=model, date_column=date_column)


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


def _optimizer_func(name: str):
    """Sets the training optimizer globally."""
    return _ensure_engine().optimizer(name)


def _spacebyte_func(enabled: bool = True):
    """Enables or disables SpaceByte raw byte tokenisation globally."""
    return _ensure_engine().spacebyte(enabled)


def _lr_scheduler_func(name: str):
    """Sets the learning rate scheduler globally."""
    return _ensure_engine().lr_scheduler(name)


def _weight_decay_func(value: float):
    """Sets the weight decay coefficient globally."""
    return _ensure_engine().weight_decay(value)


def _warmup_func(steps: int):
    """Sets the number of learning rate warmup steps globally."""
    return _ensure_engine().warmup(steps)


def _clip_grad_func(max_norm: float):
    """Sets gradient clipping norm globally."""
    return _ensure_engine().clip_grad(max_norm)


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
    "optimizer",
    "spacebyte",
    "lr_scheduler",
    "weight_decay",
    "warmup",
    "clip_grad",
    "MOOE",
    "__version__",
]

_restore_public_api()
