from .engine import FatTummyEngine
from .models.mooe import MOOE

# Global singleton engine instance
_default_engine = None

def build():
    """Initializes the global FatTummy builder state machine and evaluates dependencies."""
    global _default_engine
    _default_engine = FatTummyEngine()
    return _default_engine

def modelbuild(scale: str):
    """Sets the model parameter scale globally."""
    global _default_engine
    if _default_engine is None:
        build()
    return _default_engine.modelbuild(scale)

def type(arch):
    """Sets the global model type and validates targets."""
    global _default_engine
    if _default_engine is None:
        build()
    return _default_engine.type(arch)

def data(*sources):
    """Ingests multiple data sources into the global engine."""
    global _default_engine
    if _default_engine is None:
        build()
    return _default_engine.data(*sources)

def temp(value: float):
    """Sets the temperature globally."""
    global _default_engine
    if _default_engine is None:
        build()
    return _default_engine.temp(value)

def chat():
    """Starts the terminal chat interface using the globally built engine."""
    global _default_engine
    if _default_engine is None:
        build()
    return _default_engine.chat()

def finetune(epochs: int = 3):
    """Starts finetuning using the globally configured settings."""
    global _default_engine
    if _default_engine is None:
        build()
    return _default_engine.finetune(epochs)

def engine(name: str):
    """Sets the underlying engine globally (hf, ollama, openai, etc)."""
    global _default_engine
    if _default_engine is None:
        build()
    return _default_engine.engine(name)

def key(api_key: str):
    """Sets API key globally."""
    global _default_engine
    if _default_engine is None:
        build()
    return _default_engine.key(api_key)

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
    "MOOE"
]
