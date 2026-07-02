"""Interactive terminal wizard for ``ft.build()``."""

from __future__ import annotations

import sys
from typing import Any, Dict

from .data.loader import resolve_datasets

SAMPLES = {
    "scale": "tiny",
    "token_limit": "512",
    "type": "mooe",
    "hf_model": "gpt2",
    "temperature": "0.7",
    "epochs": "3",
    "engine": "mooe",
    "api_provider": "openai",
}

API_PROVIDERS = {
    "openai": "openai",
    "anthropic": "anthropic",
    "claude": "anthropic",
    "gemini": "gemini",
    "google": "gemini",
}

NATIVE_CHOICES = {"mooe", "lion", "spacebyte"}

LOGO_LINES = [
    "",
    "  ███████╗ █████╗ ████████╗   ████████╗██╗   ██╗███╗   ███╗███╗   ███╗██╗   ██╗",
    "  ██╔════╝██╔══██╗╚══██╔══╝   ╚══██╔══╝██║   ██║████╗ ████║████╗ ████║╚██╗ ██╔╝",
    "  █████╗  ███████║   ██║         ██║   ██║   ██║██╔████╔██║██╔████╔██║ ╚████╔╝ ",
    "  ██╔══╝  ██╔══██║   ██║         ██║   ██║   ██║██║╚██╔╝██║██║╚██╔╝██║  ╚██╔╝  ",
    "  ██║     ██║  ██║   ██║         ██║   ╚██████╔╝██║ ╚═╝ ██║██║ ╚═╝ ██║   ██║   ",
    "  ╚═╝     ╚═╝  ╚═╝   ╚═╝         ╚═╝    ╚═════╝ ╚═╝     ╚═╝╚═╝     ╚═╝   ╚═╝   ",
    "",
    "           ░▒▓█  B U I L D  ·  T R A I N  ·  C H A T  █▓▒░",
    "                  five words. infinite models.",
    "",
]

LOGO_FALLBACK = r"""
    ======================================================================
    |                                                                    |
    |   FAT TUMMY                                                        |
    |                                                                    |
    |            BUILD  -  TRAIN  -  CHAT                                |
    |            five words. infinite models.                            |
    |                                                                    |
    ======================================================================
"""


def _print_logo() -> None:
    """Print the FatTummy banner with an ASCII fallback for conservative terminals."""
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        print(_build_logo())
    except (UnicodeEncodeError, LookupError, OSError):
        print(LOGO_FALLBACK)


def _build_logo() -> str:
    """Build the original boxed Unicode logo with a frame that fits the text."""
    width = max(len(line) for line in LOGO_LINES)
    top = "    ╔" + "═" * (width + 2) + "╗"
    body = ["    ║ " + line.ljust(width) + " ║" for line in LOGO_LINES]
    bottom = "    ╚" + "═" * (width + 2) + "╝"
    return "\n".join(["", top, *body, bottom])


def collect_config() -> Dict[str, Any]:
    """Collect wizard configuration from terminal prompts."""
    _print_logo()
    print("FatTummy 0.2.3")
    print("  a = Adv     Full control")
    print("  b = Breeze  Minimal prompts")
    mode = input("Choose mode [b]: ").strip().lower()
    return _run_advanced() if mode in {"a", "adv", "advanced"} else _run_breeze()


def validate_config(config: Dict[str, Any]) -> None:
    """Validate wizard configuration before execution."""
    if config["action"] == "finetune" and not config.get("datasets_raw"):
        raise ValueError("Fine-tuning requires a dataset.")
    if config["action"] == "api" and not config.get("api_key"):
        raise ValueError("API chat requires an API key.")
    if config.get("engine") in NATIVE_CHOICES and str(config.get("type", "")).lower() not in NATIVE_CHOICES:
        config["type"] = config["engine"]


def execute_config(config: Dict[str, Any]):
    """Create an engine, apply config, and run the selected action."""
    api_only = config["action"] == "api"
    from .engine import FatTummyEngine

    engine = FatTummyEngine(api_only=api_only)
    _apply_config(engine, config)
    _run_action(engine, config)
    return engine


def run_wizard():
    """Collect, validate, and execute an interactive FatTummy configuration."""
    config = collect_config()
    validate_config(config)
    return execute_config(config)


def _prompt(label: str, default: str = "", required: bool = False) -> str:
    """Read a terminal value, using a default when allowed."""
    suffix = f" [{default}]" if default else ""
    while True:
        value = input(f"{label}{suffix}: ").strip()
        if value:
            return value
        if default:
            return default
        if not required:
            return ""
        print("This field is required.")


def _prompt_secret(label: str) -> str:
    """Read a secret value without echo when getpass is available."""
    try:
        import getpass

        return getpass.getpass(f"{label}: ").strip()
    except Exception:
        return input(f"{label}: ").strip()


def _choose_action() -> str:
    """Prompt for the high-level user action."""
    print("What do you want to do?")
    print("  1 = Make Model")
    print("  2 = Fine-tune")
    print("  3 = API Chat")
    choice = input("Choose [1]: ").strip().lower()
    if choice in {"2", "finetune", "fine-tune", "train"}:
        return "finetune"
    if choice in {"3", "api", "chat"}:
        return "api"
    return "make"


def _resolve_api_provider(raw: str) -> str:
    """Normalize API provider aliases."""
    return API_PROVIDERS.get(raw.lower(), raw.lower())


def _run_breeze() -> Dict[str, Any]:
    """Collect minimal prompts."""
    action = _choose_action()
    return _collect_for_action(action)


def _collect_for_action(action: str) -> Dict[str, Any]:
    """Collect prompts for a known action."""
    config: Dict[str, Any] = {
        "action": action,
        "scale": SAMPLES["scale"],
        "token_limit": SAMPLES["token_limit"],
        "temperature": SAMPLES["temperature"],
        "epochs": SAMPLES["epochs"],
        "hf_token": "",
        "datasets_raw": "",
        "dataset_modes": [],
        "datasets": [],
    }
    if action == "make":
        engine = _prompt("Native engine (mooe/lion/spacebyte)", SAMPLES["engine"])
        config["engine"] = engine.lower()
        config["type"] = engine.lower()
        config["datasets_raw"] = _prompt("Dataset (optional)", "")
    elif action == "finetune":
        config["engine"] = "hf"
        config["type"] = _prompt("HF model", SAMPLES["hf_model"])
        config["datasets_raw"] = _prompt("Dataset", required=True)
    else:
        config["engine"] = _resolve_api_provider(_prompt("API provider", SAMPLES["api_provider"]))
        config["api_key"] = _prompt_secret("API key")
        config["type"] = _prompt("Model name (optional)", "")
    return config


def _run_advanced() -> Dict[str, Any]:
    """Collect full-control prompts — every tunable knob is exposed."""
    action = _choose_action()
    config = _collect_for_action(action)
    config["hf_token"] = _prompt_secret("Hugging Face token (blank to skip)")
    config["scale"] = _prompt("Scale (tiny/small/medium)", SAMPLES["scale"])
    config["token_limit"] = _prompt("Token limit", SAMPLES["token_limit"])
    config["temperature"] = _prompt("Temperature", SAMPLES["temperature"])
    config["epochs"] = _prompt("Epochs", SAMPLES["epochs"])
    # Advanced-only knobs — these are missing from Breeze mode
    config["quantize"] = _prompt("Quantization (4bit/8bit/blank to skip)", "")
    config["timeout"] = _prompt("Request timeout in seconds", "120")
    return config


def _apply_config(engine: Any, config: Dict[str, Any]) -> None:
    """Apply collected values to an engine."""
    if config.get("hf_token"):
        engine.hf_login(config["hf_token"])
    engine.action(config["action"])
    engine.modelbuild(config["scale"])
    engine.token_limit(int(config["token_limit"]))
    engine.temp(float(config["temperature"]))
    engine.epochs(int(config["epochs"]))

    # Apply advanced-only settings when present
    if config.get("quantize"):
        engine.quantize(config["quantize"])
    if config.get("timeout"):
        engine.timeout(float(config["timeout"]))

    raw_datasets = config.get("datasets_raw", "")
    if raw_datasets:
        datasets, modes = resolve_datasets(raw_datasets, token=config.get("hf_token"))
        config["datasets"] = datasets
        config["dataset_modes"] = modes
        engine._dataset_modes = modes
        for dataset in datasets:
            engine.data(dataset)

    # BUG FIX: engine.engine() must be called BEFORE engine.type() so that
    # _compile_and_initialize() knows which backend to instantiate when type() triggers it.
    engine.engine(config["engine"])
    if config["action"] == "api":
        engine.key(config.get("api_key", ""))
        if config.get("type"):
            engine.type(config["type"])
    else:
        engine.type(config["type"])


def _run_action(engine: Any, config: Dict[str, Any]) -> None:
    """Run the selected action after configuration."""
    print(f"Configuration ready: action={config['action']} engine={config['engine']}")
    if config.get("dataset_modes"):
        print(f"Datasets loaded via: {', '.join(config['dataset_modes'])}")
    action = config["action"]
    if action == "finetune":
        engine.finetune()
        # After fine-tuning, drop into chat only if the user wants it
        print("Fine-tuning complete. Type 'exit' at the prompt to quit.")
        engine.chat()
    elif action in {"make", "api", "chat"}:
        # BUG FIX: Previously chat() was always called unconditionally even for
        # non-interactive actions (e.g. programmatic API calls).  Now it is
        # guarded so only the appropriate actions enter the chat loop.
        engine.chat()
