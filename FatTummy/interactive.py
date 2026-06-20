"""Interactive terminal wizard for ft.build()."""

import sys

from .data.loader import resolve_datasets

LOGO_DRAMATIC = r"""
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║                                                                          ║
    ║   ███████╗ █████╗ ████████╗   ████████╗██╗   ██╗███╗   ███╗███╗   ███╗██╗   ██╗   ║
    ║   ██╔════╝██╔══██╗╚══██╔══╝   ╚══██╔══╝██║   ██║████╗ ████║███╗ ████║╚██╗ ██╔╝   ║
    ║   █████╗  ███████║   ██║         ██║   ██║   ██║██╔████╔██║██╔████╔██║ ╚████╔╝    ║
    ║   ██╔══╝  ██╔══██║   ██║         ██║   ██║   ██║██║╚██╔╝██║██║╚██╔╝██║  ╚██╔╝     ║
    ║   ██║     ██║  ██║   ██║         ██║   ╚██████╔╝██║ ╚═╝ ██║██║ ╚═╝ ██║   ██║       ║
    ║   ╚═╝     ╚═╝  ╚═╝   ╚═╝         ╚═╝    ╚═════╝ ╚═╝     ╚═╝╚═╝     ╚═╝   ╚═╝       ║
    ║                                                                          ║
    ║            ░▒▓█  B U I L D  ·  T R A I N  ·  C H A T  █▓▒░             ║
    ║                   five words. infinite models.                           ║
    ║                                                                          ║
    ╚══════════════════════════════════════════════════════════════════════════╝
"""

LOGO_FALLBACK = r"""
    ================================================================================
    |                                                                              |
    |   #######   #######   ########      ########  ##   ##  ###  ###  ###  ##  ## |
    |   ##        ##   ##   ##            ##        ##   ##  ##   ##   ##   ##  ## |
    |   #####     #######   ######        ######    ##   ##  ##   ##   ##   ####   |
    |   ##        ##   ##   ##            ##         ## ##   ##   ##   ##   ##  ## |
    |   ##        ##   ##   ########      ########    ###    ###  ###  ###  ##  ## |
    |                                                                              |
    |              ***  BUILD  .  TRAIN  .  CHAT  ***                              |
    |                   five words. infinite models.                               |
    |                                                                              |
    ================================================================================
"""

SAMPLES = {
    "scale": "10B",
    "token_limit": "4096",
    "type": "MOOE",
    "hf_model": "meta-llama/Meta-Llama-3-8B",
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


def _print_logo():
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        print(LOGO_DRAMATIC)
    except (UnicodeEncodeError, LookupError, OSError):
        print(LOGO_FALLBACK)


def _prompt(label: str, default: str = "", required: bool = False) -> str:
    """Read a line; use default when blank. Re-prompt if required and empty."""
    suffix = f" [{default}]" if default else ""
    while True:
        value = input(f"  {label}{suffix}: ").strip()
        if value:
            return value
        if default:
            return default
        if not required:
            return ""
        print("  This field is required.")


def _prompt_secret(label: str) -> str:
    """Read a token/key without echoing when possible."""
    try:
        import getpass
        value = getpass.getpass(f"  {label}: ").strip()
    except Exception:
        value = input(f"  {label}: ").strip()
    return value


def _choose_action() -> str:
    print("\n  What do you want to do?")
    print("    1 = Make Model     Build a native MOOE model and chat")
    print("    2 = Fine-tune      Train on your data with a HuggingFace model")
    print("    3 = API Chat       Use OpenAI, Anthropic, or Gemini")
    choice = input("  Choose [1]: ").strip().lower()
    if choice in ("2", "finetune", "fine-tune", "train"):
        return "finetune"
    if choice in ("3", "api", "chat"):
        return "api"
    return "make"


def _resolve_type(raw: str):
    if raw.upper() == "MOOE":
        from .models.mooe import MOOE
        return MOOE
    return raw


def _resolve_api_provider(raw: str) -> str:
    return API_PROVIDERS.get(raw.lower(), raw.lower())


def run_wizard():
    """Show logo, collect settings, then initialize and run."""
    _print_logo()
    print("  a = Adv     Full control + HuggingFace login")
    print("  b = Breeze  Minimal prompts; blank fields use samples")
    print()

    mode = input("  Choose mode [b]: ").strip().lower()
    if mode in ("a", "adv", "advanced"):
        config = _run_advanced()
    else:
        config = _run_breeze()

    api_only = config["action"] == "api"
    print("\n  Setting up environment (first run may take a few minutes)...")
    from .engine import FatTummyEngine
    engine = FatTummyEngine(api_only=api_only)

    _apply_config(engine, config)
    _run_action(engine, config)
    return engine


def _run_breeze() -> dict:
    print("\n  --- Breeze ---")
    action = _choose_action()
    config = {
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
        config["type"] = _prompt("Type", SAMPLES["type"])
        config["engine"] = "mooe"
        config["datasets_raw"] = _prompt(
            "Dataset (HF repo or file path)", required=False
        )
    elif action == "finetune":
        config["type"] = _prompt("HF Model", SAMPLES["hf_model"])
        config["engine"] = "hf"
        config["datasets_raw"] = _prompt(
            "Dataset (HF repo or file path)", required=True
        )
    else:
        provider = _prompt("API Provider", SAMPLES["api_provider"])
        config["engine"] = _resolve_api_provider(provider)
        config["api_key"] = _prompt_secret("API Key")
        config["type"] = config["engine"]

    return config


def _run_advanced() -> dict:
    print("\n  --- Adv ---")
    action = _choose_action()
    hf_token = _prompt_secret("HuggingFace Token (blank to skip)")

    config = {
        "action": action,
        "hf_token": hf_token,
        "scale": _prompt("Scale", SAMPLES["scale"]),
        "token_limit": _prompt("Token Limit", SAMPLES["token_limit"]),
        "temperature": _prompt("Temperature", SAMPLES["temperature"]),
        "epochs": _prompt("Epochs", SAMPLES["epochs"]),
        "datasets_raw": "",
        "dataset_modes": [],
        "datasets": [],
    }

    if action == "make":
        config["type"] = _prompt("Type", SAMPLES["type"])
        config["engine"] = _prompt("Engine", SAMPLES["engine"])
        config["datasets_raw"] = _prompt(
            "Dataset (HF repo or file path)", required=False
        )
    elif action == "finetune":
        config["type"] = _prompt("HF Model", SAMPLES["hf_model"])
        config["engine"] = "hf"
        config["datasets_raw"] = _prompt(
            "Dataset (HF repo or file path)", required=True
        )
    else:
        provider = _prompt("API Provider", SAMPLES["api_provider"])
        config["engine"] = _resolve_api_provider(provider)
        config["api_key"] = _prompt_secret("API Key")
        config["type"] = _prompt("Model name (optional)", "")

    return config


def _apply_config(engine, config: dict):
    if config.get("hf_token"):
        engine.hf_login(config["hf_token"])

    engine.action(config["action"])
    engine.modelbuild(config["scale"])
    engine.token_limit(int(config["token_limit"]))
    engine.temp(float(config["temperature"]))
    engine.epochs(int(config["epochs"]))

    raw_datasets = config.get("datasets_raw", "")
    if raw_datasets:
        print("\n  Loading datasets...")
        datasets, modes = resolve_datasets(raw_datasets, token=config.get("hf_token"))
        config["datasets"] = datasets
        config["dataset_modes"] = modes
        engine._dataset_modes = modes
        for dataset in datasets:
            engine.data(dataset)
    elif config["action"] == "finetune":
        raise ValueError("Fine-tuning requires a dataset (HF repo or local file).")

    if config["action"] == "api":
        engine.engine(config["engine"])
        engine.key(config.get("api_key", ""))
        if config.get("type"):
            engine._model_type = config["type"]
    else:
        if config.get("engine"):
            engine.engine(config["engine"])
        engine.type(_resolve_type(config["type"]))


def _run_action(engine, config: dict):
    action = config["action"]
    print()
    print("  Configuration ready.")
    print(f"  Action={action}  Temp={config['temperature']}  Epochs={config['epochs']}")
    if config.get("dataset_modes"):
        print(f"  Datasets loaded via: {', '.join(config['dataset_modes'])}")
    print()

    if action == "finetune":
        engine.finetune()
        print("\n  Fine-tuning complete. Opening chat...")
        engine.chat()
    else:
        engine.chat()
