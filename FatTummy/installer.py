"""Dependency auditing helpers.

FatTummy keeps heavyweight integrations optional. The builder should never run
``pip install`` behind the user's back; adapters raise clear dependency errors
when a selected backend is actually used.
"""

from __future__ import annotations

import importlib.util
import sys
from typing import Dict, Iterable, List, Mapping

MIN_PYTHON = (3, 8)
MAX_TESTED_PYTHON = (3, 13)

OPTIONAL_EXTRAS: Mapping[str, Mapping[str, str]] = {
    "data": {"datasets": "datasets", "huggingface_hub": "huggingface_hub"},
    "hf": {"transformers": "transformers", "torch": "torch"},
    "ollama": {},
    "openai": {"openai": "openai"},
    "anthropic": {"anthropic": "anthropic"},
    "gemini": {"google-genai": "google.genai"},
    "native": {"torch": "torch"},
    "train": {"torch": "torch"},
}


def _check_python_version() -> None:
    """Validate the interpreter version and print non-fatal compatibility notes."""
    if sys.version_info < MIN_PYTHON:
        raise RuntimeError(
            f"FatTummy requires Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]} or newer."
        )
    if sys.version_info[:2] > MAX_TESTED_PYTHON:
        print(
            "FatTummy warning: this Python version is newer than the latest "
            "tested PyTorch runtime."
        )


def is_package_installed(import_name: str) -> bool:
    """Return True when an importable package/module is available."""
    return importlib.util.find_spec(import_name) is not None


def missing_dependencies(extras: Iterable[str]) -> Dict[str, List[str]]:
    """Return missing pip packages grouped by requested optional extra."""
    missing: Dict[str, List[str]] = {}
    for extra in extras:
        required = OPTIONAL_EXTRAS.get(extra, {})
        absent = [pip_name for pip_name, import_name in required.items() if not is_package_installed(import_name)]
        if absent:
            missing[extra] = absent
    return missing


def format_install_hint(extras: Iterable[str]) -> str:
    """Build a concise pip hint for missing optional extras."""
    missing = missing_dependencies(extras)
    packages = sorted({pkg for group in missing.values() for pkg in group})
    if not packages:
        return ""
    return "Install optional dependencies with: pip install " + " ".join(packages)


def ensure_api_deps() -> None:
    """Audit cloud API dependencies without installing them."""
    _check_python_version()
    hint = format_install_hint(["openai", "anthropic", "gemini"])
    if hint:
        print(f"FatTummy optional API dependencies missing. {hint}")


def detect_hardware_and_install() -> None:
    """Audit local dependency availability without mutating the environment."""
    _check_python_version()
    hint = format_install_hint(["data", "hf", "native", "train", "openai", "anthropic", "gemini"])
    if hint:
        print(f"FatTummy optional dependencies missing. {hint}")


def ensure_installed(api_only: bool = False) -> None:
    """Audit environment readiness for the builder."""
    if api_only:
        ensure_api_deps()
    else:
        detect_hardware_and_install()
