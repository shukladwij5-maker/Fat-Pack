import os
import sys
import subprocess
import importlib

MIN_PYTHON = (3, 9)
MAX_PYTHON = (3, 13)


def _check_python_version():
    if sys.version_info < MIN_PYTHON:
        print(f"FatTummy requires Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]} or newer.")
    elif sys.version_info > MAX_PYTHON:
        print(
            f"FatTummy Warning: Python {sys.version_info.major}.{sys.version_info.minor} "
            f"is not fully supported by PyTorch."
        )
        print("  For Make Model / Fine-tune, install Python 3.11 or 3.12:")
        print("  https://www.python.org/downloads/")
        print("  API Chat may still work without PyTorch.\n")


def _is_package_installed(package_name):
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False


def _install_package(package_str, extra_args=None):
    cmd = [sys.executable, "-m", "pip", "install", package_str]
    if extra_args:
        cmd.extend(extra_args)
    try:
        print(f"  Installing {package_str}...")
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        print(f"  Failed to install {package_str}: {e}")


def _install_dependencies(packages):
    for dep in packages:
        pkg_name = "google.genai" if dep == "google-genai" else dep
        if not _is_package_installed(pkg_name):
            _install_package(dep)


def ensure_api_deps():
    """Install cloud API packages only (no PyTorch)."""
    _check_python_version()
    _install_dependencies(["openai", "google-genai", "anthropic"])


def detect_hardware_and_install():
    """
    Intelligent runtime environment checker.
    Installs torch_xla on TPU, PyTorch cu121 on GPU, and manages dependencies.
    """
    _check_python_version()
    dependencies = [
        "transformers",
        "datasets",
        "pandas",
        "huggingface_hub",
        "openai",
        "google-genai",
        "anthropic",
    ]
    _install_dependencies(dependencies)

    is_tpu = "TPU_NAME" in os.environ or "XRT_TPU_CONFIG" in os.environ
    is_gpu = False
    try:
        subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT)
        is_gpu = True
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    if is_tpu:
        if not _is_package_installed("torch_xla"):
            print("  FatTummy detected TPU VM. Installing torch_xla...")
            _install_package("torch~=2.2.0", ["-f", "https://storage.googleapis.com/libtpu-releases/index.html"])
            _install_package("torch_xla[tpu]~=2.2.0", ["-f", "https://storage.googleapis.com/libtpu-releases/index.html"])
    elif is_gpu:
        if not _is_package_installed("torch"):
            print("  FatTummy detected NVIDIA GPU. Installing PyTorch...")
            _install_package("torch", ["--index-url", "https://download.pytorch.org/whl/cu121"])
    else:
        if not _is_package_installed("torch"):
            print("  FatTummy detected CPU. Installing PyTorch...")
            _install_package("torch")

    if not _is_package_installed("torch"):
        print("  Warning: PyTorch is not available. Make Model and Fine-tune will not work.")


def ensure_installed(api_only=False):
    """Entry point for the FatTummy builder to lazily audit the environment."""
    if api_only:
        ensure_api_deps()
    else:
        detect_hardware_and_install()
