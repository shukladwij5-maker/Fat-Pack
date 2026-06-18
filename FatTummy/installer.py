import os
import sys
import subprocess
import importlib

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
        print(f"FatTummy installing: {package_str} ...")
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        print(f"Failed to install {package_str}: {e}")

def detect_hardware_and_install():
    """
    Intelligent runtime environment checker.
    Installs torch_xla on TPU, PyTorch cu121 on GPU, and manages dependencies.
    """
    dependencies = [
        "transformers", 
        "datasets", 
        "pandas", 
        "huggingface_hub", 
        "openai", 
        "google-genai", 
        "anthropic"
    ]
    
    # Check dependencies
    for dep in dependencies:
        pkg_name = "google.genai" if dep == "google-genai" else dep
        if not _is_package_installed(pkg_name):
            _install_package(dep)

    # Detect TPU VM environment
    is_tpu = "TPU_NAME" in os.environ or "XRT_TPU_CONFIG" in os.environ
    # Detect GPU
    is_gpu = False
    try:
        subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT)
        is_gpu = True
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    if is_tpu:
        if not _is_package_installed("torch_xla"):
            print("FatTummy detected TPU VM. Installing torch_xla...")
            # Example wheels url, would be real in prod
            _install_package("torch~=2.2.0", ["-f", "https://storage.googleapis.com/libtpu-releases/index.html"])
            _install_package("torch_xla[tpu]~=2.2.0", ["-f", "https://storage.googleapis.com/libtpu-releases/index.html"])
    elif is_gpu:
        if not _is_package_installed("torch"):
            print("FatTummy detected NVIDIA GPU. Installing PyTorch cu121...")
            _install_package("torch", ["--index-url", "https://download.pytorch.org/whl/cu121"])
    else:
        # CPU fallback
        if not _is_package_installed("torch"):
            print("FatTummy detected CPU. Installing PyTorch...")
            _install_package("torch")

    # Final import check to ensure torch is available
    if not _is_package_installed("torch"):
        print("Warning: PyTorch installation failed or is not accessible.")

def ensure_installed():
    """Entry point for the FatTummy builder to lazily audit the environment."""
    detect_hardware_and_install()
