# FatTummy

A declarative, ultra-minimalist Python framework designed to collapse complex data processing, hardware detection (GPU/TPU), multi-engine inference (APIs + Local), fine-tuning, and custom architecture deployment into a beautiful, stateless interface.

## Installation

```bash
pip install fattummy
```

**Python 3.11 or 3.12 recommended.** Python 3.14 is not yet supported by PyTorch — `import FatTummy` works, but Make Model and Fine-tune need an older Python. API Chat works on any version.

## Building the package

On **Python 3.14**, `python -m build` is very slow. Use the fast local builder instead:

```bash
python build_release.py
```

Or with Python 3.12: `py -3.12 -m pip install build hatchling && py -3.12 -m build`

Upload to PyPI:

```bash
pip install twine
twine upload dist/*
```

## Test in Google Colab

```python
!pip install fattummy

import FatTummy as ft

ft.build(interactive=False)
ft.engine("openai")
ft.key("YOUR_API_KEY")
ft.chat()
```

For the full wizard in Colab, use `ft.build()` (interactive mode).

## Quick Start

Build and chat with a model in five words:

```python
import FatTummy as ft

ft.build()
```

The terminal shows the FatTummy logo and lets you pick a mode:

- **Breeze** — pick an action, then answer a few prompts; blank fields use samples
- **Adv** — full control plus HuggingFace token login

### Actions

1. **Make Model** — build a native MOOE model and chat
2. **Fine-tune** — train a HuggingFace model on your data, then chat
3. **API Chat** — use OpenAI, Anthropic, or Gemini

### Datasets

Provide a **HuggingFace repo** (`user/dataset`) or a **local file** (`.json`, `.jsonl`, `.csv`, `.txt`).

FatTummy checks the dataset size automatically:

- **Under 500 MB** — full download
- **500 MB or larger** (or unknown size) — streaming

Separate multiple sources with commas.

### Programmatic API

For scripts and pipelines, disable the wizard and chain calls as before:

```python
import FatTummy as ft

ft.build(interactive=False)
ft.modelbuild("10B")
ft.type(ft.MOOE)
ft.data("bigcode/the-stack-v2", "bigcode/starcoderdata")
ft.temp(0.7)
ft.chat()
```

The framework automatically detects if you are on a TPU or GPU, installs the correct PyTorch wheels natively, configures Hugging Face dependencies, and launches an interactive chat session.
