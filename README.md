# FatTummy

A declarative, ultra-minimalist Python framework designed to collapse complex data processing, hardware detection (GPU/TPU), multi-engine inference (APIs + Local), fine-tuning, and custom architecture deployment into a beautiful, stateless 5-command interface.

## Installation

```bash
pip install fattummy
```

## Quick Start

Building and chatting with a custom 10B Mixture of Experts model:

```python
import fattummy as ft

ft.build()
ft.modelbuild("10B")
ft.type(ft.MOOE)
ft.data("bigcode/the-stack-v2", "bigcode/starcoderdata")
ft.temp(0.7)
ft.chat()
```

The framework automatically detects if you are on a TPU or GPU, installs the correct PyTorch wheels natively, configures Hugging Face dependencies, and launches an interactive chat session.
