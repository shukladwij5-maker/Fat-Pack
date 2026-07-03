# ==============================================================================
# FatTummy TPU Training Demo — Google Colab
# ==============================================================================
# Instructions:
# 1. Open Google Colab (https://colab.research.google.com).
# 2. Select TPU Runtime: "Runtime" -> "Change runtime type" -> Select "TPU v2" or "TPU v4".
# 3. Upload the built wheel file ("dist/fattummy-0.2.6-py3-none-any.whl") to Colab.
# 4. Copy, paste, and run the cells below.

# ------------------------------------------------------------------------------
# Cell 1: Install FatTummy and Dependencies
# ------------------------------------------------------------------------------
# !pip install fattummy-0.2.6-py3-none-any.whl

# ------------------------------------------------------------------------------
# Cell 2: Build and Train on TPU
# ------------------------------------------------------------------------------
import os
import torch
import FatTummy as ft
from FatTummy.models.mooe import MOOE, MOOEConfig

# 1. Initialize Engine for TPU Execution
# We configure Lion optimizer, SpaceByte raw byte tokenization, Cosine scheduling,
# and gradient clipping.
engine = (
    ft.build(interactive=False)
      .engine("mooe")      # Select native MoE architecture
      .optimizer("lion")   # Use Lion optimizer (automatically installs if missing)
      .spacebyte(True)     # Enable SpaceByte raw UTF-8 byte tokenisation
      .lr_scheduler("cosine")
      .weight_decay(0.01)
      .warmup(5)
      .clip_grad(1.0)
      .data([
          "The quick brown fox jumps over the lazy dog.",
          "Colab TPU acceleration makes MoE training extremely fast.",
          "SpaceByte tokenisation processes UTF-8 bytes directly with no tokenizer.",
          "Lion optimizer converges quickly with small learning rates."
      ])
)

# 2. Configure the Model (vocab_size=256 matches the SpaceByte byte range)
config = MOOEConfig(
    hidden_size=256,
    intermediate_size=1024,
    num_layers=4,
    num_experts=4,
    top_k=2,
    vocab_size=256
)
model = MOOE(config)

# Attach model to the engine
engine._model_instance = model
engine._compiled = True
engine._engine_name = "mooe"

# 3. Start Training
# FatTummy automatically detects the TPU runtime environment, compiles the graph 
# under PyTorch XLA, flushes via mark_step(), and handles metrics asynchronously.
print("Starting TPU Training...")
engine.finetune(epochs=5)
print("TPU Training Complete!")

# 4. Generate text from the trained weights
print("\nGenerating text:")
prompt = "The quick brown"
response = engine.generate(prompt)
print(f"Prompt: {prompt}")
print(f"Generated: {response}")
