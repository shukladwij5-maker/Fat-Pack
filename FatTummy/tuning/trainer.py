"""Simple real fine-tuning loop used by FatTummy."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, List, Optional

from ..exceptions import FatTummyDatasetError, FatTummyDependencyError

try:
    from torch.utils.data import IterableDataset as _TorchIterableDataset
except ImportError:  # pragma: no cover - torch is optional until training starts.
    _TorchIterableDataset = object

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HAS_MODERN_AMP = False  # will be resolved lazily


def _resolve_amp_flag() -> bool:
    """Return True when torch.amp (modern API) is available."""
    try:
        import torch
        return hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler")
    except Exception:
        return False


class FatTummyTrainer:
    """Minimal trainer for causal language-model fine-tuning."""

    def __init__(
        self,
        model: Any,
        dataset: Any,
        tokenizer: Optional[Any] = None,
        epochs: int = 3,
        batch_size: int = 2,
        learning_rate: float = 5e-5,
        max_length: int = 128,
        output_dir: str = "fattummy_checkpoints",
        eval_dataset: Optional[Any] = None,
        log_every: int = 10,
        max_train_examples: Optional[int] = None,
        gradient_accumulation_steps: int = 1,
        mixed_precision: bool = True,
    ) -> None:
        self.model = model
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_length = max_length
        self.output_dir = Path(output_dir)
        self.eval_dataset = eval_dataset
        self.log_every = log_every
        self.max_train_examples = max_train_examples
        self.gradient_accumulation_steps = max(1, gradient_accumulation_steps)
        self.mixed_precision = mixed_precision

        # BUG FIX: When no tokenizer is provided (native byte-level mode), token IDs
        # are produced by ord(char) % vocab_size.  Extract vocab_size from model.config
        # so that native models with small vocabularies (e.g. vocab_size=64) never
        # generate out-of-range indices.  Defaults to 256 (full byte range).
        if tokenizer is None:
            try:
                self._vocab_size: int = int(model.config.vocab_size)
            except AttributeError:
                self._vocab_size = 256
        else:
            self._vocab_size = 256  # unused when tokenizer is present

        # Checkpoint resume support
        self._load_latest_checkpoint_if_exists()

    def _load_latest_checkpoint_if_exists(self) -> None:
        """Find the latest epoch directory under output_dir and load weights if possible."""
        if not self.output_dir.exists():
            return
        epochs = []
        for p in self.output_dir.glob("epoch-*"):
            try:
                epochs.append(int(p.name.split("-")[1]))
            except (IndexError, ValueError):
                continue
        if not epochs:
            return
        latest_epoch = max(epochs)
        checkpoint_dir = self.output_dir / f"epoch-{latest_epoch}"
        print(f"FatTummy: Found existing checkpoint directory '{checkpoint_dir}'. Resuming weights...")

        if hasattr(self.model, "from_pretrained"):
            try:
                if (checkpoint_dir / "model.safetensors").exists() or (checkpoint_dir / "pytorch_model.bin").exists():
                    self.model = self.model.from_pretrained(str(checkpoint_dir))
                    print("FatTummy: Successfully loaded Hugging Face model weights.")
            except Exception as e:
                print(f"FatTummy warning: Failed to resume HF model from pretrained checkpoint: {e}")
        else:
            import torch
            model_pt = checkpoint_dir / "model.pt"
            if model_pt.exists():
                try:
                    self.model.load_state_dict(torch.load(model_pt, map_location="cpu"))
                    print("FatTummy: Successfully loaded native model weights.")
                except Exception as e:
                    print(f"FatTummy warning: Failed to load native state dict: {e}")

    def finetune(self, epochs: Optional[int] = None) -> None:
        """Run the fine-tuning loop on TPU when available, otherwise GPU/CPU."""
        if epochs is not None:
            self.epochs = epochs
        if self._has_tpu():
            try:
                self._finetune_tpu()
                return
            except FatTummyDependencyError:
                print("FatTummy: torch_xla unavailable; falling back to GPU/CPU.")
        self._finetune_gpu_cpu()

    def _has_tpu(self) -> bool:
        """Return True when common TPU environment markers are present."""
        return "TPU_NAME" in os.environ or "XRT_TPU_CONFIG" in os.environ

    def _finetune_tpu(self) -> None:
        """Optional TPU entry point with clean fallback if torch_xla is absent."""
        try:
            import torch_xla.core.xla_model as xm
        except ImportError as exc:
            raise FatTummyDependencyError("TPU training requires torch_xla.") from exc
        device = xm.xla_device()
        self._enable_gradient_checkpointing()
        self._run_training_loop_xla(device=device, xm=xm)

    def _finetune_gpu_cpu(self) -> None:
        """Train on CUDA when available, otherwise CPU."""
        try:
            import torch
        except ImportError as exc:
            raise FatTummyDependencyError("Fine-tuning requires PyTorch. Install it with: pip install torch") from exc
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._run_training_loop(device=device)

    def _enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing on HF models to reduce activation memory."""
        if hasattr(self.model, "gradient_checkpointing_enable"):
            try:
                self.model.gradient_checkpointing_enable()
                print("FatTummy: Gradient checkpointing enabled (reduces activation memory).")
            except Exception as e:
                print(f"FatTummy warning: Could not enable gradient checkpointing: {e}")

    def _run_training_loop(self, device: Any, xla_model: Optional[Any] = None) -> None:
        """Shared training loop for CUDA/CPU devices with AMP and gradient accumulation."""
        import torch
        from torch.utils.data import DataLoader

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model.to(device)
        self.model.train()

        is_cuda = getattr(device, "type", "") == "cuda"
        # Use multiple DataLoader workers only on platforms that support it well.
        # Windows + CUDA often needs 0 workers to avoid spawn issues; Linux/Colab can use 2.
        num_workers = 0 if sys.platform == "win32" else 2
        pin_memory = is_cuda

        loader = self._build_dataloader(device, num_workers=num_workers, pin_memory=pin_memory)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)

        use_amp = is_cuda and self.mixed_precision
        # Use the modern torch.amp API (PyTorch >= 2.4); fall back to the
        # legacy torch.cuda.amp API on older installs.
        use_modern_amp = use_amp and _resolve_amp_flag()
        if use_modern_amp:
            scaler = torch.amp.GradScaler(device="cuda")
        elif use_amp:
            scaler = torch.cuda.amp.GradScaler()  # legacy fallback
        else:
            scaler = None

        global_step = 0
        optimizer.zero_grad(set_to_none=True)
        running_loss_sum = 0.0

        for epoch in range(1, self.epochs + 1):
            batch_count = 0
            for batch in loader:
                batch = {key: value.to(device, non_blocking=pin_memory) for key, value in batch.items()}

                if use_modern_amp:
                    autocast_ctx = torch.amp.autocast(device_type="cuda")
                elif use_amp:
                    autocast_ctx = torch.cuda.amp.autocast()  # legacy
                else:
                    import contextlib
                    autocast_ctx = contextlib.nullcontext()

                with autocast_ctx:
                    output = self.model(input_ids=batch["input_ids"], labels=batch["labels"])
                    loss = output["loss"] if isinstance(output, dict) else output.loss
                    loss = loss / self.gradient_accumulation_steps

                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (global_step + 1) % self.gradient_accumulation_steps == 0:
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                global_step += 1
                batch_count += 1

                # .item() synchronises the device stream — only do it when logging.
                if global_step % self.log_every == 0:
                    step_loss = loss.detach().item() * self.gradient_accumulation_steps
                    running_loss_sum += step_loss
                    print(f"FatTummy train step={global_step} loss={step_loss:.4f}")
                else:
                    running_loss_sum += loss.detach().item() * self.gradient_accumulation_steps

            if batch_count == 0:
                raise FatTummyDatasetError("Fine-tuning dataset did not yield any text examples.")
            avg_loss = running_loss_sum / batch_count
            running_loss_sum = 0.0
            eval_loss = self._evaluate(device) if self.eval_dataset is not None else None
            suffix = f" eval_loss={eval_loss:.4f}" if eval_loss is not None else ""
            print(f"FatTummy epoch={epoch}/{self.epochs} loss={avg_loss:.4f}{suffix}")
            self._save_checkpoint(epoch)

    def _run_training_loop_xla(self, device: Any, xm: Any) -> None:
        """XLA/TPU-specific training loop.

        Key differences from the CUDA loop:
        - ``xm.mark_step()`` MUST be called after every backward + optimizer step
          to flush the XLA lazy graph.  Calling ``.item()`` before ``mark_step()``
          forces the entire accumulated graph to run at once, blowing HBM.
        - Loss values are collected asynchronously via ``xm.add_step_closure()``
          so we never block on a CPU read inside the hot loop.
        - AMP is NOT used; TPUs natively operate in bf16 via XLA.
        """
        import torch
        from torch.utils.data import DataLoader

        # TPU HBM is limited — warn if settings are aggressive.
        if self.max_length > 256 or self.batch_size > 1:
            print(
                f"FatTummy WARNING: max_length={self.max_length} batch_size={self.batch_size} "
                "may exceed TPU HBM (15.75 GB) for large models.  "
                "Consider: max_length<=256, batch_size=1, gradient_accumulation_steps>=4."
            )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model.to(device)
        self.model.train()

        loader = self._build_dataloader(device, num_workers=0, pin_memory=False)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)

        global_step = 0
        optimizer.zero_grad(set_to_none=True)

        # Collect losses asynchronously; closures run after mark_step().
        _loss_log: List[float] = []

        def _record_loss(loss_tensor: Any, step: int, log_every: int) -> None:
            val = loss_tensor.item()
            _loss_log.append(val)
            if step % log_every == 0:
                print(f"FatTummy train step={step} loss={val:.4f}")

        for epoch in range(1, self.epochs + 1):
            batch_count = 0
            for batch in loader:
                batch = {key: value.to(device) for key, value in batch.items()}

                output = self.model(input_ids=batch["input_ids"], labels=batch["labels"])
                loss = output["loss"] if isinstance(output, dict) else output.loss
                loss = loss / self.gradient_accumulation_steps
                loss.backward()

                if (global_step + 1) % self.gradient_accumulation_steps == 0:
                    xm.optimizer_step(optimizer)
                    optimizer.zero_grad(set_to_none=True)

                # Schedule the CPU read AFTER mark_step() flushes the graph.
                # Capturing loss_detached + step in closure to avoid late-binding.
                loss_detached = loss.detach()
                captured_step = global_step + 1
                xm.add_step_closure(
                    _record_loss,
                    args=(loss_detached, captured_step, self.log_every),
                )

                # CRITICAL: flush the XLA lazy graph every step.
                xm.mark_step()

                global_step += 1
                batch_count += 1

            if batch_count == 0:
                raise FatTummyDatasetError("Fine-tuning dataset did not yield any text examples.")

            # Drain any pending closures before computing epoch stats.
            xm.mark_step()
            avg_loss = (sum(_loss_log) / len(_loss_log)) if _loss_log else float("nan")
            _loss_log.clear()
            print(f"FatTummy epoch={epoch}/{self.epochs} loss={avg_loss:.4f}")
            self._save_checkpoint(epoch)

    def _build_dataloader(self, device: Any, num_workers: int, pin_memory: bool) -> Any:
        """Build a DataLoader from self.dataset, handling streaming vs map-style."""
        from torch.utils.data import DataLoader

        data_source = self._select_split(self.dataset)
        is_streaming = self._is_streaming_source(data_source)
        if is_streaming:
            train_dataset = _IterableTextDataset(
                lambda: self._iter_texts(data_source, limit=self.max_train_examples),
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                vocab_size=self._vocab_size,
            )
            return DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                collate_fn=_collate_batch,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
        else:
            examples = list(self._iter_texts(data_source, limit=self.max_train_examples))
            if not examples:
                raise FatTummyDatasetError("Fine-tuning dataset did not yield any text examples.")
            train_dataset = _TextDataset(
                examples, tokenizer=self.tokenizer, max_length=self.max_length,
                vocab_size=self._vocab_size,
            )
            return DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=_collate_batch,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )

    def _evaluate(self, device: Any) -> Optional[float]:
        """Run a simple evaluation loop and return mean loss."""
        import torch
        from torch.utils.data import DataLoader

        if self.eval_dataset is None:
            return None
        examples = list(self._iter_texts(self._select_split(self.eval_dataset)))
        if not examples:
            return None
        dataset = _TextDataset(
            examples, tokenizer=self.tokenizer, max_length=self.max_length,
            vocab_size=self._vocab_size,
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, collate_fn=_collate_batch)
        self.model.eval()
        losses: List[float] = []
        with torch.no_grad():
            for batch in loader:
                batch = {key: value.to(device) for key, value in batch.items()}
                output = self.model(input_ids=batch["input_ids"], labels=batch["labels"])
                loss = output["loss"] if isinstance(output, dict) else output.loss
                losses.append(float(loss.detach().cpu()))
        self.model.train()
        return sum(losses) / len(losses) if losses else None

    def _save_checkpoint(self, epoch: int) -> None:
        """Save model weights or delegate to a model-specific save method."""
        checkpoint_dir = self.output_dir / f"epoch-{epoch}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(str(checkpoint_dir))
            if self.tokenizer is not None and hasattr(self.tokenizer, "save_pretrained"):
                self.tokenizer.save_pretrained(str(checkpoint_dir))
            return

        import torch

        torch.save(self.model.state_dict(), checkpoint_dir / "model.pt")

    def _select_split(self, dataset: Any) -> Any:
        """Return the train split from dataset dictionaries when present."""
        if isinstance(dataset, dict) and "train" in dataset:
            return dataset["train"]
        return dataset

    def _is_streaming_source(self, dataset: Any) -> bool:
        """Return True for iterable datasets that should not be materialized.

        Hugging Face streaming datasets expose ``__iter__`` but NOT a meaningful
        ``__len__``.  However, in some HF versions the class *does* define
        ``__len__`` and raises ``TypeError`` when called.  We therefore:
          1. Always treat known HF IterableDataset types as streaming.
          2. Fall back to the __iter__-without-__len__ heuristic.
        """
        if isinstance(dataset, (str, list, tuple, dict)):
            return False
        # Detect HF datasets.IterableDataset by class name (avoids import overhead).
        cls_name = type(dataset).__name__
        if cls_name in ("IterableDataset", "IterableDatasetDict"):
            return True
        # Also catch any torch IterableDataset subclass.
        if isinstance(dataset, _TorchIterableDataset):
            return True
        if not hasattr(dataset, "__iter__"):
            return False
        # Has __len__? Try calling it; if it raises, treat as streaming.
        if hasattr(dataset, "__len__"):
            try:
                dataset.__len__()
                return False  # real length ➜ map-style
            except Exception:
                return True  # __len__ unusable ➜ streaming
        return True

    def _iter_texts(self, dataset: Any, limit: Optional[int] = None) -> Iterator[str]:
        """Yield text strings from common Python and datasets containers."""
        yielded = 0

        def emit(text: str) -> Iterator[str]:
            nonlocal yielded
            if limit is None or yielded < limit:
                yielded += 1
                yield text

        if dataset is None:
            return
        if isinstance(dataset, str):
            yield from emit(dataset)
            return
        if isinstance(dataset, list) and dataset and not isinstance(dataset[0], dict):
            for item in dataset:
                yield from emit(str(item))
                if limit is not None and yielded >= limit:
                    return
            return
        iterable: Iterable[Any]
        if isinstance(dataset, dict):
            if "train" in dataset:
                iterable = dataset["train"]
            else:
                iterable = [dataset]
        else:
            iterable = dataset
        for item in iterable:
            if isinstance(item, str):
                yield from emit(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content") or item.get("prompt") or item.get("completion")
                if text:
                    yield from emit(str(text))
            if limit is not None and yielded >= limit:
                return


class _TextDataset:
    """Torch dataset that tokenizes text for causal LM training."""

    def __init__(self, texts: List[str], tokenizer: Optional[Any], max_length: int, vocab_size: int = 256) -> None:
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.vocab_size = vocab_size

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return _encode_text(
            self.texts[index], tokenizer=self.tokenizer,
            max_length=self.max_length, vocab_size=self.vocab_size,
        )


class _IterableTextDataset(_TorchIterableDataset):
    """Iterable torch dataset for streaming text sources."""

    def __init__(
        self,
        text_factory: Callable[[], Iterator[str]],
        tokenizer: Optional[Any],
        max_length: int,
        vocab_size: int = 256,
    ) -> None:
        self.text_factory = text_factory
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.vocab_size = vocab_size

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for text in self.text_factory():
            yield _encode_text(
                text, tokenizer=self.tokenizer,
                max_length=self.max_length, vocab_size=self.vocab_size,
            )


def _encode_text(text: str, tokenizer: Optional[Any], max_length: int, vocab_size: int = 256) -> dict[str, Any]:
    """Encode text into input IDs and causal-LM labels.

    When no tokenizer is provided (native byte-level mode), characters are
    mapped with ``ord(char) % vocab_size`` so that the resulting IDs are always
    within the model's embedding table range.  The previous ``min(ord(char), 255)``
    implementation was incorrect for models with small vocabularies (e.g. 64).
    """
    if tokenizer is not None:
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None,
        )
        input_ids = encoded["input_ids"]
    else:
        # BUG FIX: use % vocab_size instead of min(ord(char), 255) so token IDs
        # never exceed the model's embedding table size.
        input_ids = [ord(char) % vocab_size for char in text[:max_length]]
    if len(input_ids) < 2:
        input_ids = input_ids + [0] * (2 - len(input_ids))
    return {"input_ids": input_ids, "labels": list(input_ids)}


def _collate_batch(examples: List[dict[str, Any]]) -> dict[str, Any]:
    """Pad variable-length examples into tensors."""
    import torch

    max_len = max(len(example["input_ids"]) for example in examples)
    input_ids = []
    labels = []
    for example in examples:
        ids = list(example["input_ids"])
        pad = max_len - len(ids)
        input_ids.append(ids + [0] * pad)
        labels.append(list(example["labels"]) + [-100] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }
