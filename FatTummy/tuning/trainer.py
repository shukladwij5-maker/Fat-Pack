"""Simple real fine-tuning loop used by FatTummy."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, List, Literal, Optional

from ..exceptions import FatTummyDatasetError, FatTummyDependencyError

try:
    from torch.utils.data import IterableDataset as _TorchIterableDataset
except ImportError:  # pragma: no cover - torch is optional until training starts.
    _TorchIterableDataset = object

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HAS_MODERN_AMP = False  # will be resolved lazily

OptimizerChoice = Literal["adamw", "lion"]
SchedulerChoice = Literal["cosine", "linear", "none"]


def _resolve_amp_flag() -> bool:
    """Return True when torch.amp (modern API) is available."""
    try:
        import torch
        return hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler")
    except Exception:
        return False


def _ensure_lion() -> Any:
    """Return the Lion optimizer class, auto-installing lion-pytorch if needed.

    We install silently and non-interactively so the user is never blocked by a
    missing package mid-session.  The install is attempted only once per process.
    """
    try:
        from lion_pytorch import Lion  # type: ignore[import]
        return Lion
    except ImportError:
        pass

    print("FatTummy: Lion optimizer not found — installing lion-pytorch automatically...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "lion-pytorch", "--quiet"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError as exc:
        raise FatTummyDependencyError(
            "Failed to auto-install lion-pytorch. "
            "Run manually: pip install lion-pytorch"
        ) from exc

    try:
        from lion_pytorch import Lion  # type: ignore[import]
        print("FatTummy: lion-pytorch installed successfully.")
        return Lion
    except ImportError as exc:
        raise FatTummyDependencyError(
            "lion-pytorch was installed but cannot be imported. "
            "Restart your runtime and try again."
        ) from exc


def _encode_spacebyte(text: str, max_length: int) -> List[int]:
    """SpaceByte encoding: raw UTF-8 bytes, vocab_size=256.

    SpaceByte is a byte-level tokenisation scheme where each byte of the
    UTF-8 encoded string becomes one token (0-255).  No special tokens,
    no BPE merges — just bytes.  This matches the spacebyte engine's
    vocab_size=256 and makes the model architecture-agnostic to language.
    """
    raw = text.encode("utf-8", errors="replace")[:max_length]
    return list(raw)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class FatTummyTrainer:
    """Minimal trainer for causal language-model fine-tuning.

    Advanced knobs exposed:
      optimizer         – "adamw" (default) or "lion" (auto-installed if missing)
      use_spacebyte     – True  → raw UTF-8 byte tokenisation (vocab_size=256)
                          False → ord(char) % vocab_size  (existing behaviour)
      lr_scheduler      – "cosine" | "linear" | "none"
      weight_decay      – L2 regularisation coefficient (default 0.01)
      warmup_steps      – linear warmup for the first N optimiser steps
      clip_grad_norm    – max gradient norm (None = disabled)
    """

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
        # ── Advanced knobs ───────────────────────────────────────────────────
        optimizer: OptimizerChoice = "adamw",
        use_spacebyte: bool = False,
        lr_scheduler: SchedulerChoice = "none",
        weight_decay: float = 0.01,
        warmup_steps: int = 0,
        clip_grad_norm: Optional[float] = None,
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
        self.optimizer_choice: OptimizerChoice = optimizer.lower()  # type: ignore[assignment]
        self.use_spacebyte = use_spacebyte
        self.lr_scheduler_choice: SchedulerChoice = lr_scheduler.lower()  # type: ignore[assignment]
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.clip_grad_norm = clip_grad_norm

        # Resolve vocab_size for native byte-level encoding.
        # SpaceByte always uses 256 (full byte range).
        # Otherwise extract from model.config, falling back to 256.
        if tokenizer is None:
            if use_spacebyte:
                self._vocab_size: int = 256
            else:
                try:
                    self._vocab_size = int(model.config.vocab_size)
                except AttributeError:
                    self._vocab_size = 256
        else:
            self._vocab_size = 256  # unused when tokenizer is present

        # Checkpoint resume support
        self._load_latest_checkpoint_if_exists()

    # ------------------------------------------------------------------
    # Checkpoint resume
    # ------------------------------------------------------------------

    def _load_latest_checkpoint_if_exists(self) -> None:
        """Find the latest epoch directory under output_dir and load weights."""
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
                    self.model.load_state_dict(torch.load(model_pt, map_location="cpu", weights_only=False))
                    print("FatTummy: Successfully loaded native model weights.")
                except Exception as e:
                    print(f"FatTummy warning: Failed to load native state dict: {e}")

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def finetune(self, epochs: Optional[int] = None) -> None:
        """Run the fine-tuning loop on TPU when available, otherwise GPU/CPU."""
        if epochs is not None:
            self.epochs = epochs

        enc_label = "SpaceByte (UTF-8 bytes)" if self.use_spacebyte else f"byte-ord%{self._vocab_size}"
        tok_label  = "HF tokenizer" if self.tokenizer else enc_label
        print(
            f"FatTummy: optimizer={self.optimizer_choice}  "
            f"scheduler={self.lr_scheduler_choice}  "
            f"tokenizer={tok_label}  "
            f"lr={self.learning_rate}  wd={self.weight_decay}  "
            f"warmup={self.warmup_steps}  clip={self.clip_grad_norm}"
        )

        if self._has_tpu():
            try:
                self._finetune_tpu()
                return
            except FatTummyDependencyError:
                print("FatTummy: torch_xla unavailable; falling back to GPU/CPU.")
        self._finetune_gpu_cpu()

    # ------------------------------------------------------------------
    # Device dispatch
    # ------------------------------------------------------------------

    def _has_tpu(self) -> bool:
        return "TPU_NAME" in os.environ or "XRT_TPU_CONFIG" in os.environ

    def _finetune_tpu(self) -> None:
        try:
            import torch_xla.core.xla_model as xm
        except ImportError as exc:
            raise FatTummyDependencyError("TPU training requires torch_xla.") from exc
        device = xm.xla_device()
        self._enable_gradient_checkpointing()
        self._run_training_loop_xla(device=device, xm=xm)

    def _finetune_gpu_cpu(self) -> None:
        try:
            import torch
        except ImportError as exc:
            raise FatTummyDependencyError("Fine-tuning requires PyTorch. Install it with: pip install torch") from exc
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._run_training_loop(device=device)

    def _enable_gradient_checkpointing(self) -> None:
        if hasattr(self.model, "gradient_checkpointing_enable"):
            try:
                self.model.gradient_checkpointing_enable()
                print("FatTummy: Gradient checkpointing enabled (reduces activation memory).")
            except Exception as e:
                print(f"FatTummy warning: Could not enable gradient checkpointing: {e}")

    # ------------------------------------------------------------------
    # Optimizer factory
    # ------------------------------------------------------------------

    def _build_optimizer(self, device_type: str) -> Any:
        """Create the configured optimizer, auto-installing Lion if needed."""
        import torch

        params = self.model.parameters()
        choice = self.optimizer_choice

        if choice == "lion":
            Lion = _ensure_lion()
            # Lion's effective LR is typically 3-10× smaller than AdamW.
            # We halve the user's LR automatically when they haven't set a tiny value.
            lion_lr = self.learning_rate / 3 if self.learning_rate >= 1e-4 else self.learning_rate
            print(f"FatTummy: Using Lion optimizer (lr adjusted to {lion_lr:.2e}, wd={self.weight_decay}).")
            return Lion(params, lr=lion_lr, weight_decay=self.weight_decay)

        # Default: AdamW
        return torch.optim.AdamW(params, lr=self.learning_rate, weight_decay=self.weight_decay)

    # ------------------------------------------------------------------
    # Scheduler factory
    # ------------------------------------------------------------------

    def _build_scheduler(self, optimizer: Any, total_steps: int) -> Optional[Any]:
        """Create a learning-rate scheduler (or None)."""
        choice = self.lr_scheduler_choice
        if choice == "none" or total_steps <= 0:
            return None

        try:
            from torch.optim.lr_scheduler import (
                CosineAnnealingLR,
                LinearLR,
                SequentialLR,
            )
        except ImportError:
            print("FatTummy warning: Could not import scheduler — skipping.")
            return None

        warmup = self.warmup_steps
        main_steps = max(1, total_steps - warmup)

        if choice == "cosine":
            main_sched = CosineAnnealingLR(optimizer, T_max=main_steps, eta_min=self.learning_rate * 0.1)
        else:  # linear
            main_sched = LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=main_steps)

        if warmup > 0:
            warmup_sched = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup)
            return SequentialLR(optimizer, schedulers=[warmup_sched, main_sched], milestones=[warmup])

        return main_sched

    # ------------------------------------------------------------------
    # Main training loop (CUDA / CPU)
    # ------------------------------------------------------------------

    def _run_training_loop(self, device: Any) -> None:
        import contextlib
        import torch
        from torch.utils.data import DataLoader

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model.to(device)
        self.model.train()

        is_cuda = getattr(device, "type", "") == "cuda"
        num_workers = 0 if sys.platform == "win32" else 2
        pin_memory = is_cuda

        loader = self._build_dataloader(device, num_workers=num_workers, pin_memory=pin_memory)

        # Estimate total optimiser steps for scheduler
        try:
            steps_per_epoch = max(1, len(loader))
        except Exception:
            steps_per_epoch = 100  # safe fallback for streaming loaders
        total_steps = (steps_per_epoch * self.epochs) // self.gradient_accumulation_steps

        optimizer = self._build_optimizer(getattr(device, "type", "cpu"))
        scheduler = self._build_scheduler(optimizer, total_steps)
        if scheduler is not None:
            print(f"FatTummy: lr_scheduler={self.lr_scheduler_choice} over {total_steps} optimiser steps.")

        use_amp = is_cuda and self.mixed_precision
        use_modern_amp = use_amp and _resolve_amp_flag()
        if use_modern_amp:
            scaler = torch.amp.GradScaler(device="cuda")
        elif use_amp:
            scaler = torch.cuda.amp.GradScaler()
        else:
            scaler = None

        if use_modern_amp:
            autocast_ctx = torch.amp.autocast(device_type="cuda")
        elif use_amp:
            autocast_ctx = torch.cuda.amp.autocast()
        else:
            autocast_ctx = contextlib.nullcontext()

        global_step = 0
        optimizer_step = 0
        optimizer.zero_grad(set_to_none=True)
        running_loss_sum = 0.0

        for epoch in range(1, self.epochs + 1):
            batch_count = 0
            for batch in loader:
                batch = {k: v.to(device, non_blocking=pin_memory) for k, v in batch.items()}

                with autocast_ctx:
                    output = self.model(input_ids=batch["input_ids"], labels=batch["labels"])
                    loss = output["loss"] if isinstance(output, dict) else output.loss
                    loss = loss / self.gradient_accumulation_steps

                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (global_step + 1) % self.gradient_accumulation_steps == 0:
                    if self.clip_grad_norm is not None:
                        if scaler is not None:
                            scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_step += 1

                global_step += 1
                batch_count += 1

                if global_step % self.log_every == 0:
                    step_loss = loss.detach().item() * self.gradient_accumulation_steps
                    running_loss_sum += step_loss
                    lr_now = optimizer.param_groups[0]["lr"]
                    print(f"FatTummy train step={global_step} loss={step_loss:.4f} lr={lr_now:.2e}")
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

    # ------------------------------------------------------------------
    # XLA / TPU training loop
    # ------------------------------------------------------------------

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
        optimizer = self._build_optimizer("xla")

        global_step = 0
        optimizer.zero_grad(set_to_none=True)

        _loss_log: List[float] = []

        def _record_loss(loss_tensor: Any, step: int, log_every: int) -> None:
            val = loss_tensor.item()
            _loss_log.append(val)
            if step % log_every == 0:
                print(f"FatTummy train step={step} loss={val:.4f}")

        for epoch in range(1, self.epochs + 1):
            batch_count = 0
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}

                output = self.model(input_ids=batch["input_ids"], labels=batch["labels"])
                loss = output["loss"] if isinstance(output, dict) else output.loss
                loss = loss / self.gradient_accumulation_steps
                loss.backward()

                if (global_step + 1) % self.gradient_accumulation_steps == 0:
                    if self.clip_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                    xm.optimizer_step(optimizer)
                    optimizer.zero_grad(set_to_none=True)

                loss_detached = loss.detach()
                captured_step = global_step + 1
                xm.add_step_closure(
                    _record_loss,
                    args=(loss_detached, captured_step, self.log_every),
                )
                xm.mark_step()

                global_step += 1
                batch_count += 1

            if batch_count == 0:
                raise FatTummyDatasetError("Fine-tuning dataset did not yield any text examples.")

            xm.mark_step()
            avg_loss = (sum(_loss_log) / len(_loss_log)) if _loss_log else float("nan")
            _loss_log.clear()
            print(f"FatTummy epoch={epoch}/{self.epochs} loss={avg_loss:.4f}")
            self._save_checkpoint(epoch)

    # ------------------------------------------------------------------
    # DataLoader
    # ------------------------------------------------------------------

    def _build_dataloader(self, device: Any, num_workers: int, pin_memory: bool) -> Any:
        from torch.utils.data import DataLoader

        data_source = self._select_split(self.dataset)
        is_streaming = self._is_streaming_source(data_source)
        if is_streaming:
            train_dataset = _IterableTextDataset(
                lambda: self._iter_texts(data_source, limit=self.max_train_examples),
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                vocab_size=self._vocab_size,
                use_spacebyte=self.use_spacebyte,
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
                examples,
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                vocab_size=self._vocab_size,
                use_spacebyte=self.use_spacebyte,
            )
            return DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=_collate_batch,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _evaluate(self, device: Any) -> Optional[float]:
        import torch
        from torch.utils.data import DataLoader

        if self.eval_dataset is None:
            return None
        examples = list(self._iter_texts(self._select_split(self.eval_dataset)))
        if not examples:
            return None
        dataset = _TextDataset(
            examples,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            vocab_size=self._vocab_size,
            use_spacebyte=self.use_spacebyte,
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, collate_fn=_collate_batch)
        self.model.eval()
        losses: List[float] = []
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                output = self.model(input_ids=batch["input_ids"], labels=batch["labels"])
                loss = output["loss"] if isinstance(output, dict) else output.loss
                losses.append(float(loss.detach().cpu()))
        self.model.train()
        return sum(losses) / len(losses) if losses else None

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def _save_checkpoint(self, epoch: int) -> None:
        checkpoint_dir = self.output_dir / f"epoch-{epoch}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(str(checkpoint_dir))
            if self.tokenizer is not None and hasattr(self.tokenizer, "save_pretrained"):
                self.tokenizer.save_pretrained(str(checkpoint_dir))
            return
        import torch
        torch.save(self.model.state_dict(), checkpoint_dir / "model.pt")

    # ------------------------------------------------------------------
    # Dataset utilities
    # ------------------------------------------------------------------

    def _select_split(self, dataset: Any) -> Any:
        if isinstance(dataset, dict) and "train" in dataset:
            return dataset["train"]
        return dataset

    def _is_streaming_source(self, dataset: Any) -> bool:
        if isinstance(dataset, (str, list, tuple, dict)):
            return False
        cls_name = type(dataset).__name__
        if cls_name in ("IterableDataset", "IterableDatasetDict"):
            return True
        if isinstance(dataset, _TorchIterableDataset):
            return True
        if not hasattr(dataset, "__iter__"):
            return False
        if hasattr(dataset, "__len__"):
            try:
                dataset.__len__()
                return False
            except Exception:
                return True
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


# ---------------------------------------------------------------------------
# Dataset wrappers
# ---------------------------------------------------------------------------


class _TextDataset:
    """Torch dataset that tokenizes text for causal LM training."""

    def __init__(
        self,
        texts: List[str],
        tokenizer: Optional[Any],
        max_length: int,
        vocab_size: int = 256,
        use_spacebyte: bool = False,
    ) -> None:
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.use_spacebyte = use_spacebyte

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return _encode_text(
            self.texts[index],
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            vocab_size=self.vocab_size,
            use_spacebyte=self.use_spacebyte,
        )


class _IterableTextDataset(_TorchIterableDataset):
    """Iterable torch dataset for streaming text sources."""

    def __init__(
        self,
        text_factory: Callable[[], Iterator[str]],
        tokenizer: Optional[Any],
        max_length: int,
        vocab_size: int = 256,
        use_spacebyte: bool = False,
    ) -> None:
        self.text_factory = text_factory
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.use_spacebyte = use_spacebyte

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for text in self.text_factory():
            yield _encode_text(
                text,
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                vocab_size=self.vocab_size,
                use_spacebyte=self.use_spacebyte,
            )


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------


def _encode_text(
    text: str,
    tokenizer: Optional[Any],
    max_length: int,
    vocab_size: int = 256,
    use_spacebyte: bool = False,
) -> dict[str, Any]:
    """Encode text into input IDs and causal-LM labels.

    Three modes:
    1. HF tokenizer provided  → use it as-is (truncation + no padding).
    2. use_spacebyte=True     → raw UTF-8 bytes (vocab_size=256 assumed).
    3. Default byte-ord mode  → ord(char) % vocab_size (handles tiny vocabs).
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
    elif use_spacebyte:
        input_ids = _encode_spacebyte(text, max_length)
    else:
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
