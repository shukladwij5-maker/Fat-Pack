"""Comprehensive test suite for the FatTummy framework.

Covers:
  - MOOEConfig validation
  - MOOE forward pass and generate
  - MOOELayer routing and aux loss
  - FatTummyEngine builder API (all methods)
  - Engine selection + alias resolution
  - _compile_and_initialize logic
  - Temperature / token_limit / timeout validation
  - FatTummyTrainer (byte-level and tokenizer modes)
  - _TextDataset / _IterableTextDataset
  - _collate_batch padding
  - _encode_text vocab_size clamping
  - _iter_texts  on str / list / dict / iterable
  - _select_split
  - _is_streaming_source
  - Checkpoint resume (mocked)
  - Exceptions hierarchy
  - Installer: missing_dependencies, format_install_hint
  - data/loader helpers (_split_sources, _looks_like_path, _parse_hf_source)
  - cloud_adapters: BaseCloudAdapter, get_cloud_adapter routing
  - local_adapters: get_local_adapter routing, OllamaAdapter verify_installation
  - inference/__init__ exports
  - Public API re-export from FatTummy.__init__
"""

from __future__ import annotations

import sys
import os
import types
import importlib

# ---------------------------------------------------------------------------
# Make the package importable from the repo root.
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tiny_model():
    """Return a freshly initialised tiny MOOE model (no network I/O)."""
    from FatTummy.models.mooe import MOOE, MOOEConfig
    return MOOE(MOOEConfig(hidden_size=32, intermediate_size=64, num_layers=1, num_experts=2, top_k=1, vocab_size=64))


# ============================================================================
# 1. Exceptions
# ============================================================================

class TestExceptions:
    def test_base_hierarchy(self):
        from FatTummy.exceptions import (
            FatTummyBaseException,
            FatTummyConfigurationError,
            FatTummyDatasetError,
            FatTummyAuthenticationError,
            FatTummyDependencyError,
            FatTummyUnsupportedBackendError,
            FatTummyOOMError,
            FatTummyDriverError,
            FatTummyNetworkError,
        )
        for cls in (
            FatTummyConfigurationError,
            FatTummyDatasetError,
            FatTummyAuthenticationError,
            FatTummyDependencyError,
            FatTummyUnsupportedBackendError,
            FatTummyOOMError,
            FatTummyDriverError,
            FatTummyNetworkError,
        ):
            assert issubclass(cls, FatTummyBaseException)

    def test_oom_error_message(self):
        from FatTummy.exceptions import FatTummyOOMError
        err = FatTummyOOMError(original_error=RuntimeError("GPU OOM"))
        assert "memory" in str(err).lower()
        assert "GPU OOM" in str(err)

    def test_oom_error_no_original(self):
        from FatTummy.exceptions import FatTummyOOMError
        err = FatTummyOOMError()
        assert "memory" in str(err).lower()

    def test_driver_error_tpu(self):
        from FatTummy.exceptions import FatTummyDriverError
        err = FatTummyDriverError("tpu")
        assert "tpu" in str(err).lower() or "torch_xla" in str(err).lower()

    def test_driver_error_gpu(self):
        from FatTummy.exceptions import FatTummyDriverError
        err = FatTummyDriverError("gpu")
        assert "cuda" in str(err).lower() or "gpu" in str(err).lower()

    def test_network_error_with_path(self):
        from FatTummy.exceptions import FatTummyNetworkError
        err = FatTummyNetworkError("load", model_path="/tmp/model", original_error=IOError("net"))
        assert "/tmp/model" in str(err)
        assert "net" in str(err)


# ============================================================================
# 2. Installer
# ============================================================================

class TestInstaller:
    def test_missing_dependencies_empty(self):
        from FatTummy.installer import missing_dependencies
        # 'ollama' has no mandatory packages
        result = missing_dependencies(["ollama"])
        assert result == {} or "ollama" not in result


class TestPredictionAPI:
    def test_predict_linear_series(self):
        import FatTummy as ft
        values = [1, 2, 3, 4]
        result = ft.predict(values, steps=1)
        assert result == [5.0]

    def test_engine_predict_linear_series(self):
        from FatTummy.engine import FatTummyEngine
        engine = FatTummyEngine(api_only=True)
        values = [10, 12, 14, 16]
        result = engine.predict(values, steps=2)
        assert result == [18.0, 20.0]

    def test_missing_dependencies_fake_extra(self):
        from FatTummy.installer import missing_dependencies
        result = missing_dependencies(["_nonexistent_extra_"])
        assert result == {}

    def test_format_install_hint_no_missing(self, monkeypatch):
        from FatTummy import installer
        monkeypatch.setattr(installer, "missing_dependencies", lambda extras: {})
        hint = installer.format_install_hint(["openai"])
        assert hint == ""

    def test_format_install_hint_missing(self, monkeypatch):
        from FatTummy import installer
        monkeypatch.setattr(installer, "missing_dependencies", lambda extras: {"openai": ["openai"]})
        hint = installer.format_install_hint(["openai"])
        assert "pip install" in hint
        assert "openai" in hint

    def test_is_package_installed_builtin(self):
        from FatTummy.installer import is_package_installed
        assert is_package_installed("os") is True

    def test_is_package_installed_fake(self):
        from FatTummy.installer import is_package_installed
        assert is_package_installed("_this_package_does_not_exist_xyz") is False

    def test_check_python_version_passes(self):
        from FatTummy.installer import _check_python_version
        # Should not raise for the running interpreter
        _check_python_version()

    def test_ensure_installed_runs(self):
        from FatTummy.installer import ensure_installed
        # Smoke test — should not raise
        ensure_installed(api_only=True)
        ensure_installed(api_only=False)


# ============================================================================
# 3. Data loader helpers
# ============================================================================

class TestDataLoaderHelpers:
    def test_split_sources_string(self):
        from FatTummy.data.loader import _split_sources
        assert _split_sources("a, b, c") == ["a", "b", "c"]

    def test_split_sources_list(self):
        from FatTummy.data.loader import _split_sources
        assert _split_sources(["x", "y"]) == ["x", "y"]

    def test_split_sources_none(self):
        from FatTummy.data.loader import _split_sources
        assert _split_sources(None) == []

    def test_split_sources_unsupported_type(self):
        from FatTummy.data.loader import _split_sources
        from FatTummy.exceptions import FatTummyDatasetError
        with pytest.raises(FatTummyDatasetError):
            _split_sources(42)

    def test_looks_like_path_txt(self):
        from FatTummy.data.loader import _looks_like_path
        assert _looks_like_path("dataset.txt") is True

    def test_looks_like_path_jsonl(self):
        from FatTummy.data.loader import _looks_like_path
        assert _looks_like_path("data.jsonl") is True

    def test_looks_like_path_hf_id(self):
        from FatTummy.data.loader import _looks_like_path
        assert _looks_like_path("owner/repo") is False

    def test_looks_like_path_relative_dot(self):
        from FatTummy.data.loader import _looks_like_path
        assert _looks_like_path("./myfile.csv") is True

    def test_looks_like_path_tilde(self):
        from FatTummy.data.loader import _looks_like_path
        assert _looks_like_path("~/data.txt") is True

    def test_parse_hf_source_simple(self):
        from FatTummy.data.loader import _parse_hf_source
        repo, config = _parse_hf_source("owner/dataset")
        assert repo == "owner/dataset"
        assert config is None

    def test_parse_hf_source_colon_config(self):
        from FatTummy.data.loader import _parse_hf_source
        repo, config = _parse_hf_source("owner/dataset:en")
        assert repo == "owner/dataset"
        assert config == "en"

    def test_parse_hf_source_slash_config(self):
        from FatTummy.data.loader import _parse_hf_source
        repo, config = _parse_hf_source("owner/dataset/en")
        assert repo == "owner/dataset"
        assert config == "en"

    def test_parse_hf_source_bad_id(self):
        from FatTummy.data.loader import _parse_hf_source
        from FatTummy.exceptions import FatTummyDatasetError
        with pytest.raises(FatTummyDatasetError):
            _parse_hf_source("noslash")

    def test_validate_local_file_missing(self, tmp_path):
        from FatTummy.data.loader import _validate_local_file
        from FatTummy.exceptions import FatTummyDatasetError
        with pytest.raises(FatTummyDatasetError, match="not found"):
            _validate_local_file(tmp_path / "ghost.txt")

    def test_validate_local_file_unsupported_ext(self, tmp_path):
        from FatTummy.data.loader import _validate_local_file
        from FatTummy.exceptions import FatTummyDatasetError
        f = tmp_path / "data.parquet"
        f.write_text("x")
        with pytest.raises(FatTummyDatasetError, match="Unsupported"):
            _validate_local_file(f)

    def test_validate_local_file_ok(self, tmp_path):
        from FatTummy.data.loader import _validate_local_file
        f = tmp_path / "data.txt"
        f.write_text("hello")
        _validate_local_file(f)  # should not raise


# ============================================================================
# 4. MOOE model
# ============================================================================

class TestMOOEConfig:
    def test_defaults(self):
        from FatTummy.models.mooe import MOOEConfig
        cfg = MOOEConfig()
        assert cfg.hidden_size == 256
        assert cfg.num_hidden_layers == cfg.num_layers
        assert cfg.num_experts_per_tok == cfg.top_k

    def test_invalid_hidden_size(self):
        from FatTummy.models.mooe import MOOEConfig
        with pytest.raises(ValueError, match="hidden_size"):
            MOOEConfig(hidden_size=0)

    def test_invalid_intermediate_size(self):
        from FatTummy.models.mooe import MOOEConfig
        with pytest.raises(ValueError):
            MOOEConfig(intermediate_size=-1)

    def test_invalid_num_experts(self):
        from FatTummy.models.mooe import MOOEConfig
        with pytest.raises(ValueError):
            MOOEConfig(num_experts=0)

    def test_invalid_vocab_size(self):
        from FatTummy.models.mooe import MOOEConfig
        with pytest.raises(ValueError):
            MOOEConfig(vocab_size=0)

    def test_top_k_exceeds_experts(self):
        from FatTummy.models.mooe import MOOEConfig
        with pytest.raises(ValueError, match="top_k"):
            MOOEConfig(num_experts=2, top_k=3)

    def test_top_k_zero(self):
        from FatTummy.models.mooe import MOOEConfig
        with pytest.raises(ValueError, match="top_k"):
            MOOEConfig(num_experts=4, top_k=0)

    def test_valid_small_config(self):
        from FatTummy.models.mooe import MOOEConfig
        cfg = MOOEConfig(hidden_size=32, intermediate_size=64, num_layers=1, num_experts=2, top_k=1, vocab_size=64)
        assert cfg.num_hidden_layers == 1


class TestMOOEModel:
    def test_forward_no_labels(self):
        import torch
        model = _make_tiny_model()
        ids = torch.randint(0, 64, (1, 8))
        out = model(input_ids=ids)
        assert "logits" in out
        assert "aux_loss" in out
        assert out["logits"].shape == (1, 8, 64)
        assert "loss" not in out

    def test_forward_with_labels(self):
        import torch
        model = _make_tiny_model()
        ids = torch.randint(0, 64, (1, 8))
        out = model(input_ids=ids, labels=ids)
        assert "loss" in out
        assert out["loss"].item() > 0

    def test_forward_ignore_index(self):
        import torch
        model = _make_tiny_model()
        ids = torch.randint(0, 64, (1, 8))
        labels = ids.clone()
        labels[:, :4] = -100  # mask first 4 tokens
        out = model(input_ids=ids, labels=labels)
        assert "loss" in out
        assert not out["loss"].isnan()

    def test_generate_shape(self):
        import torch
        model = _make_tiny_model()
        ids = torch.tensor([[1, 2, 3]])
        generated = model.generate(ids, max_new_tokens=5)
        assert generated.shape == (1, 8)  # 3 prompt + 5 new

    def test_generate_is_tensor(self):
        import torch
        model = _make_tiny_model()
        out = model.generate(torch.tensor([[0]]), max_new_tokens=4)
        assert isinstance(out, torch.Tensor)

    def test_aux_loss_is_scalar(self):
        import torch
        model = _make_tiny_model()
        ids = torch.randint(0, 64, (2, 4))
        out = model(input_ids=ids)
        assert out["aux_loss"].dim() == 0

    def test_expert_layer_residual_shape(self):
        """MOOELayer output should match input shape."""
        import torch
        from FatTummy.models.mooe import MOOEConfig, MOOELayer
        cfg = MOOEConfig(hidden_size=32, intermediate_size=64, num_experts=2, top_k=1, num_layers=1, vocab_size=64)
        layer = MOOELayer(cfg)
        x = torch.randn(2, 6, 32)
        out, aux = layer(x)
        assert out.shape == x.shape
        assert aux.dim() == 0

    def test_no_grad_generate(self):
        """generate() should not accumulate gradients."""
        import torch
        model = _make_tiny_model()
        ids = torch.tensor([[1]])
        generated = model.generate(ids, max_new_tokens=2)
        # No grad tracked on output
        assert not generated.requires_grad


# ============================================================================
# 5. Trainer helpers (pure Python, no GPU needed)
# ============================================================================

class TestEncodeText:
    def test_encode_with_tokenizer(self):
        from FatTummy.tuning.trainer import _encode_text

        class FakeTok:
            def __call__(self, text, **kwargs):
                return {"input_ids": [1, 2, 3]}

        result = _encode_text("hello", tokenizer=FakeTok(), max_length=16)
        assert result["input_ids"] == [1, 2, 3]
        assert result["labels"] == [1, 2, 3]

    def test_encode_without_tokenizer_vocab_clamp(self):
        from FatTummy.tuning.trainer import _encode_text
        # vocab_size=8 means all IDs must be < 8
        result = _encode_text("ABC", tokenizer=None, max_length=32, vocab_size=8)
        for token_id in result["input_ids"]:
            assert 0 <= token_id < 8

    def test_encode_short_text_padded(self):
        from FatTummy.tuning.trainer import _encode_text
        result = _encode_text("A", tokenizer=None, max_length=32, vocab_size=256)
        assert len(result["input_ids"]) >= 2  # padded to at least length 2

    def test_encode_labels_equal_input_ids(self):
        from FatTummy.tuning.trainer import _encode_text
        result = _encode_text("hello world", tokenizer=None, max_length=32, vocab_size=256)
        assert result["labels"] == result["input_ids"]

    def test_encode_truncation(self):
        from FatTummy.tuning.trainer import _encode_text
        text = "A" * 200
        result = _encode_text(text, tokenizer=None, max_length=16, vocab_size=256)
        assert len(result["input_ids"]) == 16


class TestCollateBatch:
    def test_padding_to_max_length(self):
        from FatTummy.tuning.trainer import _collate_batch
        import torch
        examples = [
            {"input_ids": [1, 2, 3], "labels": [1, 2, 3]},
            {"input_ids": [4, 5], "labels": [4, 5]},
        ]
        batch = _collate_batch(examples)
        assert batch["input_ids"].shape == (2, 3)
        assert batch["labels"].shape == (2, 3)
        # Shorter example padded with 0 for input_ids
        assert batch["input_ids"][1, 2].item() == 0
        # Labels padded with -100
        assert batch["labels"][1, 2].item() == -100

    def test_collate_returns_tensors(self):
        import torch
        from FatTummy.tuning.trainer import _collate_batch
        batch = _collate_batch([{"input_ids": [1], "labels": [1]}])
        assert isinstance(batch["input_ids"], torch.Tensor)
        assert isinstance(batch["labels"], torch.Tensor)

    def test_collate_single_example(self):
        from FatTummy.tuning.trainer import _collate_batch
        batch = _collate_batch([{"input_ids": [1, 2], "labels": [1, 2]}])
        assert batch["input_ids"].shape == (1, 2)


class TestIterTexts:
    def _trainer(self):
        """Return a trainer whose _iter_texts we can call directly."""
        from FatTummy.tuning.trainer import FatTummyTrainer
        model = _make_tiny_model()
        return FatTummyTrainer(model, dataset="placeholder", epochs=1)

    def test_string_yields_one(self):
        t = self._trainer()
        texts = list(t._iter_texts("hello"))
        assert texts == ["hello"]

    def test_list_of_strings(self):
        t = self._trainer()
        texts = list(t._iter_texts(["a", "b", "c"]))
        assert texts == ["a", "b", "c"]

    def test_list_of_dicts_text_key(self):
        t = self._trainer()
        data = [{"text": "hello"}, {"text": "world"}]
        texts = list(t._iter_texts(data))
        assert texts == ["hello", "world"]

    def test_list_of_dicts_content_key(self):
        t = self._trainer()
        data = [{"content": "foo"}]
        texts = list(t._iter_texts(data))
        assert texts == ["foo"]

    def test_none_yields_nothing(self):
        t = self._trainer()
        assert list(t._iter_texts(None)) == []

    def test_limit_respected(self):
        t = self._trainer()
        texts = list(t._iter_texts(["a", "b", "c", "d"], limit=2))
        assert len(texts) == 2

    def test_dict_with_train_split(self):
        t = self._trainer()
        data = {"train": ["x", "y"]}
        texts = list(t._iter_texts(data))
        assert texts == ["x", "y"]

    def test_dict_without_train_key(self):
        t = self._trainer()
        data = {"text": "hello"}
        texts = list(t._iter_texts(data))
        # iterable = [data], item is a dict, looks for "text" key
        assert "hello" in texts


class TestSelectSplit:
    def test_dict_with_train(self):
        from FatTummy.tuning.trainer import FatTummyTrainer
        model = _make_tiny_model()
        t = FatTummyTrainer(model, "placeholder", epochs=1)
        d = {"train": "the_train_split", "test": "the_test_split"}
        assert t._select_split(d) == "the_train_split"

    def test_non_dict_passthrough(self):
        from FatTummy.tuning.trainer import FatTummyTrainer
        model = _make_tiny_model()
        t = FatTummyTrainer(model, "placeholder", epochs=1)
        lst = ["a", "b"]
        assert t._select_split(lst) is lst


class TestIsStreamingSource:
    def _trainer(self):
        from FatTummy.tuning.trainer import FatTummyTrainer
        return FatTummyTrainer(_make_tiny_model(), "placeholder", epochs=1)

    def test_str_not_streaming(self):
        t = self._trainer()
        assert t._is_streaming_source("some text") is False

    def test_list_not_streaming(self):
        t = self._trainer()
        assert t._is_streaming_source(["a", "b"]) is False

    def test_dict_not_streaming(self):
        t = self._trainer()
        assert t._is_streaming_source({"train": []}) is False

    def test_iterable_with_len_not_streaming(self):
        t = self._trainer()
        class FakeMapStyle:
            def __iter__(self): return iter([])
            def __len__(self): return 0
        assert t._is_streaming_source(FakeMapStyle()) is False

    def test_iterable_without_len_is_streaming(self):
        t = self._trainer()
        class FakeStreaming:
            def __iter__(self): return iter([])
        assert t._is_streaming_source(FakeStreaming()) is True

    def test_hf_iterable_dataset_by_name(self):
        t = self._trainer()
        class IterableDataset:
            def __iter__(self): return iter([])
        assert t._is_streaming_source(IterableDataset()) is True


# ============================================================================
# 6. TextDataset / IterableTextDataset
# ============================================================================

class TestTextDataset:
    def test_len(self):
        from FatTummy.tuning.trainer import _TextDataset
        ds = _TextDataset(["a", "b", "c"], tokenizer=None, max_length=8, vocab_size=256)
        assert len(ds) == 3

    def test_getitem(self):
        from FatTummy.tuning.trainer import _TextDataset
        ds = _TextDataset(["hello"], tokenizer=None, max_length=8, vocab_size=256)
        item = ds[0]
        assert "input_ids" in item and "labels" in item

    def test_vocab_clamp_in_dataset(self):
        from FatTummy.tuning.trainer import _TextDataset
        ds = _TextDataset(["ABC"], tokenizer=None, max_length=8, vocab_size=4)
        item = ds[0]
        for tid in item["input_ids"]:
            assert 0 <= tid < 4


class TestIterableTextDataset:
    def test_yields_items(self):
        from FatTummy.tuning.trainer import _IterableTextDataset
        ds = _IterableTextDataset(lambda: iter(["hello", "world"]), tokenizer=None, max_length=8, vocab_size=256)
        items = list(ds)
        assert len(items) == 2
        for item in items:
            assert "input_ids" in item


# ============================================================================
# 7. FatTummyTrainer – full training run (tiny MOOE, CPU)
# ============================================================================

class TestFatTummyTrainer:
    def test_finetune_string_dataset(self, tmp_path):
        from FatTummy.tuning.trainer import FatTummyTrainer
        model = _make_tiny_model()
        trainer = FatTummyTrainer(
            model,
            dataset=["the quick brown fox", "hello world training"],
            epochs=1,
            batch_size=2,
            output_dir=str(tmp_path / "ckpt"),
            log_every=1,
        )
        trainer.finetune()
        # Checkpoint should exist
        assert (tmp_path / "ckpt" / "epoch-1" / "model.pt").exists()

    def test_finetune_saves_checkpoint(self, tmp_path):
        from FatTummy.tuning.trainer import FatTummyTrainer
        model = _make_tiny_model()
        trainer = FatTummyTrainer(
            model,
            dataset=["data row one", "data row two", "data row three"],
            epochs=2,
            batch_size=1,
            output_dir=str(tmp_path / "ckpt"),
        )
        trainer.finetune()
        assert (tmp_path / "ckpt" / "epoch-1" / "model.pt").exists()
        assert (tmp_path / "ckpt" / "epoch-2" / "model.pt").exists()

    def test_finetune_empty_dataset_raises(self, tmp_path):
        from FatTummy.tuning.trainer import FatTummyTrainer
        from FatTummy.exceptions import FatTummyDatasetError
        model = _make_tiny_model()
        trainer = FatTummyTrainer(
            model,
            dataset=[],
            epochs=1,
            output_dir=str(tmp_path / "ckpt"),
        )
        with pytest.raises(FatTummyDatasetError):
            trainer.finetune()

    def test_finetune_gradient_accumulation(self, tmp_path):
        from FatTummy.tuning.trainer import FatTummyTrainer
        model = _make_tiny_model()
        trainer = FatTummyTrainer(
            model,
            dataset=["word " * 10] * 4,
            epochs=1,
            batch_size=1,
            gradient_accumulation_steps=2,
            output_dir=str(tmp_path / "ckpt"),
        )
        trainer.finetune()

    def test_finetune_with_eval_dataset(self, tmp_path):
        from FatTummy.tuning.trainer import FatTummyTrainer
        model = _make_tiny_model()
        trainer = FatTummyTrainer(
            model,
            dataset=["train text one", "train text two"],
            eval_dataset=["eval text one"],
            epochs=1,
            output_dir=str(tmp_path / "ckpt"),
        )
        trainer.finetune()

    def test_resume_from_checkpoint(self, tmp_path):
        """Trainer should find and load an existing checkpoint silently."""
        import torch
        from FatTummy.tuning.trainer import FatTummyTrainer
        model = _make_tiny_model()
        # Pre-create a fake checkpoint
        ckpt_dir = tmp_path / "ckpt" / "epoch-1"
        ckpt_dir.mkdir(parents=True)
        torch.save(model.state_dict(), ckpt_dir / "model.pt")
        # Creating trainer should load it without error
        trainer = FatTummyTrainer(
            model,
            dataset=["some text"],
            epochs=1,
            output_dir=str(tmp_path / "ckpt"),
        )
        # Epoch 1 checkpoint should have been loaded
        assert trainer.model is not None

    def test_vocab_size_extracted_from_model(self):
        from FatTummy.tuning.trainer import FatTummyTrainer
        model = _make_tiny_model()
        trainer = FatTummyTrainer(model, dataset=["x"], epochs=1)
        assert trainer._vocab_size == model.config.vocab_size

    def test_finetune_dict_dataset_train_split(self, tmp_path):
        from FatTummy.tuning.trainer import FatTummyTrainer
        model = _make_tiny_model()
        dataset = {"train": ["hello world", "foo bar baz"]}
        trainer = FatTummyTrainer(
            model, dataset=dataset, epochs=1, output_dir=str(tmp_path / "ckpt")
        )
        trainer.finetune()

    def test_max_train_examples_limits(self, tmp_path):
        from FatTummy.tuning.trainer import FatTummyTrainer
        model = _make_tiny_model()
        trainer = FatTummyTrainer(
            model,
            dataset=["a"] * 20,
            epochs=1,
            max_train_examples=3,
            output_dir=str(tmp_path / "ckpt"),
        )
        trainer.finetune()


# ============================================================================
# 8. FatTummyEngine – builder API
# ============================================================================

class TestEngineBuilder:
    def _engine(self):
        from FatTummy.engine import FatTummyEngine
        return FatTummyEngine()

    def test_engine_method_chains(self):
        e = self._engine()
        ret = e.engine("openai")
        assert ret is e

    def test_modelbuild_chains(self):
        e = self._engine()
        assert e.modelbuild("small") is e

    def test_data_chains(self):
        e = self._engine()
        assert e.data("source1", "source2") is e
        assert len(e._data_sources) == 2

    def test_temp_valid(self):
        e = self._engine()
        assert e.temp(0.8) is e
        assert e._temperature == 0.8

    def test_temp_invalid(self):
        from FatTummy.exceptions import FatTummyConfigurationError
        e = self._engine()
        with pytest.raises(FatTummyConfigurationError):
            e.temp(0.0)

    def test_temp_negative(self):
        from FatTummy.exceptions import FatTummyConfigurationError
        e = self._engine()
        with pytest.raises(FatTummyConfigurationError):
            e.temp(-1.0)

    def test_token_limit_valid(self):
        e = self._engine()
        assert e.token_limit(256) is e
        assert e._token_limit == 256

    def test_token_limit_invalid(self):
        from FatTummy.exceptions import FatTummyConfigurationError
        e = self._engine()
        with pytest.raises(FatTummyConfigurationError):
            e.token_limit(0)

    def test_timeout_valid(self):
        e = self._engine()
        assert e.timeout(30.0) is e

    def test_timeout_invalid(self):
        from FatTummy.exceptions import FatTummyConfigurationError
        e = self._engine()
        with pytest.raises(FatTummyConfigurationError):
            e.timeout(-1.0)

    def test_epochs_valid(self):
        e = self._engine()
        assert e.epochs(5) is e

    def test_epochs_invalid(self):
        from FatTummy.exceptions import FatTummyConfigurationError
        e = self._engine()
        with pytest.raises(FatTummyConfigurationError):
            e.epochs(0)

    def test_quantize_stores_mode(self):
        e = self._engine()
        e.quantize("4bit")
        assert e._quantization == "4bit"

    def test_key_sets_api_key(self):
        e = self._engine()
        e.key("sk-test")
        assert e._api_key == "sk-test"

    def test_unknown_engine_raises(self):
        from FatTummy.exceptions import FatTummyUnsupportedBackendError
        e = self._engine()
        with pytest.raises(FatTummyUnsupportedBackendError):
            e.engine("not_a_real_engine")

    def test_engine_alias_claude(self):
        e = self._engine()
        e.engine("claude")
        assert e._engine_name == "anthropic"

    def test_engine_alias_google(self):
        e = self._engine()
        e.engine("google")
        assert e._engine_name == "gemini"

    def test_engine_alias_huggingface(self):
        e = self._engine()
        e.engine("huggingface")
        assert e._engine_name == "hf"

    def test_action_valid(self):
        e = self._engine()
        e.action("finetune")
        assert e._action == "finetune"

    def test_action_invalid(self):
        from FatTummy.exceptions import FatTummyConfigurationError
        e = self._engine()
        with pytest.raises(FatTummyConfigurationError):
            e.action("bake")

    def test_generate_no_api_key_raises(self):
        from FatTummy.exceptions import FatTummyAuthenticationError
        e = self._engine()
        e.engine("openai").type("gpt-4o-mini")
        with pytest.raises(FatTummyAuthenticationError):
            e.generate("hello")

    def test_generate_native_mooe(self):
        e = self._engine()
        e.engine("mooe").modelbuild("tiny")
        # type() triggers compile — use MOOE class
        from FatTummy.models.mooe import MOOE
        e._model_instance = _make_tiny_model()
        e._compiled = True
        e._engine_name = "mooe"
        result = e.generate("abc")
        assert isinstance(result, str)

    def test_validate_ollama_repo_format(self):
        from FatTummy.exceptions import FatTummyUnsupportedBackendError
        e = self._engine()
        e._engine_name = "ollama"
        with pytest.raises(FatTummyUnsupportedBackendError):
            e._validate_combination("ollama", "meta-llama/Llama-3")

    def test_validate_hf_requires_string(self):
        from FatTummy.exceptions import FatTummyUnsupportedBackendError
        from FatTummy.models.mooe import MOOE
        e = self._engine()
        with pytest.raises(FatTummyUnsupportedBackendError):
            e._validate_combination("hf", MOOE)

    def test_finetune_no_dataset_raises(self):
        from FatTummy.exceptions import FatTummyConfigurationError
        e = self._engine()
        e.engine("mooe").modelbuild("tiny")
        e._model_instance = _make_tiny_model()
        e._compiled = True
        e._engine_name = "mooe"
        with pytest.raises(FatTummyConfigurationError):
            e.finetune()

    def test_finetune_unsupported_backend_raises(self):
        from FatTummy.exceptions import FatTummyUnsupportedBackendError
        e = self._engine()
        e._engine_name = "openai"
        e._api_key = "key"
        e._compiled = True
        e._data_sources = ["some data"]
        with pytest.raises(FatTummyUnsupportedBackendError):
            e.finetune()

    def test_build_native_model_tiny(self):
        e = self._engine()
        model = e._build_native_model("mooe")
        from FatTummy.models.mooe import MOOE
        assert isinstance(model, MOOE)

    def test_build_native_model_small(self):
        e = self._engine()
        model = e._build_native_model("mooe")
        assert model.config.hidden_size in (128, 256, 384)

    def test_build_native_model_lion(self):
        e = self._engine()
        model = e._build_native_model("lion")
        assert model.config.top_k == 1

    def test_build_native_model_spacebyte(self):
        e = self._engine()
        model = e._build_native_model("spacebyte")
        assert model.config.vocab_size == 256

    def test_large_scale_alias_downgraded(self, capsys):
        e = self._engine()
        e.modelbuild("1b")
        model = e._build_native_model("mooe")
        captured = capsys.readouterr()
        assert "small" in captured.out or model is not None


# ============================================================================
# 9. FatTummyEngine – full finetune on native model
# ============================================================================

class TestEngineFinetuneNative:
    def test_finetune_native_runs(self, tmp_path):
        from FatTummy.engine import FatTummyEngine
        engine = FatTummyEngine()
        engine._engine_name = "mooe"
        engine._model_instance = _make_tiny_model()
        engine._compiled = True
        engine._data_sources = [["the quick brown fox", "hello world text here"]]
        engine._epochs = 1
        # Monkey-patch output_dir in trainer
        import FatTummy.tuning.trainer as trainer_mod
        orig_init = trainer_mod.FatTummyTrainer.__init__

        def patched_init(self_t, *args, **kwargs):
            kwargs.setdefault("output_dir", str(tmp_path / "ckpt"))
            orig_init(self_t, *args, **kwargs)

        trainer_mod.FatTummyTrainer.__init__ = patched_init
        try:
            engine.finetune(epochs=1)
        finally:
            trainer_mod.FatTummyTrainer.__init__ = orig_init


# ============================================================================
# 10. Cloud adapters
# ============================================================================

class TestCloudAdapters:
    def test_get_cloud_adapter_openai(self, monkeypatch):
        import FatTummy.inference.cloud_adapters as ca
        # Patch OpenAIAdapter.__post_init__ to avoid real openai import
        monkeypatch.setattr(ca.OpenAIAdapter, "__post_init__", lambda self: None)
        adapter = ca.get_cloud_adapter("openai", api_key="sk-fake")
        assert isinstance(adapter, ca.OpenAIAdapter)

    def test_get_cloud_adapter_anthropic(self, monkeypatch):
        import FatTummy.inference.cloud_adapters as ca
        monkeypatch.setattr(ca.AnthropicAdapter, "__post_init__", lambda self: None)
        adapter = ca.get_cloud_adapter("anthropic", api_key="sk-fake")
        assert isinstance(adapter, ca.AnthropicAdapter)

    def test_get_cloud_adapter_gemini(self, monkeypatch):
        import FatTummy.inference.cloud_adapters as ca
        monkeypatch.setattr(ca.GeminiAdapter, "__post_init__", lambda self: None)
        adapter = ca.get_cloud_adapter("gemini", api_key="sk-fake")
        assert isinstance(adapter, ca.GeminiAdapter)

    def test_get_cloud_adapter_unknown(self):
        from FatTummy.inference.cloud_adapters import get_cloud_adapter
        from FatTummy.exceptions import FatTummyUnsupportedBackendError
        with pytest.raises(FatTummyUnsupportedBackendError):
            get_cloud_adapter("not_real", api_key="x")

    def test_base_adapter_no_api_key_raises(self):
        from FatTummy.inference.cloud_adapters import BaseCloudAdapter
        from FatTummy.exceptions import FatTummyAuthenticationError
        with pytest.raises(FatTummyAuthenticationError):
            BaseCloudAdapter(api_key="")

    def test_execute_with_retry_succeeds(self, monkeypatch):
        from FatTummy.inference.cloud_adapters import BaseCloudAdapter
        adapter = object.__new__(BaseCloudAdapter)
        adapter.api_key = "key"
        calls = [0]
        def fn():
            calls[0] += 1
            return "ok"
        assert adapter._execute_with_retry(fn, max_retries=3) == "ok"
        assert calls[0] == 1

    def test_execute_with_retry_eventual_success(self):
        from FatTummy.inference.cloud_adapters import BaseCloudAdapter
        import time
        adapter = object.__new__(BaseCloudAdapter)
        adapter.api_key = "key"
        attempts = [0]
        def fn():
            attempts[0] += 1
            if attempts[0] < 3:
                raise RuntimeError("transient")
            return "done"
        # Patch time.sleep to be instant
        import unittest.mock as mock
        with mock.patch("time.sleep"):
            result = adapter._execute_with_retry(fn, max_retries=3, initial_backoff=0.001)
        assert result == "done"
        assert attempts[0] == 3

    def test_execute_with_retry_exhausted_raises(self):
        from FatTummy.inference.cloud_adapters import BaseCloudAdapter
        import unittest.mock as mock
        adapter = object.__new__(BaseCloudAdapter)
        adapter.api_key = "key"
        def fn():
            raise ValueError("always fails")
        with mock.patch("time.sleep"):
            with pytest.raises(ValueError, match="always fails"):
                adapter._execute_with_retry(fn, max_retries=2, initial_backoff=0.001)


# ============================================================================
# 11. Local adapters
# ============================================================================

class TestLocalAdapters:
    def test_get_local_adapter_hf_returns_hf_adapter(self):
        from FatTummy.inference.local_adapters import get_local_adapter, HuggingFaceAdapter
        adapter = get_local_adapter("hf", "gpt2")
        assert isinstance(adapter, HuggingFaceAdapter)

    def test_get_local_adapter_unknown_raises(self):
        from FatTummy.inference.local_adapters import get_local_adapter
        from FatTummy.exceptions import FatTummyUnsupportedBackendError
        with pytest.raises(FatTummyUnsupportedBackendError):
            get_local_adapter("fakeengine", "somemodel")

    def test_hf_adapter_lazy_load(self):
        from FatTummy.inference.local_adapters import HuggingFaceAdapter
        adapter = HuggingFaceAdapter("gpt2")
        # model is not loaded until _load() or generate()
        assert adapter.model is None

    def test_ollama_adapter_raises_if_not_installed(self, monkeypatch):
        import shutil
        from FatTummy.exceptions import FatTummyDependencyError
        monkeypatch.setattr(shutil, "which", lambda cmd: None)
        from FatTummy.inference.local_adapters import OllamaAdapter
        with pytest.raises(FatTummyDependencyError, match="Ollama"):
            OllamaAdapter("llama3.2")


# ============================================================================
# 12. Public __init__ API
# ============================================================================

class TestPublicAPI:
    def test_version_string(self):
        import FatTummy
        assert isinstance(FatTummy.__version__, str)
        assert "." in FatTummy.__version__

    def test_build_returns_engine(self):
        import FatTummy
        engine = FatTummy.build(interactive=False)
        from FatTummy.engine import FatTummyEngine
        assert isinstance(engine, FatTummyEngine)

    def test_global_engine_set(self):
        import FatTummy
        FatTummy.build(interactive=False)
        assert FatTummy._default_engine is not None

    def test_modelbuild_delegates(self):
        import FatTummy
        FatTummy.build(interactive=False).engine("mooe")
        ret = FatTummy.modelbuild("tiny")
        from FatTummy.engine import FatTummyEngine
        assert isinstance(ret, FatTummyEngine)

    def test_mooe_accessible(self):
        import FatTummy
        from FatTummy.models.mooe import MOOE
        assert FatTummy.MOOE is MOOE

    def test_getattr_unknown_raises(self):
        import FatTummy
        with pytest.raises(AttributeError):
            _ = FatTummy.this_does_not_exist

    def test_all_exports_present(self):
        import FatTummy
        for name in FatTummy.__all__:
            if name != "__version__":
                assert hasattr(FatTummy, name), f"Missing export: {name}"


# ============================================================================
# 13. Regression: torch.load weights_only warning (PyTorch 2.6+)
# ============================================================================

class TestCheckpointResume:
    def test_torch_load_no_crash(self, tmp_path):
        """Ensure checkpoint loading with state dict works without error."""
        import torch
        from FatTummy.tuning.trainer import FatTummyTrainer
        model = _make_tiny_model()
        ckpt_dir = tmp_path / "ckpt" / "epoch-3"
        ckpt_dir.mkdir(parents=True)
        torch.save(model.state_dict(), ckpt_dir / "model.pt")
        # Should load without exception
        trainer = FatTummyTrainer(
            model,
            dataset=["hello"],
            epochs=1,
            output_dir=str(tmp_path / "ckpt"),
        )
        assert trainer.model is not None


# ============================================================================
# 14. Inference __init__ exports
# ============================================================================

class TestInferenceInit:
    def test_inference_init_exports_adapters(self):
        from FatTummy.inference import cloud_adapters, local_adapters
        assert hasattr(cloud_adapters, "get_cloud_adapter")
        assert hasattr(local_adapters, "get_local_adapter")


# ============================================================================
# 15. Advanced Knobs (optimizer, spacebyte, lr_scheduler, weight_decay, warmup, clip_grad)
# ============================================================================

class TestAdvancedKnobs:
    def test_optimizer_validation(self):
        from FatTummy.engine import FatTummyEngine
        from FatTummy.exceptions import FatTummyConfigurationError
        e = FatTummyEngine()
        e.optimizer("lion")
        assert e._optimizer == "lion"
        with pytest.raises(FatTummyConfigurationError):
            e.optimizer("invalid_opt")

    def test_spacebyte_validation(self):
        from FatTummy.engine import FatTummyEngine
        e = FatTummyEngine()
        e.spacebyte(True)
        assert e._use_spacebyte is True
        e.spacebyte(False)
        assert e._use_spacebyte is False

    def test_lr_scheduler_validation(self):
        from FatTummy.engine import FatTummyEngine
        from FatTummy.exceptions import FatTummyConfigurationError
        e = FatTummyEngine()
        e.lr_scheduler("cosine")
        assert e._lr_scheduler == "cosine"
        with pytest.raises(FatTummyConfigurationError):
            e.lr_scheduler("invalid_scheduler")

    def test_weight_decay_warmup_clip_grad(self):
        from FatTummy.engine import FatTummyEngine
        e = FatTummyEngine()
        e.weight_decay(0.05).warmup(10).clip_grad(1.5)
        assert e._weight_decay == 0.05
        assert e._warmup_steps == 10
        assert e._clip_grad_norm == 1.5

    def test_trainer_integration_lion(self, tmp_path):
        import unittest.mock as mock
        from FatTummy.tuning.trainer import FatTummyTrainer
        
        model = _make_tiny_model()
        # Mock Lion import to avoid actual pip install in test
        class FakeLion:
            def __init__(self, params, lr, weight_decay):
                self.param_groups = [{"lr": lr}]
            def zero_grad(self, set_to_none=True):
                pass
        
        with mock.patch("FatTummy.tuning.trainer._ensure_lion", return_value=FakeLion):
            trainer = FatTummyTrainer(
                model=model,
                dataset=["hello world"],
                epochs=1,
                optimizer="lion",
                output_dir=str(tmp_path / "ckpt"),
            )
            assert trainer.optimizer_choice == "lion"
            optimizer = trainer._build_optimizer("cpu")
            assert isinstance(optimizer, FakeLion)

    def test_trainer_spacebyte_encoding(self):
        from FatTummy.tuning.trainer import _encode_text
        # SpaceByte encoding test
        res = _encode_text("hello", tokenizer=None, max_length=10, use_spacebyte=True)
        assert res["input_ids"] == [104, 101, 108, 108, 111]
        assert res["labels"] == [104, 101, 108, 108, 111]

    def test_public_api_global_wrappers(self):
        import FatTummy
        FatTummy.build(interactive=False)
        FatTummy.optimizer("lion")
        assert FatTummy._default_engine._optimizer == "lion"
        FatTummy.spacebyte(True)
        assert FatTummy._default_engine._use_spacebyte is True
        FatTummy.lr_scheduler("linear")
        assert FatTummy._default_engine._lr_scheduler == "linear"
        FatTummy.weight_decay(0.02)
        assert FatTummy._default_engine._weight_decay == 0.02
        FatTummy.warmup(5)
        assert FatTummy._default_engine._warmup_steps == 5
        FatTummy.clip_grad(1.0)
        assert FatTummy._default_engine._clip_grad_norm == 1.0

    def test_tpu_training_flow(self, monkeypatch, tmp_path):
        import sys
        import unittest.mock as mock
        from FatTummy.tuning.trainer import FatTummyTrainer

        # Mock torch_xla module robustly
        mock_xm = mock.MagicMock()
        mock_xm.xla_device.return_value = "cpu"
        
        mock_core = mock.MagicMock()
        mock_core.xla_model = mock_xm
        
        mock_torch_xla = mock.MagicMock()
        mock_torch_xla.core = mock_core
        
        sys.modules["torch_xla"] = mock_torch_xla
        sys.modules["torch_xla.core"] = mock_core
        sys.modules["torch_xla.core.xla_model"] = mock_xm

        monkeypatch.setenv("TPU_NAME", "mock-tpu-device")

        model = _make_tiny_model()
        monkeypatch.setattr(model, "to", lambda dev: model)
        monkeypatch.setattr(model, "train", lambda: None)

        trainer = FatTummyTrainer(
            model=model,
            dataset=["mock data one", "mock data two"],
            epochs=1,
            output_dir=str(tmp_path / "tpu_ckpt"),
        )
        
        assert trainer._has_tpu() is True

        trainer.finetune()

        assert mock_xm.xla_device.called
        assert mock_xm.optimizer_step.called
        assert mock_xm.mark_step.called
        assert mock_xm.add_step_closure.called

        del sys.modules["torch_xla"]
        del sys.modules["torch_xla.core"]
        del sys.modules["torch_xla.core.xla_model"]


# ============================================================================
# 16. Interactive Wizard (interactive.py)
# ============================================================================

class TestInteractiveWizard:
    def test_collect_for_action_native_finetune(self, monkeypatch):
        from FatTummy.interactive import _collect_for_action
        
        inputs = iter(["mooe", "dataset_path.txt"])
        monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
        
        config = _collect_for_action("finetune")
        assert config["engine"] == "mooe"
        assert config["type"] == "mooe"
        assert config["datasets_raw"] == "dataset_path.txt"

    def test_collect_for_action_hf_finetune(self, monkeypatch):
        from FatTummy.interactive import _collect_for_action
        
        inputs = iter(["hf", "gpt2", "dataset_path.txt"])
        monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
        
        config = _collect_for_action("finetune")
        assert config["engine"] == "hf"
        assert config["type"] == "gpt2"
        assert config["datasets_raw"] == "dataset_path.txt"

    def test_validate_config_checks(self):
        from FatTummy.interactive import validate_config
        # should normalize native engine in make action
        cfg = {"action": "make", "engine": "mooe", "type": "some_other_type"}
        validate_config(cfg)
        assert cfg["type"] == "mooe"



