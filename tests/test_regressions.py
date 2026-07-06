from __future__ import annotations

import importlib

import pytest


def test_case_insensitive_import_aliases():
    import FatTummy

    fattummy = importlib.import_module("fattummy")
    fattummy_alt = importlib.import_module("Fattummy")

    assert hasattr(fattummy, "build")
    assert hasattr(fattummy_alt, "build")
    assert fattummy.build is FatTummy.build


def test_trainer_iter_texts_handles_mapping_splits():
    from FatTummy.tuning.trainer import FatTummyTrainer
    from FatTummy.models.mooe import MOOE, MOOEConfig

    trainer = FatTummyTrainer(
        model=MOOE(MOOEConfig(hidden_size=32, intermediate_size=64, num_layers=1, num_experts=2, top_k=1, vocab_size=64)),
        dataset={"train": [{"text": "hello"}, {"text": "world"}]},
        epochs=1,
    )

    assert list(trainer._iter_texts(trainer.dataset)) == ["hello", "world"]


def test_blank_hf_tokens_are_normalized_to_none():
    from FatTummy.data.loader import _normalize_hf_token

    assert _normalize_hf_token(None) is None
    assert _normalize_hf_token("") is None
    assert _normalize_hf_token("   ") is None
    assert _normalize_hf_token("token-123") == "token-123"


def test_lion_helper_does_not_auto_install_missing_dependency(monkeypatch):
    import builtins

    from FatTummy.exceptions import FatTummyDependencyError
    from FatTummy.tuning import trainer as trainer_module

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "lion_pytorch":
            raise ImportError("missing")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(FatTummyDependencyError):
        trainer_module._ensure_lion()


def test_predictor_dependencies_do_not_auto_install_missing_packages(monkeypatch):
    from FatTummy import predictor

    monkeypatch.setattr(predictor.importlib.util, "find_spec", lambda name: None)

    assert predictor.ensure_predictor_dependencies() is False
