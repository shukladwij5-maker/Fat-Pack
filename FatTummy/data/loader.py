"""Dataset resolution for local files and Hugging Face datasets."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

from ..exceptions import FatTummyDatasetError, FatTummyDependencyError, FatTummyNetworkError

STREAM_SIZE_BYTES = 500 * 1024 * 1024
SUPPORTED_LOCAL_EXTENSIONS = {".json", ".jsonl", ".csv", ".txt"}
HF_REPO_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _split_sources(raw_sources: Any) -> List[str]:
    """Normalize comma-separated strings, paths, and source sequences."""
    if raw_sources is None:
        return []
    if isinstance(raw_sources, (str, os.PathLike)):
        return [part.strip() for part in os.fspath(raw_sources).split(",") if part.strip()]
    if isinstance(raw_sources, Sequence):
        sources: List[str] = []
        for item in raw_sources:
            sources.extend(_split_sources(item))
        return sources
    raise FatTummyDatasetError(f"Unsupported dataset source type: {type(raw_sources).__name__}")


def _looks_like_path(source: str) -> bool:
    """Return True when a source resembles a local path rather than a repo ID."""
    path = Path(source)
    return (
        path.suffix.lower() in SUPPORTED_LOCAL_EXTENSIONS
        or (path.suffix and "/" not in source)
        or source.startswith((".", "~"))
        or "\\" in source
        or os.path.isabs(source)
    )


def _validate_local_file(path: Path) -> None:
    """Validate a supported local dataset file."""
    if not path.exists():
        raise FatTummyDatasetError(f"Dataset file not found: {path}")
    if not path.is_file():
        raise FatTummyDatasetError(f"Dataset source is not a file: {path}")
    if path.suffix.lower() not in SUPPORTED_LOCAL_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_LOCAL_EXTENSIONS))
        raise FatTummyDatasetError(f"Unsupported dataset extension '{path.suffix}'. Supported: {supported}")


def _validate_hf_repo_id(repo_id: str) -> None:
    """Validate a Hugging Face dataset repo ID of the form namespace/name."""
    if not HF_REPO_RE.match(repo_id):
        raise FatTummyDatasetError(
            f"Malformed Hugging Face dataset repo ID '{repo_id}'. Expected 'namespace/name'."
        )


def _parse_hf_source(source: str) -> tuple[str, Optional[str]]:
    """Parse HF source strings like namespace/repo, namespace/repo/config, or namespace/repo:config."""
    if ":" in source:
        repo, config = source.split(":", 1)
    else:
        parts = source.split("/")
        if len(parts) < 2:
            raise FatTummyDatasetError(
                f"Malformed Hugging Face dataset source '{source}'. Expected 'namespace/name' or 'namespace/name/config'."
            )
        repo = "/".join(parts[:2])
        config = "/".join(parts[2:]) if len(parts) > 2 else None

    _validate_hf_repo_id(repo)
    return repo, config


def _require_datasets_package() -> Any:
    """Import datasets.load_dataset or raise a clear optional dependency error."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise FatTummyDependencyError(
            "Dataset loading requires the optional 'datasets' package. "
            "Install it with: pip install datasets huggingface_hub"
        ) from exc
    return load_dataset


def _hf_dataset_size_bytes(repo_id: str, token: Optional[str] = None) -> Optional[int]:
    """Return a best-effort byte size for a Hugging Face dataset repository."""
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise FatTummyDependencyError(
            "Hugging Face dataset size checks require 'huggingface_hub'. "
            "Install it with: pip install huggingface_hub"
        ) from exc

    try:
        info = HfApi(token=token).dataset_info(repo_id)
    except Exception as exc:
        raise FatTummyNetworkError("huggingface_dataset_info", original_error=exc) from exc

    total = 0
    known = False
    for sibling in info.siblings or []:
        size = getattr(sibling, "size", None)
        if size is not None:
            known = True
            total += int(size)
    return total if known else None


def _load_local(path: Path, streaming: bool) -> Any:
    """Load a local dataset through the datasets library."""
    load_dataset = _require_datasets_package()
    ext = path.suffix.lower()
    builder = {"json": "json", "jsonl": "json", "csv": "csv", "txt": "text"}[ext.lstrip(".")]
    try:
        return load_dataset(builder, data_files=str(path), streaming=streaming)
    except Exception as exc:
        raise FatTummyDatasetError(f"Failed to load local dataset '{path}': {exc}") from exc


def _load_hf(repo_id: str, config: Optional[str], streaming: bool, token: Optional[str] = None) -> Any:
    """Load a Hugging Face dataset repository, optionally with a config name."""
    load_dataset = _require_datasets_package()
    try:
        return load_dataset(repo_id, name=config, streaming=streaming, token=token)
    except Exception as exc:
        tag = f"/{config}" if config else ""
        raise FatTummyDatasetError(
            f"Failed to load Hugging Face dataset '{repo_id}{tag}': {exc}"
        ) from exc


def load_dataset_source(source: str, token: Optional[str] = None) -> Tuple[Any, str]:
    """Load one dataset source and return ``(dataset, mode)``.

    Modes are ``local`` for local files below 500 MB, ``streaming`` for local or
    Hugging Face sources at/above 500 MB, and ``download`` for smaller HF repos.
    """
    source = source.strip()
    if not source:
        raise FatTummyDatasetError("Dataset source cannot be empty.")

    if _looks_like_path(source):
        path = Path(source).expanduser()
        _validate_local_file(path)
        streaming = path.stat().st_size >= STREAM_SIZE_BYTES
        mode = "streaming" if streaming else "local"
        return _load_local(path, streaming=streaming), mode

    repo_id, config = _parse_hf_source(source)
    size = _hf_dataset_size_bytes(repo_id, token=token)
    streaming = True if size is None else size >= STREAM_SIZE_BYTES
    mode = "streaming" if streaming else "download"
    return _load_hf(repo_id, config=config, streaming=streaming, token=token), mode


def resolve_datasets(raw_sources: Any, token: Optional[str] = None) -> Tuple[List[Any], List[str]]:
    """Resolve comma-separated dataset sources into loaded datasets and modes."""
    datasets: List[Any] = []
    modes: List[str] = []
    for source in _split_sources(raw_sources):
        dataset, mode = load_dataset_source(source, token=token)
        datasets.append(dataset)
        modes.append(mode)
    return datasets, modes
