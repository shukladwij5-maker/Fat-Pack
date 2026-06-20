"""Load datasets from HuggingFace repos or local files with size-aware streaming."""

import os
from typing import Optional, Tuple

# Datasets larger than this use streaming instead of a full download.
STREAM_SIZE_BYTES = 500 * 1024 * 1024


def _is_local_file(source: str) -> bool:
    return os.path.isfile(source)


def _is_hf_repo(source: str) -> bool:
    if _is_local_file(source):
        return False
    if len(source) > 1 and source[1] == ":":
        return False
    if source.startswith((".", os.path.sep)):
        return False
    return "/" in source


def _hf_dataset_size_bytes(repo_id: str, token: str = None) -> Optional[int]:
    try:
        from huggingface_hub import HfApi

        info = HfApi(token=token).dataset_info(repo_id)
        siblings = info.siblings or []
        return sum(getattr(entry, "size", 0) or 0 for entry in siblings)
    except Exception:
        return None


def _load_hf(repo_id: str, streaming: bool, token: str = None):
    from datasets import load_dataset

    return load_dataset(repo_id, streaming=streaming, token=token)


def _load_local(path: str, streaming: bool):
    from datasets import load_dataset

    ext = os.path.splitext(path)[1].lower()
    if ext in (".json", ".jsonl"):
        builder = "json"
    elif ext == ".csv":
        builder = "csv"
    elif ext == ".txt":
        builder = "text"
    else:
        builder = "json"

    return load_dataset(builder, data_files=path, streaming=streaming)


def load_dataset_source(source: str, token: str = None):
    """
    Load one dataset source. Returns (dataset, mode) where mode is
    'stream' or 'download'. Returns (None, None) for empty input.
    """
    source = source.strip()
    if not source:
        return None, None

    if _is_local_file(source):
        size = os.path.getsize(source)
        streaming = size > STREAM_SIZE_BYTES
        mode = "stream" if streaming else "download"
        print(f"  Dataset '{source}' ({_format_size(size)}) -> {mode}")
        return _load_local(source, streaming=streaming), mode

    if _is_hf_repo(source):
        size = _hf_dataset_size_bytes(source, token)
        if size is None:
            streaming = True
            print(f"  Dataset '{source}' (size unknown) -> stream")
        else:
            streaming = size > STREAM_SIZE_BYTES
            print(f"  Dataset '{source}' ({_format_size(size)}) -> "
                  f"{'stream' if streaming else 'download'}")
        mode = "stream" if streaming else "download"
        return _load_hf(source, streaming=streaming, token=token), mode

    raise ValueError(
        f"Unrecognized dataset '{source}'. Use a HuggingFace repo (user/name) "
        "or a local file path."
    )


def resolve_datasets(raw_sources: str, token: str = None):
    """Load multiple comma-separated dataset sources."""
    sources = [part.strip() for part in raw_sources.split(",") if part.strip()]
    datasets = []
    modes = []
    for source in sources:
        dataset, mode = load_dataset_source(source, token=token)
        if dataset is not None:
            datasets.append(dataset)
            modes.append(mode)
    return datasets, modes


def _format_size(num_bytes: int) -> str:
    if num_bytes >= 1024 ** 3:
        return f"{num_bytes / 1024 ** 3:.1f} GB"
    if num_bytes >= 1024 ** 2:
        return f"{num_bytes / 1024 ** 2:.1f} MB"
    if num_bytes >= 1024:
        return f"{num_bytes / 1024:.1f} KB"
    return f"{num_bytes} B"
