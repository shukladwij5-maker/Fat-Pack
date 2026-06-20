"""Fast wheel/sdist builder — no setuptools or hatchling required."""

import hashlib
import os
import tarfile
import zipfile
from pathlib import Path

VERSION = "0.1.5"
NAME = "fattummy"
PACKAGE_DIR = "FatTummy"
SKIP_DIRS = {".git", "dist", "build", "__pycache__", ".egg-info", ".cursor"}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _metadata() -> bytes:
    readme = Path("README.md").read_text(encoding="utf-8")
    body = f"""Metadata-Version: 2.1
Name: {NAME}
Version: {VERSION}
Summary: A declarative, ultra-minimalist ML framework for zero-boilerplate hardware-agnostic inference and training.
Author-email: Origin-Labs <Shukladwij5@gmail.com>
License: GPL-3.0
Project-URL: Homepage, https://github.com/shukladwij5-maker/fattummy
Classifier: Programming Language :: Python :: 3
Classifier: License :: OSI Approved :: GNU General Public License v3 (GPLv3)
Classifier: Operating System :: OS Independent
Classifier: Topic :: Scientific/Engineering :: Artificial Intelligence
Requires-Python: >=3.8
Description-Content-Type: text/markdown

{readme}
"""
    return body.encode("utf-8")


def _package_files():
    files = []
    for root, dirs, filenames in os.walk(PACKAGE_DIR):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for filename in filenames:
            if filename.endswith(".pyc"):
                continue
            path = Path(root) / filename
            arc = path.as_posix()
            files.append((arc, path.read_bytes()))
    return files


def build_wheel(dist: Path) -> Path:
    dist_info = f"{NAME}-{VERSION}.dist-info"
    wheel_info = (
        b"Wheel-Version: 1.0\nGenerator: fattummy-build\n"
        b"Root-Is-Purelib: true\nTag: py3-none-any\n"
    )
    metadata = _metadata()

    entries = {}
    records = []

    for arc, data in _package_files():
        entries[arc] = data
        records.append((arc, _sha256(data), len(data)))

    for arc, data in (
        (f"{dist_info}/METADATA", metadata),
        (f"{dist_info}/WHEEL", wheel_info),
    ):
        entries[arc] = data
        records.append((arc, _sha256(data), len(data)))

    record_lines = [f"{arc},sha256={digest},{size}" for arc, digest, size in records]
    record_body = "\n".join(record_lines) + f"\n{dist_info}/RECORD,,\n"
    entries[f"{dist_info}/RECORD"] = record_body.encode("utf-8")

    wheel_path = dist / f"{NAME}-{VERSION}-py3-none-any.whl"
    with zipfile.ZipFile(wheel_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for arc, data in sorted(entries.items()):
            archive.writestr(arc, data)

    print(f"Built {wheel_path}")
    return wheel_path


def build_sdist(dist: Path) -> Path:
    sdist_path = dist / f"{NAME}-{VERSION}.tar.gz"
    prefix = f"{NAME}-{VERSION}"

    with tarfile.open(sdist_path, "w:gz") as archive:
        for root, dirs, filenames in os.walk("."):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for filename in filenames:
                if filename.endswith(".pyc"):
                    continue
                path = Path(root) / filename
                rel = path.as_posix().lstrip("./")
                if rel.startswith("dist/"):
                    continue
                archive.add(path, arcname=f"{prefix}/{rel}")

    print(f"Built {sdist_path}")
    return sdist_path


def main():
    dist = Path("dist")
    dist.mkdir(exist_ok=True)
    build_wheel(dist)
    build_sdist(dist)


if __name__ == "__main__":
    main()
