"""Artifact storage utilities for container runners.

V1: local filesystem only. S3/NFS support is future work.
"""

from __future__ import annotations

import shutil
from pathlib import Path


def save_to_local(name: str, source: Path, storage_dir: Path) -> Path:
    dest = storage_dir / name
    if source.is_dir():
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(source, dest)
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
    return dest
