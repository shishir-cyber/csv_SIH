"""SHA-256 helpers for files and byte strings."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Union

PathLike = Union[str, Path]


def sha256_file(path: PathLike, chunk_size: int = 1024 * 1024) -> str:
    """Return the hex SHA-256 digest of ``path``, reading the file in chunks.

    Args:
        path: File to hash.
        chunk_size: Read size in bytes. Defaults to 1 MiB.

    Raises:
        FileNotFoundError: If ``path`` does not exist or is not a file.
        OSError: If the file cannot be read.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Cannot hash missing file: {file_path}")

    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    """Return the hex SHA-256 digest of an in-memory byte string."""
    return hashlib.sha256(payload).hexdigest()
