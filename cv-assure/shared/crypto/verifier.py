"""Ed25519 verification helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Union

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import load_pem_public_key

PathLike = Union[str, Path]


def load_public_key(path: PathLike) -> Ed25519PublicKey:
    """Load an Ed25519 public key from a PEM file."""
    key = load_pem_public_key(Path(path).read_bytes())
    if not isinstance(key, Ed25519PublicKey):
        raise TypeError(f"Expected Ed25519 public key in {path}")
    return key


def verify_bytes(payload: bytes, signature: bytes, public_key_path: PathLike) -> bool:
    """Return True if ``signature`` is a valid Ed25519 signature of ``payload``."""
    try:
        load_public_key(public_key_path).verify(signature, payload)
        return True
    except (InvalidSignature, ValueError, TypeError):
        return False


def verify_hex(payload: bytes, signature_hex: str, public_key_path: PathLike) -> bool:
    """Verify a hex-encoded Ed25519 signature."""
    try:
        signature = bytes.fromhex(signature_hex.strip())
    except ValueError:
        return False
    return verify_bytes(payload, signature, public_key_path)
