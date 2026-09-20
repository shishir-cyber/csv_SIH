"""Ed25519 signing helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Union

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    PublicFormat,
)

PathLike = Union[str, Path]


def load_private_key(path: PathLike) -> Ed25519PrivateKey:
    """Load an Ed25519 private key from a PEM file."""
    from cryptography.hazmat.primitives.serialization import load_pem_private_key

    pem = Path(path).read_bytes()
    key = load_pem_private_key(pem, password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise TypeError(f"Expected Ed25519 private key in {path}")
    return key


def sign_bytes(payload: bytes, private_key_path: PathLike) -> bytes:
    """Return a raw 64-byte Ed25519 signature over ``payload``."""
    return load_private_key(private_key_path).sign(payload)


def sign_hex(payload: bytes, private_key_path: PathLike) -> str:
    """Return the hex-encoded Ed25519 signature over ``payload``."""
    return sign_bytes(payload, private_key_path).hex()


def write_keypair(private_key_path: PathLike, public_key_path: PathLike) -> None:
    """Generate and write a new Ed25519 PEM key pair."""
    private_key = Ed25519PrivateKey.generate()
    Path(private_key_path).parent.mkdir(parents=True, exist_ok=True)
    Path(private_key_path).write_bytes(
        private_key.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption())
    )
    Path(public_key_path).write_bytes(
        private_key.public_key().public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)
    )
