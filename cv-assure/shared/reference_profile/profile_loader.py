"""Reference-profile YAML loader."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

from shared.crypto.hashing import sha256_file
from shared.crypto.verifier import verify_hex

PathLike = Union[str, Path]


def load_reference_profile(path: PathLike) -> Dict[str, Any]:
    """Load a reference-profile YAML/JSON document as a dict."""
    profile_path = Path(path)
    text = profile_path.read_text(encoding="utf-8")
    if profile_path.suffix.lower() in {".json"}:
        data = json.loads(text)
    else:
        data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"Reference profile must be a mapping: {profile_path}")
    return data


def canonical_profile_bytes(profile: Dict[str, Any]) -> bytes:
    """Canonical JSON of a profile with the signature field stripped.

    The attestation envelope (algorithm, public key path, etc.) remains so the
    signature covers the claimed identity of the signer.
    """
    stripped = copy.deepcopy(profile)
    attestation = stripped.get("attestation")
    if isinstance(attestation, dict):
        attestation.pop("signature", None)
    return json.dumps(stripped, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def default_public_key_path() -> Path:
    """Project-level Ed25519 public key used when a profile does not name one."""
    return Path(__file__).resolve().parents[1] / "keys" / "public_key.pem"


def verify_profile_attestation(
    profile_path: PathLike,
    public_key_path: Optional[PathLike] = None,
) -> bool:
    """Return True if ``profile_path`` carries a valid Ed25519 attestation."""
    path = Path(profile_path)
    if not path.is_file():
        return False

    profile = load_reference_profile(path)
    attestation = profile.get("attestation") or {}
    if not isinstance(attestation, dict):
        attestation = {}

    signature_hex = attestation.get("signature")
    if not signature_hex:
        detached = path.with_name(path.name + ".sig")
        if detached.is_file():
            signature_hex = detached.read_text(encoding="utf-8").strip()
    if not signature_hex:
        return False

    key_path = public_key_path or attestation.get("public_key_path") or attestation.get("public_key")
    if key_path:
        key_file = Path(str(key_path))
        if not key_file.is_file():
            key_file = (path.parent / key_file).resolve()
    else:
        key_file = default_public_key_path()
    if not key_file.is_file():
        return False

    payload = canonical_profile_bytes(profile)
    return verify_hex(payload, str(signature_hex), key_file)


def profile_digest(profile_path: PathLike) -> str:
    """SHA-256 of the on-disk reference profile file."""
    return sha256_file(profile_path)
