"""
canonical_serialize.py
----------------------
Deterministic hashing & serialization logic for CV-ASSURE M3 Inference Provenance.
Ensures reproducible byte representation across heterogeneous platforms.
"""

import json
import hashlib
from typing import Dict, Any


def canonical_json_bytes(data: Dict[str, Any]) -> bytes:
    """
    Return deterministically formatted, UTF-8 encoded JSON bytes
    (sorted keys, compact separators without whitespace).
    """
    return json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")


def canonical_cbor_bytes(data: Dict[str, Any]) -> bytes:
    """
    Return CBOR encoded bytes if cbor2 is installed, otherwise fallback to canonical JSON bytes.
    """
    try:
        import cbor2
        return cbor2.dumps(data, canonical=True)
    except ImportError:
        return canonical_json_bytes(data)


def compute_canonical_receipt_hash(payload: Dict[str, Any], prev_receipt_hash: str) -> str:
    """
    Compute SHA-256 digest over receipt payload + prev_receipt_hash binding.
    """
    binding_obj = {
        "payload": payload,
        "prev_receipt_hash": prev_receipt_hash,
    }
    canonical_bytes = canonical_json_bytes(binding_obj)
    return hashlib.sha256(canonical_bytes).hexdigest()
