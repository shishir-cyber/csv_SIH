"""
signer.py
---------
Ed25519 Cryptographic Signer for CV-ASSURE M3 Inference Receipts (Clause 2.2.3).
Signs canonical receipt digests using an attested Ed25519 private key.
"""

from pathlib import Path
from typing import Dict, Any, Union
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519


class ReceiptSigner:
    """
    Ed25519 Receipt Signer.
    """

    def __init__(self, private_key: Union[bytes, str, Path, ed25519.Ed25519PrivateKey]):
        if isinstance(private_key, ed25519.Ed25519PrivateKey):
            self._private_key = private_key
        else:
            if isinstance(private_key, (str, Path)):
                pem_bytes = Path(private_key).read_bytes()
            else:
                pem_bytes = private_key

            self._private_key = serialization.load_pem_private_key(pem_bytes, password=None)

    def sign_hash(self, receipt_hash: str) -> str:
        """
        Sign a hex-encoded receipt hash using Ed25519.
        Returns a hex-encoded signature string.
        """
        sig_bytes = self._private_key.sign(receipt_hash.encode("utf-8"))
        return sig_bytes.hex()

    def sign_receipt(self, receipt_hash: str, receipt_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Attach signature to receipt structure.
        """
        sig_hex = self.sign_hash(receipt_hash)
        return {
            "record_type": "receipt",
            "payload": receipt_payload,
            "prev_receipt_hash": receipt_payload.get("prev_receipt_hash", "0" * 64),
            "receipt_hash": receipt_hash,
            "signature": sig_hex,
        }
