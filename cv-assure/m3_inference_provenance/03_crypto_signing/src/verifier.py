"""
verifier.py
-----------
Ed25519 Cryptographic Signature Verifier for CV-ASSURE M3 Inference Receipts.
Validates Ed25519 signatures against attested public keys.
"""

from pathlib import Path
from typing import Union, Dict, Any
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.exceptions import InvalidSignature


class ReceiptVerifier:
    """
    Ed25519 Receipt Signature Verifier.
    """

    def __init__(self, public_key: Union[bytes, str, Path, ed25519.Ed25519PublicKey]):
        if isinstance(public_key, ed25519.Ed25519PublicKey):
            self._public_key = public_key
        else:
            if isinstance(public_key, (str, Path)):
                pem_bytes = Path(public_key).read_bytes()
            else:
                pem_bytes = public_key

            self._public_key = serialization.load_pem_public_key(pem_bytes)

    def verify_signature(self, receipt_hash: str, signature_hex: str) -> bool:
        """
        Verify that signature_hex is a valid Ed25519 signature over receipt_hash.
        Returns True if valid, False otherwise.
        """
        try:
            sig_bytes = bytes.fromhex(signature_hex)
            self._public_key.verify(sig_bytes, receipt_hash.encode("utf-8"))
            return True
        except (InvalidSignature, ValueError, TypeError):
            return False

    def verify_receipt(self, signed_receipt_record: Dict[str, Any]) -> bool:
        """
        Verify a signed receipt record dict.
        """
        receipt_hash = signed_receipt_record.get("receipt_hash", "")
        sig_hex = signed_receipt_record.get("signature", "")
        if not receipt_hash or not sig_hex:
            return False
        return self.verify_signature(receipt_hash, sig_hex)
