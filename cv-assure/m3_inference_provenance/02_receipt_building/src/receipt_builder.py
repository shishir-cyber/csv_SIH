"""
receipt_builder.py
-------------------
Receipt Builder module assembling raw inference artifacts from Step 1
into canonical, schema-validated Clause 2.2.3 inference receipts.
"""

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional

from schema import InferenceReceiptSchema
from canonical_serialize import compute_canonical_receipt_hash, canonical_json_bytes


class ReceiptBuilder:
    """
    Assembles, validates, and serializes inference receipts.
    """

    def __init__(self, runtime_version: str = "onnxruntime-1.30.0"):
        self.runtime_version = runtime_version

    def build_receipt(
        self,
        input_bytes: bytes,
        output_bytes: bytes,
        model_bytes: bytes,
        preprocess_config: Dict[str, Any],
        sequence_number: int,
        prev_receipt_hash: str = "0" * 64,
        image_path: Optional[str] = None,
        model_arch_name: str = "resnet18_onnx",
    ) -> Dict[str, Any]:
        """
        Assemble a canonical inference receipt payload dictionary.

        Args:
            input_bytes: Raw bytes of the input image file.
            output_bytes: Raw bytes of the ONNX inference output tensor.
            model_bytes: Raw bytes of the ONNX model file.
            preprocess_config: Preprocessing configuration dictionary.
            sequence_number: Monotonic 1-indexed sequence counter.
            prev_receipt_hash: SHA-256 hash of previous receipt.
            image_path: Optional filename/path.
            model_arch_name: Architecture descriptor.

        Returns:
            Validated receipt payload dict.
        """
        input_sha256 = hashlib.sha256(input_bytes).hexdigest()
        output_sha256 = hashlib.sha256(output_bytes).hexdigest()
        model_weight_digest = hashlib.sha256(model_bytes).hexdigest()
        model_arch_hash = hashlib.sha256(model_arch_name.encode("utf-8")).hexdigest()
        config_hash = hashlib.sha256(canonical_json_bytes(preprocess_config)).hexdigest()

        nonce = f"nonce-{sequence_number:04d}-{input_sha256[:8]}"
        timestamp_utc = datetime.now(timezone.utc).isoformat()

        receipt_obj = InferenceReceiptSchema(
            input_image_sha256=input_sha256,
            preprocess_config_hash=config_hash,
            model_weight_digest=model_weight_digest,
            model_arch_hash=model_arch_hash,
            runtime_version=self.runtime_version,
            output_payload_sha256=output_sha256,
            nonce=nonce,
            sequence_number=sequence_number,
            timestamp_utc=timestamp_utc,
            prev_receipt_hash=prev_receipt_hash,
            image_path=image_path,
        )

        # Validate schema rules
        receipt_obj.validate()

        return receipt_obj.to_dict()

    def get_receipt_hash(self, payload: Dict[str, Any], prev_receipt_hash: str) -> str:
        """Return the canonical SHA-256 digest of a receipt payload."""
        return compute_canonical_receipt_hash(payload, prev_receipt_hash)
