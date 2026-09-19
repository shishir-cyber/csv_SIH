"""
schema.py
---------
Receipt Schema Definition for CV-ASSURE M3 Inference Provenance.
Defines mandatory fields for Clause 2.2.3 compliant inference receipts.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
import uuid
from datetime import datetime, timezone


@dataclass
class InferenceReceiptSchema:
    """
    Mandatory schema fields for Clause 2.2.3 inference receipts.
    """

    input_image_sha256: str
    preprocess_config_hash: str
    model_weight_digest: str
    model_arch_hash: str
    runtime_version: str
    output_payload_sha256: str
    nonce: str
    sequence_number: int
    timestamp_utc: str
    prev_receipt_hash: str = "0" * 64
    image_path: Optional[str] = None

    def validate(self) -> bool:
        """Validate presence and format of required schema fields."""
        if not self.input_image_sha256 or len(self.input_image_sha256) != 64:
            raise ValueError("input_image_sha256 must be a valid 64-character SHA-256 hex string")
        if not self.model_weight_digest or len(self.model_weight_digest) != 64:
            raise ValueError("model_weight_digest must be a valid 64-character SHA-256 hex string")
        if not self.output_payload_sha256 or len(self.output_payload_sha256) != 64:
            raise ValueError("output_payload_sha256 must be a valid 64-character SHA-256 hex string")
        if self.sequence_number < 1:
            raise ValueError("sequence_number must be >= 1")
        if not self.nonce:
            raise ValueError("nonce cannot be empty")
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert dataclass to dictionary."""
        d = asdict(self)
        if d.get("image_path") is None:
            d.pop("image_path", None)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InferenceReceiptSchema":
        """Construct schema object from dictionary."""
        return cls(
            input_image_sha256=data["input_image_sha256"],
            preprocess_config_hash=data["preprocess_config_hash"],
            model_weight_digest=data["model_weight_digest"],
            model_arch_hash=data["model_arch_hash"],
            runtime_version=data["runtime_version"],
            output_payload_sha256=data["output_payload_sha256"],
            nonce=data["nonce"],
            sequence_number=data["sequence_number"],
            timestamp_utc=data["timestamp_utc"],
            prev_receipt_hash=data.get("prev_receipt_hash", "0" * 64),
            image_path=data.get("image_path"),
        )
