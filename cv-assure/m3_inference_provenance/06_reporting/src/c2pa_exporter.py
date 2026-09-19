"""
c2pa_exporter.py
----------------
C2PA Manifest Exporter for CV-ASSURE M3 Inference Provenance.

Generates a C2PA-compatible manifest JSON with assertion store and ingredient
references, embedding explicit security caveats regarding C2PA limitations.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional


class C2PAExporter:
    """
    Exports C2PA manifest structure for interoperability.
    """

    C2PA_SECURITY_CAVEAT = (
        "C2PA is a serialization format rather than a security guarantee. "
        "Independent formal methods analysis indicates core C2PA protocols fall short of "
        "end-to-end execution integrity without hardware TEE or ZK proofs."
    )

    def __init__(self, claim_generator: str = "CV-ASSURE v2 / M3 Provenance Module"):
        self.claim_generator = claim_generator

    def generate_manifest(
        self,
        title: str,
        asset_sha256: str,
        model_digest: str,
        signature_hex: str,
        receipt_sequence: int = 1,
    ) -> Dict[str, Any]:
        """Build C2PA Manifest JSON dict."""
        manifest = {
            "c2pa_version": "2.1.0",
            "title": title,
            "format": "application/json",
            "instance_id": f"urn:uuid:cv-assure-m3-{asset_sha256[:16]}",
            "claim_generator": self.claim_generator,
            "claim_generator_info": {
                "name": "CV-ASSURE M3 Inference Provenance Engine",
                "version": "2.0.0",
            },
            "security_caveat": self.C2PA_SECURITY_CAVEAT,
            "assertions": [
                {
                    "label": "c2pa.hash.data",
                    "data": {
                        "name": "input_image_sha256",
                        "hash": asset_sha256,
                        "alg": "sha256",
                    },
                },
                {
                    "label": "cvassure.model.binding",
                    "data": {
                        "model_weight_digest": model_digest,
                        "sequence_number": receipt_sequence,
                        "provenance_tier": "T2",
                    },
                },
                {
                    "label": "c2pa.actions",
                    "data": {
                        "actions": [
                            {
                                "action": "c2pa.inference",
                                "softwareAgent": "CV-ASSURE ONNX Pipeline",
                                "when": datetime.now(timezone.utc).isoformat(),
                            }
                        ]
                    },
                },
            ],
            "signature": {
                "algorithm": "ed25519",
                "sig_bytes_hex": signature_hex,
            },
        }
        return manifest

    def save_manifest(self, manifest_dict: Dict[str, Any], output_path: str | Path) -> Path:
        """Write manifest JSON to disk."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(manifest_dict, f, indent=2)
        return path
