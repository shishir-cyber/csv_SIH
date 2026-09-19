"""
coverage_generator.py
----------------------
Coverage Statement Generator for CV-ASSURE M3 Inference Provenance
(Clause 2.2.5 Compliance).

Generates coverage registry mapping supported, partial, and unsupported
attack threat models for M3 inference provenance.
"""

import json
from pathlib import Path
from typing import Dict, Any


class CoverageGenerator:
    """
    Generates Clause 2.2.5 coverage statement for M3.
    """

    def generate_coverage_statement(self) -> Dict[str, Any]:
        """Return structured coverage statement dictionary."""
        return {
            "module": "M3_INFERENCE_PROVENANCE",
            "clause": "2.2.3 / 2.2.5",
            "supported": [
                "receipt_payload_tampering",
                "record_replay_attack",
                "record_deletion_or_reordering",
                "hash_chain_discontinuity",
                "model_substitution",
                "merkle_checkpoint_inconsistency",
                "invalid_ed25519_signature",
            ],
            "partial": [
                "host_fabricated_results",  # Mitigated via 1% spot re-execution
            ],
            "unsupported": [
                "dishonest_signer_at_generation_time",  # Requires TEE or ZK proof
            ],
            "assumptions": [
                "reference_profile_is_clean",
                "signing_keys_uncompromised",
                "verifier_key_held_in_dual_custody",
            ],
        }

    def save_coverage_statement(self, output_path: str | Path) -> Path:
        """Save coverage statement to disk as JSON."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cov = self.generate_coverage_statement()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cov, f, indent=2)
        return path
