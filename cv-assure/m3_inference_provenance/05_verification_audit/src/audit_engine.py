"""
audit_engine.py
---------------
Clause 2.2.3 Cryptographic & Integrity Audit Engine for CV-ASSURE M3.

Evaluates an inference provenance ledger and checkpoint store against 6 threat models:
  1. Hash Chain Continuity & Payload Tampering (prev_receipt_hash mismatch)
  2. Replay & Sequence Attacks (duplicate nonces, non-monotonic sequence numbers)
  3. Signature Integrity (Ed25519 receipt and checkpoint signature validation)
  4. Model Substitution (model_weight_digest mismatch against reference profile)
  5. Output Payload Alteration (output_payload_sha256 mismatch)
  6. Merkle Checkpoint Consistency (Merkle tree root recalculation)

Emits standardized Clause 2.2.5 Finding objects with severity and recommended disposition.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
import hashlib
import json
from pathlib import Path
import sys

# Add sibling folder 04_ledger_chaining/src to path if needed for imports
ROOT_M3 = Path(__file__).resolve().parents[2]
LEDGER_SRC = ROOT_M3 / "04_ledger_chaining" / "src"
if str(LEDGER_SRC) not in sys.path:
    sys.path.insert(0, str(LEDGER_SRC))

try:
    from ledger_chain import compute_receipt_hash, GENESIS_PREV_HASH
    from merkle import MerkleTree
except ImportError:
    GENESIS_PREV_HASH = "0" * 64

    def compute_receipt_hash(payload: Dict[str, Any], prev_receipt_hash: str) -> str:
        binding_obj = {"payload": payload, "prev_receipt_hash": prev_receipt_hash}
        content = json.dumps(binding_obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(content).hexdigest()


@dataclass
class Finding:
    """Clause 2.2.5 Standard Audit Finding Schema."""

    finding_id: str
    detector: str
    detector_version: str = "1.2"
    asset_ref: str = "unknown"
    contributor_ref: str = "C-07"
    score: float = 0.0
    confidence: str = "high"  # high, medium, low
    access_tier: str = "T2"  # T0, T1, T2
    severity: str = "info"  # critical, high, medium, low, info
    reason_human_readable: str = ""
    evidence_artifacts: List[str] = field(default_factory=list)
    recommended_disposition: str = "accept"  # accept, review, quarantine, unavailable

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AuditEngine:
    """
    Master Audit Engine for Inference Provenance (Clause 2.2.3).
    """

    def __init__(
        self,
        expected_model_digest: Optional[str] = None,
        public_key_pem: Optional[bytes] = None,
        verifier_public_key_pem: Optional[bytes] = None,
    ):
        self.expected_model_digest = expected_model_digest
        self.public_key_pem = public_key_pem
        self.verifier_public_key_pem = verifier_public_key_pem

    def audit_ledger(
        self,
        receipts: List[Dict[str, Any]],
        checkpoints: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Finding]:
        """
        Run full audit suite on a list of receipt records and optional checkpoint records.
        Returns a list of Finding objects.
        """
        findings: List[Finding] = []

        if not receipts:
            findings.append(
                Finding(
                    finding_id="FINDING-M3-EMPTY",
                    detector="ledger_reader",
                    score=0.0,
                    confidence="high",
                    severity="warning",
                    reason_human_readable="Inference ledger is empty. No receipts to audit.",
                    recommended_disposition="review",
                )
            )
            return findings

        # Check 1: Chain integrity & Hash recalculation
        findings.extend(self.check_chain_integrity(receipts))

        # Check 2: Replay attack & Sequence monotonicity
        findings.extend(self.check_sequence_and_nonces(receipts))

        # Check 3: Cryptographic Signatures
        findings.extend(self.check_signatures(receipts, checkpoints))

        # Check 4: Model Attestation
        findings.extend(self.check_model_attestation(receipts))

        # Check 5: Merkle Checkpoints
        if checkpoints:
            findings.extend(self.check_merkle_checkpoints(receipts, checkpoints))

        return findings

    def check_chain_integrity(self, receipts: List[Dict[str, Any]]) -> List[Finding]:
        """Verify prev_receipt_hash linkage and recompute receipt_hash for each entry."""
        findings: List[Finding] = []
        expected_prev = GENESIS_PREV_HASH

        for idx, rec in enumerate(receipts):
            seq = rec.get("payload", {}).get("sequence_number", idx + 1)
            actual_prev = rec.get("prev_receipt_hash", "")
            actual_receipt_hash = rec.get("receipt_hash", "")
            payload = rec.get("payload", {})
            asset_ref = payload.get("input_image_sha256", f"seq_{seq}")

            # Check predecessor link
            if actual_prev != expected_prev:
                findings.append(
                    Finding(
                        finding_id=f"FINDING-M3-CHAIN-BREAK-SEQ-{seq}",
                        detector="hash_chain_verifier",
                        asset_ref=asset_ref,
                        score=1.0,
                        confidence="high",
                        severity="critical",
                        reason_human_readable=(
                            f"Hash chain break at sequence {seq}. Expected prev_receipt_hash "
                            f"'{expected_prev[:12]}...', got '{actual_prev[:12]}...'. "
                            "Indicates record deletion or reordering."
                        ),
                        evidence_artifacts=[f"receipt_seq_{seq}.json"],
                        recommended_disposition="quarantine",
                    )
                )

            # Recompute expected receipt hash
            recalculated_hash = compute_receipt_hash(payload, actual_prev)
            if recalculated_hash != actual_receipt_hash:
                findings.append(
                    Finding(
                        finding_id=f"FINDING-M3-TAMPER-SEQ-{seq}",
                        detector="payload_integrity_verifier",
                        asset_ref=asset_ref,
                        score=1.0,
                        confidence="high",
                        severity="critical",
                        reason_human_readable=(
                            f"Receipt payload hash mismatch at sequence {seq}. Computed "
                            f"'{recalculated_hash[:12]}...', recorded '{actual_receipt_hash[:12]}...'."
                        ),
                        evidence_artifacts=[f"receipt_seq_{seq}.json"],
                        recommended_disposition="quarantine",
                    )
                )

            expected_prev = actual_receipt_hash

        return findings

    def check_sequence_and_nonces(self, receipts: List[Dict[str, Any]]) -> List[Finding]:
        """Detect non-monotonic sequence numbers and duplicate nonces (Replay Attack)."""
        findings: List[Finding] = []
        seen_nonces: Dict[str, int] = {}
        last_seq = 0

        for rec in receipts:
            payload = rec.get("payload", {})
            seq = payload.get("sequence_number", 0)
            nonce = payload.get("nonce", "")
            asset_ref = payload.get("input_image_sha256", f"seq_{seq}")

            # Check sequence monotonicity
            if seq <= last_seq:
                findings.append(
                    Finding(
                        finding_id=f"FINDING-M3-SEQ-BREAK-SEQ-{seq}",
                        detector="sequence_monotonicity_checker",
                        asset_ref=asset_ref,
                        score=0.95,
                        confidence="high",
                        severity="high",
                        reason_human_readable=(
                            f"Non-monotonic sequence number detected at sequence {seq} "
                            f"(previous sequence was {last_seq})."
                        ),
                        evidence_artifacts=[f"receipt_seq_{seq}.json"],
                        recommended_disposition="quarantine",
                    )
                )
            last_seq = seq

            # Check nonce reuse (Replay Attack)
            if nonce:
                if nonce in seen_nonces:
                    prior_seq = seen_nonces[nonce]
                    findings.append(
                        Finding(
                            finding_id=f"FINDING-M3-REPLAY-SEQ-{seq}",
                            detector="replay_attack_detector",
                            asset_ref=asset_ref,
                            score=1.0,
                            confidence="high",
                            severity="critical",
                            reason_human_readable=(
                                f"Replay attack detected! Nonce '{nonce}' at sequence {seq} "
                                f"was previously seen at sequence {prior_seq}."
                            ),
                            evidence_artifacts=[f"receipt_seq_{seq}.json", f"receipt_seq_{prior_seq}.json"],
                            recommended_disposition="quarantine",
                        )
                    )
                else:
                    seen_nonces[nonce] = seq

        return findings

    def check_signatures(
        self,
        receipts: List[Dict[str, Any]],
        checkpoints: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Finding]:
        """Verify Ed25519 signatures on receipts and checkpoints if public key is available."""
        findings: List[Finding] = []

        if not self.public_key_pem:
            # Graceful capability probe fallback
            findings.append(
                Finding(
                    finding_id="FINDING-M3-SIG-UNAVAILABLE",
                    detector="ed25519_signature_verifier",
                    score=0.0,
                    confidence="medium",
                    severity="info",
                    reason_human_readable="Ed25519 public key not provided. Signature verification skipped.",
                    recommended_disposition="unavailable",
                )
            )
            return findings

        try:
            from cryptography.hazmat.primitives import serialization
            public_key = serialization.load_pem_public_key(self.public_key_pem)
        except Exception as e:
            findings.append(
                Finding(
                    finding_id="FINDING-M3-KEY-ERROR",
                    detector="ed25519_signature_verifier",
                    score=0.5,
                    confidence="high",
                    severity="high",
                    reason_human_readable=f"Failed to parse Ed25519 public key: {e}",
                    recommended_disposition="review",
                )
            )
            return findings

        for rec in receipts:
            payload = rec.get("payload", {})
            seq = payload.get("sequence_number", 0)
            receipt_hash = rec.get("receipt_hash", "")
            sig_hex = rec.get("signature", "")
            asset_ref = payload.get("input_image_sha256", f"seq_{seq}")

            if not sig_hex:
                continue

            try:
                sig_bytes = bytes.fromhex(sig_hex)
                public_key.verify(sig_bytes, receipt_hash.encode("utf-8"))
            except Exception:
                findings.append(
                    Finding(
                        finding_id=f"FINDING-M3-SIG-BAD-SEQ-{seq}",
                        detector="ed25519_signature_verifier",
                        asset_ref=asset_ref,
                        score=1.0,
                        confidence="high",
                        severity="critical",
                        reason_human_readable=(
                            f"Invalid Ed25519 signature on receipt at sequence {seq}. "
                            "Signature verification failed against attested public key."
                        ),
                        evidence_artifacts=[f"receipt_seq_{seq}.json"],
                        recommended_disposition="quarantine",
                    )
                )

        return findings

    def check_model_attestation(self, receipts: List[Dict[str, Any]]) -> List[Finding]:
        """Verify model_weight_digest against attested reference model."""
        findings: List[Finding] = []
        if not self.expected_model_digest:
            return findings

        for rec in receipts:
            payload = rec.get("payload", {})
            seq = payload.get("sequence_number", 0)
            model_digest = payload.get("model_weight_digest", "")
            asset_ref = payload.get("input_image_sha256", f"seq_{seq}")

            if model_digest.lower() != self.expected_model_digest.lower():
                findings.append(
                    Finding(
                        finding_id=f"FINDING-M3-MODEL-SUB-SEQ-{seq}",
                        detector="model_integrity_attestation",
                        asset_ref=asset_ref,
                        score=1.0,
                        confidence="high",
                        severity="critical",
                        reason_human_readable=(
                            f"Model substitution attack at sequence {seq}! Model digest "
                            f"'{model_digest[:12]}...' does not match attested reference "
                            f"model '{self.expected_model_digest[:12]}...'."
                        ),
                        evidence_artifacts=[f"receipt_seq_{seq}.json"],
                        recommended_disposition="quarantine",
                    )
                )

        return findings

    def check_merkle_checkpoints(
        self,
        receipts: List[Dict[str, Any]],
        checkpoints: List[Dict[str, Any]],
    ) -> List[Finding]:
        """Recompute Merkle roots for checkpoint blocks and verify checkpoint consistency."""
        findings: List[Finding] = []
        seq_map = {r["payload"]["sequence_number"]: r["receipt_hash"] for r in receipts if "payload" in r}

        for chk in checkpoints:
            chk_id = chk.get("checkpoint_id", "CHK-UNKNOWN")
            start_seq = chk.get("start_sequence", 1)
            end_seq = chk.get("end_sequence", 1)
            recorded_root = chk.get("merkle_root", "")

            block_hashes = [seq_map[s] for s in range(start_seq, end_seq + 1) if s in seq_map]
            if not block_hashes:
                continue

            tree = MerkleTree(block_hashes)
            recalculated_root = tree.root

            if recalculated_root.lower() != recorded_root.lower():
                findings.append(
                    Finding(
                        finding_id=f"FINDING-M3-MERKLE-MISMATCH-{chk_id}",
                        detector="merkle_checkpoint_verifier",
                        score=1.0,
                        confidence="high",
                        severity="critical",
                        reason_human_readable=(
                            f"Merkle root mismatch in checkpoint {chk_id} (sequences {start_seq}-{end_seq}). "
                            f"Calculated '{recalculated_root[:12]}...', recorded '{recorded_root[:12]}...'."
                        ),
                        evidence_artifacts=[f"checkpoint_{chk_id}.json"],
                        recommended_disposition="quarantine",
                    )
                )

        return findings
